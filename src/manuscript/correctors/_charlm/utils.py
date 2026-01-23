import json
import random
import csv
import re
import Levenshtein
import torch
from datetime import datetime
from statistics import mean
from tqdm import tqdm


class CharLMCorrector:
    def __init__(self, model, c2i, i2c, device, max_len, mask_threshold=0.2,
                 apply_threshold=0.95, max_edits=3, lexicon=None, min_word_len=4):
        self.model, self.c2i, self.i2c = model, c2i, i2c
        self.device, self.max_len = device, max_len
        self.mask_threshold, self.apply_threshold = mask_threshold, apply_threshold
        self.max_edits = max_edits
        self.min_word_len = min_word_len
        self.lexicon = frozenset(w.lower() for w in lexicon) if lexicon else None
        self._word_pattern =  re.compile(r'(\w+)|(\W+)', re.UNICODE)

    def _tokenize(self, text):
        tokens = []
        for m in self._word_pattern.finditer(text):
            word_part, other_part = m.groups()
            if word_part:
                tokens.append((word_part, True))
            else:
                tokens.append((other_part, False))
        return tokens

    def correct_word(self, text, return_trace=False, return_p_cur=False):
        tokens = self._tokenize(text)
        result_parts = []
        all_traces = []
        all_confidences = []
        
        for token, is_word in tokens:
            if not is_word:
                result_parts.append(token)
                continue
            
            word_lower = token.lower()
            
            if len(word_lower) < self.min_word_len:
                result_parts.append(token)
                continue
            
            if self.lexicon and word_lower in self.lexicon:
                result_parts.append(token)
                continue
            
            corrected, trace, confidences = self._correct_single_word(word_lower)
            
            if corrected != word_lower:
                corrected = self._restore_case(token, corrected)
            else:
                corrected = token
            
            result_parts.append(corrected)
            all_traces.extend(trace)
            all_confidences.extend(confidences)
        
        result = "".join(result_parts)
        
        if return_trace or return_p_cur:
            return result, all_traces, all_confidences
        return result

    def _correct_single_word(self, word):
        return reconstruct_word(
            self.model, word, self.c2i, self.i2c, self.device, self.max_len,
            self.mask_threshold, self.apply_threshold, self.max_edits,
            return_trace=True, return_p_cur=True, lexicon=self.lexicon
        )

    def _restore_case(self, original, corrected):
        result = []
        for i, ch in enumerate(corrected):
            if i < len(original) and original[i].isupper():
                result.append(ch.upper())
            else:
                result.append(ch)
        return "".join(result)


def evaluate_ocr_confidence(model, eval_pairs, c2i, device, max_len):
    model.eval()
    correct_confs, incorrect_confs = [], []
    unk, mask = c2i["<UNK>"], c2i["<MASK>"]

    for inc, cor in tqdm(eval_pairs, desc="OCR confidence", leave=False):
        L = min(len(inc), max_len)
        for i in range(L):
            ids = [(c2i.get(inc[j], unk) if j != i else mask) for j in range(L)]
            ids += [c2i["<PAD>"]] * (max_len - L)
            x = torch.tensor(ids, device=device).unsqueeze(0)
            with torch.no_grad():
                probs = torch.softmax(model(x)[0, i], dim=-1)
            conf = probs[c2i.get(inc[i], unk)].item()
            (correct_confs if inc[i] == cor[i] else incorrect_confs).append(conf)

    def stats(xs):
        if not xs:
            return {}
        xs = sorted(xs)
        n = len(xs)
        return {"n": n, "mean": sum(xs)/n, "p25": xs[n//4], "median": xs[n//2], "p75": xs[3*n//4]}

    return stats(correct_confs), stats(incorrect_confs)


def reconstruct_word(model, word, c2i, i2c, device, max_len, mask_threshold,
                     apply_threshold, max_edits, return_trace=False, return_p_cur=False, 
                     lexicon=None):
    model.eval()
    original_chars = list(word[:max_len])
    chars = original_chars.copy()
    L = len(chars)
    if L == 0:
        return (word, [], []) if (return_trace or return_p_cur) else word

    unk, mask, pad = c2i["<UNK>"], c2i["<MASK>"], c2i["<PAD>"]
    batch = []
    for i in range(L):
        ids = [(c2i.get(ch, unk) if j != i else mask) for j, ch in enumerate(chars)]
        ids += [pad] * (max_len - len(ids))
        batch.append(ids)

    with torch.no_grad():
        probs = torch.softmax(model(torch.tensor(batch, device=device)), dim=-1)

    confidences = [(i, probs[i, i, c2i.get(chars[i], unk)].item(), probs[i, i]) for i in range(L)]
    candidates = sorted([(i, p, v) for i, p, v in confidences if p < mask_threshold], key=lambda x: x[1])

    trace, edits = [], 0
    
    for i, p_cur, prob_vec in candidates:
        if edits >= max_edits:
            break
        best_id = prob_vec.argmax().item()
        best_p, best_char = prob_vec[best_id].item(), i2c[best_id]
        
        if best_char in ("<UNK>", "<PAD>", "<MASK>"):
            applied = False
        elif best_char == chars[i]:
            applied = False
        elif best_p < apply_threshold:
            applied = False
        elif best_char.lower() != best_char or chars[i].lower() != chars[i]:
            if best_char.lower() == chars[i].lower():
                applied = False
            else:
                test_chars = chars.copy()
                test_chars[i] = best_char
                test_word = "".join(test_chars)
                if lexicon and test_word in lexicon:
                    applied = True
                elif lexicon:
                    applied = False
                else:
                    applied = True
        else:
            test_chars = chars.copy()
            test_chars[i] = best_char
            test_word = "".join(test_chars)
            if lexicon and test_word in lexicon:
                applied = True
            elif lexicon:
                applied = False
            else:
                applied = True
        
        trace.append({"pos": i, "old": chars[i], "best": best_char, "p_cur": round(p_cur, 4),
                      "p_best": round(best_p, 4), "applied": applied})
        if applied:
            chars[i] = best_char
            edits += 1

    result = "".join(chars)
    return (result, trace, confidences) if (return_trace or return_p_cur) else result


def cer(pred, target):
    return Levenshtein.distance(pred, target) / len(target) if target else 0.0


def evaluate_ocr_with_cer(corrector, eval_pairs, csv_path=None):
    if not eval_pairs:
        return {"exact_match": 0, "cer_before": 0, "cer_after": 0, "delta": 0,
                "improved_pct": 0, "worsened_pct": 0, "unchanged_pct": 0}

    exact, cer_before, cer_after, rows = 0, [], [], []
    for inc, cor in tqdm(eval_pairs, desc="OCR CER eval", leave=False):
        cb = cer(inc, cor)
        pred, trace, confidences = corrector.correct_word(inc, return_trace=True, return_p_cur=True)
        ca = cer(pred, cor)
        label = "improved" if ca < cb else ("worsened" if ca > cb else "unchanged")
        if pred == cor:
            exact += 1
        p_cur_list = [p for (_, p, _) in confidences]
        rows.append({"incorrect": inc, "predicted": pred, "correct": cor, "label": label,
                     "p_cur": json.dumps([round(p, 4) for p in p_cur_list]),
                     "trace": json.dumps(trace, ensure_ascii=False) if trace else "[]"})
        cer_before.append(cb)
        cer_after.append(ca)

    if csv_path:
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["incorrect", "predicted", "correct", "label", "p_cur", "trace"])
            w.writeheader()
            w.writerows(rows)

    n = len(eval_pairs)
    return {"exact_match": exact / n, "cer_before": mean(cer_before), "cer_after": mean(cer_after),
            "delta": mean(cer_after) - mean(cer_before),
            "improved_pct": 100 * sum(r["label"] == "improved" for r in rows) / n,
            "worsened_pct": 100 * sum(r["label"] == "worsened" for r in rows) / n,
            "unchanged_pct": 100 * sum(r["label"] == "unchanged" for r in rows) / n}


def log_random_examples(model, batch, logits, c2i, i2c, logger, k=3):
    x, y = batch
    preds = logits.argmax(dim=-1)
    B, T = x.shape
    pad_id, unk_id = c2i["<PAD>"], c2i["<UNK>"]
    
    valid_indices = []
    for b in range(B):
        has_unk = any(x[b, i].item() == unk_id for i in range(T) if x[b, i].item() != pad_id)
        has_mask = any(y[b, i].item() != -100 for i in range(T))
        if not has_unk and has_mask:
            valid_indices.append(b)
    
    if not valid_indices:
        return
    
    logger.log("---- MLM examples ----")
    for b in random.sample(valid_indices, min(k, len(valid_indices))):
        inp, tgt, prd = [], [], []
        for i in range(T):
            if x[b, i].item() == pad_id:
                break
            if y[b, i].item() != -100:
                inp.append("<MASK>")
                tgt.append(i2c[y[b, i].item()])
                prd.append(i2c[preds[b, i].item()])
            else:
                ch = i2c[x[b, i].item()]
                inp.append(ch)
                tgt.append(ch)
                prd.append(ch)
        logger.log(f"INPUT   : {''.join(inp)}")
        logger.log(f"TARGET  : {''.join(tgt)}")
        logger.log(f"PREDICT : {''.join(prd)}")
        logger.log("")


class Logger:
    def __init__(self, path=None):
        self.path = path
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"=== Training started: {datetime.now()} ===\n")

    def log(self, msg):
        print(msg)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")


def build_vocab(words, include_space=True):
    chars = set()
    for w in words:
        chars.update(w)
    if include_space:
        chars.add(" ")
    chars = ["<PAD>", "<MASK>", "<UNK>"] + sorted(chars)
    c2i = {c: i for i, c in enumerate(chars)}
    return c2i, {i: c for c, i in c2i.items()}, chars


def encode_str(s, c2i, max_len):
    ids = [c2i.get(ch, c2i["<UNK>"]) for ch in s[:max_len]]
    return ids + [c2i["<PAD>"]] * (max_len - len(ids))


def choose_spans(L, span_min, span_max, spans_min, spans_max):
    if L <= 3:
        return []
    positions = set()
    for _ in range(random.randint(spans_min, spans_max)):
        span_len = random.randint(span_min, span_max)
        start_max = min(L - 2, L - 1 - span_len)
        if start_max < 1:
            continue
        start = random.randint(1, start_max)
        for p in range(start, start + span_len):
            if 1 <= p <= L - 2:
                positions.add(p)
    return sorted(positions)


def masked_accuracy(logits, targets):
    with torch.no_grad():
        mask = targets != -100
        if mask.sum().item() == 0:
            return 0.0
        return (logits.argmax(dim=-1)[mask] == targets[mask]).float().mean().item()


def load_allowed_chars(charset_path):
    allowed = set()
    with open(charset_path, encoding="utf-8") as f:
        for line in f:
            ch = line.rstrip("\n\r")
            if len(ch) == 1 and ch.isalpha():
                allowed.add(ch)
    return allowed


def filter_words(words, min_len=1, allowed_chars=None):
    if allowed_chars is None:
        return words
    result = []
    for w in words:
        cleaned = "".join(ch for ch in w if ch in allowed_chars)
        if len(cleaned) >= min_len:
            result.append(cleaned)
    return result
