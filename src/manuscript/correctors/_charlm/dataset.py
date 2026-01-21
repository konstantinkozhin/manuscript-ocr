import random
import torch
from torch.utils.data import Dataset
from .utils import encode_str, choose_spans


class NgramDataset(Dataset):
    def __init__(self, text_path, c2i, max_len, span_min, span_max, spans_min, spans_max,
                 ngram_probs={1: 0.4, 2: 0.4, 3: 0.2}, steps=100_000, mask_prob=0.15):
        self.c2i = c2i
        self.max_len = max_len
        self.span_min, self.span_max = span_min, span_max
        self.spans_min, self.spans_max = spans_min, spans_max
        self.ngram_probs = ngram_probs
        self.steps = steps
        self.mask_prob = mask_prob
        with open(text_path, encoding="utf-8") as f:
            self.tokens = f.read().split()
        if len(self.tokens) < 3:
            raise ValueError("Text too short")

    def __len__(self):
        return self.steps

    def _sample(self):
        n = random.choices(list(self.ngram_probs.keys()), list(self.ngram_probs.values()))[0]
        n = min(n, len(self.tokens))
        i = random.randint(0, len(self.tokens) - n)
        return " ".join(self.tokens[i:i+n])

    def __getitem__(self, _):
        text = self._sample()
        x = encode_str(text, self.c2i, self.max_len)
        L = min(len(text), self.max_len)
        
        if random.random() < self.mask_prob:
            mask_pos = choose_spans(L, self.span_min, self.span_max, self.spans_min, self.spans_max)
        else:
            mask_pos = []
        
        if not mask_pos and L > 0:
            mask_pos = [random.randint(0, L - 1)]
        
        y = [-100] * self.max_len
        for p in mask_pos:
            if p < self.max_len:
                y[p] = x[p]
                x[p] = self.c2i["<MASK>"]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


class PairsDataset(Dataset):
    def __init__(self, pairs, c2i, max_len, min_len=4):
        self.pairs = pairs
        self.c2i = c2i
        self.max_len = max_len
        self.min_len = min_len
        self.pad, self.mask, self.unk = c2i["<PAD>"], c2i["<MASK>"], c2i["<UNK>"]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        inc, cor = self.pairs[idx]
        inc, cor = inc[:self.max_len], cor[:self.max_len]
        
        x, y = [], []
        for ci, cc in zip(inc, cor):
            if ci != cc:
                x.append(self.mask)
                y.append(self.c2i.get(cc, self.unk))
            else:
                x.append(self.c2i.get(ci, self.unk))
                y.append(-100)
        
        pad_len = self.max_len - len(x)
        x += [self.pad] * pad_len
        y += [-100] * pad_len
        return torch.tensor(x), torch.tensor(y)
