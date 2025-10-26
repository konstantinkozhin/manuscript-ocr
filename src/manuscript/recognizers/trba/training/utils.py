import random
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from numba import njit

from .vis import visualize_decoding


def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    global_step,
    best_val_loss,
    best_val_acc,
    itos,
    stoi,
    config,
    log_dir,
):
    ckpt = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "itos": itos,
        "stoi": stoi,
        "config": config,
        "log_dir": log_dir,
    }
    torch.save(ckpt, path)


def save_weights(path, model):
    torch.save(model.state_dict(), path)


def load_checkpoint(
    path,
    model=None,
    optimizer=None,
    scheduler=None,
    scaler=None,
    map_location="auto",
    strict: bool = True,
):
    if map_location == "auto":
        map_location = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_obj = torch.load(path, map_location=map_location)

    if isinstance(ckpt_obj, dict) and "model_state" in ckpt_obj:
        model_state = ckpt_obj["model_state"]
        metadata = ckpt_obj
    else:
        model_state = ckpt_obj
        metadata = {"model_state": model_state}

    if model is not None:
        model.load_state_dict(model_state, strict=strict)

    if optimizer is not None and metadata.get("optimizer_state") is not None:
        optimizer.load_state_dict(metadata["optimizer_state"])
    if scheduler is not None and metadata.get("scheduler_state") is not None:
        scheduler.load_state_dict(metadata["scheduler_state"])
    if scaler is not None and metadata.get("scaler_state") is not None:
        scaler.load_state_dict(metadata["scaler_state"])
    return metadata


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def greedy_decode(logits: torch.Tensor) -> torch.Tensor:
    if logits.dim() != 3:
        raise ValueError(f"Expected logits with shape (B, T, C), got {logits.shape}")
    return logits.argmax(dim=-1)


@njit(cache=True)
def beam_search_numba(
    log_probs,
    beam_size,
    eos_id,
    length_penalty,
    normalize_by_length,
    diverse_groups,
    diversity_strength,
):
    T, C = log_probs.shape
    beams = [(0.0, np.empty(0, dtype=np.int32))]
    group_size = max(1, beam_size // max(diverse_groups, 1))

    for t in range(T):
        all_candidates = []
        token_penalties = np.zeros(C, dtype=np.float32)

        for g in range(diverse_groups):
            group_beams = beams[g * group_size : (g + 1) * group_size]
            group_candidates = []

            for s, seq in group_beams:
                if eos_id >= 0 and seq.size > 0 and seq[-1] == eos_id:
                    group_candidates.append((s, seq))
                    continue

                row = log_probs[t].copy()
                if diversity_strength > 0:
                    row -= diversity_strength * token_penalties

                top_idx = np.argpartition(row, -group_size)[-group_size:]
                top_vals = row[top_idx]

                for i in range(group_size):
                    new_seq = np.concatenate(
                        (seq, np.array([top_idx[i]], dtype=np.int32))
                    )
                    new_score = s + top_vals[i]
                    group_candidates.append((new_score, new_seq))

                for tok in top_idx:
                    token_penalties[tok] += 1.0

            all_candidates.extend(group_candidates)

        all_candidates.sort(key=lambda x: x[0], reverse=True)
        beams = all_candidates[:beam_size]

    best_score = -1e9
    best_seq = np.empty(0, dtype=np.int32)
    for s, seq in beams:
        L = max(len(seq), 1)
        score = s
        if length_penalty > 0:
            score = s / (L**length_penalty)
        if normalize_by_length:
            score /= L
        if score > best_score:
            best_score = score
            best_seq = seq

    return best_seq


def beam_search_decode_torch(
    logits: torch.Tensor,
    beam_size: int = 5,
    eos_id: Optional[int] = None,
    pad_id: int = 0,
    temperature: float = 1.0,
    length_penalty: float = 0.0,
    normalize_by_length: bool = True,
    diverse_groups: int = 1,
    diversity_strength: float = 0.0,
    vis: bool = False,
    itos: Optional[List[str]] = None,
    lm_model=None,
    lm_tokenizer=None,
    lm_weight: float = 0.0,
    # --- новые параметры ---
    noise_level: float = 0.3,  # уровень шума для стохастичности
    topk_sampling_steps: int = 3,  # сколько первых шагов делаем sampling
    topk: int = 5,  # размер top-k при sampling
    coverage_penalty_weight: float = 0.1,  # штраф за повтор
    expand_beam_steps: int = 3,  # сколько шагов расширяем beam в начале
    seed: Optional[int] = None,
):
    """
    Улучшенный beam search:
    - diverse groups
    - стохастичность в начале
    - top-k sampling warmup
    - coverage penalty
    - расширение beam на первых шагах
    """
    if logits.dim() != 3:
        raise ValueError(f"Expected (B, T, C), got {tuple(logits.shape)}")

    B, T, C = logits.shape
    device = logits.device

    # Добавляем шум для стохастичности
    log_probs = F.log_softmax(logits / temperature, dim=-1)
    if noise_level > 0:
        noise = torch.randn_like(log_probs) * noise_level
        log_probs = log_probs + noise

    results = []

    for b in range(B):
        beams = [(0.0, [])]
        beam_history = [] if vis else None

        for t in range(T):
            # Расширяем beam в начале
            current_beam_size = beam_size * 2 if t < expand_beam_steps else beam_size

            all_candidates = []
            group_size = max(1, current_beam_size // diverse_groups)
            token_penalties = torch.zeros(C, device=device)

            for g in range(diverse_groups):
                group_beams = beams[g * group_size : (g + 1) * group_size]

                for score, seq in group_beams:
                    # Если достигнут EOS — не продолжаем
                    if eos_id is not None and len(seq) > 0 and seq[-1] == eos_id:
                        all_candidates.append((score, seq))
                        continue

                    step_log_probs = log_probs[b, t]

                    # diversity penalty
                    if diversity_strength > 0:
                        step_log_probs = (
                            step_log_probs - diversity_strength * token_penalties
                        )

                    # --- top-k sampling warmup ---
                    if t < topk_sampling_steps:
                        topk_vals, topk_ids = torch.topk(step_log_probs, topk)
                        probs = torch.softmax(topk_vals, dim=-1)
                        sampled_id = int(torch.multinomial(probs, 1))
                        chosen_id = int(topk_ids[sampled_id])
                        new_seq = seq + [chosen_id]
                        new_score = score + float(step_log_probs[chosen_id].item())
                        all_candidates.append((new_score, new_seq))
                        token_penalties[chosen_id] += 1
                        continue

                    # --- стандартный beam step ---
                    topk_log_probs, topk_ids = torch.topk(step_log_probs, group_size)
                    for k in range(group_size):
                        new_seq = seq + [int(topk_ids[k])]
                        new_score = score + float(topk_log_probs[k].item())
                        all_candidates.append((new_score, new_seq))
                    token_penalties[topk_ids] += 1

            # coverage penalty: штраф за повтор
            def coverage_penalty(seq):
                if not seq:
                    return 0.0
                unique = len(set(seq))
                return -coverage_penalty_weight * (1 - unique / len(seq))

            scored_candidates = []
            for s, seq in all_candidates:
                s += coverage_penalty(seq)
                scored_candidates.append((s, seq))

            # сортировка
            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            kept_candidates = scored_candidates[:current_beam_size]
            kept_set = {tuple(seq) for _, seq in kept_candidates}

            if vis:
                beam_history.append(
                    [
                        {"seq": seq, "score": s, "kept": tuple(seq) in kept_set}
                        for s, seq in scored_candidates
                    ]
                )

            beams = kept_candidates

        # финальное нормирование и выбор
        final_beams = []
        for s, seq in beams:
            length = max(len(seq), 1)
            score = s / (length**length_penalty) if length_penalty > 0 else s
            if normalize_by_length:
                score /= length
            final_beams.append((score, seq))

        best_seq = max(final_beams, key=lambda x: x[0])[1]
        if eos_id is not None and eos_id in best_seq:
            best_seq = best_seq[: best_seq.index(eos_id)]

        results.append(best_seq)
        # if vis and itos is not None:
        #     visualize_decoding(log_probs[b], beam_history, itos)

    # Паддинг для финального батча
    max_len = max(len(s) for s in results)
    padded = torch.full(
        (len(results), max_len), pad_id, dtype=torch.long, device=device
    )
    for i, seq in enumerate(results):
        padded[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)

    return padded


def beam_search_decode(
    logits: torch.Tensor,
    beam_size: int = 5,
    eos_id: Optional[int] = -1,
    pad_id: int = 0,
    temperature: float = 1.0,
    length_penalty: float = 0.0,
    normalize_by_length: bool = True,
    diverse_groups: int = 1,
    diversity_strength: float = 0.0,
    vis: bool = False,
    itos: Optional[List[str]] = None,
    lm_model=None,
    lm_tokenizer=None,
    lm_weight: float = 0.0,
    noise_level: float = 0.3,
    topk_sampling_steps: int = 3,
    topk: int = 5,
    coverage_penalty_weight: float = 0.1,
    expand_beam_steps: int = 3,
    seed: Optional[int] = None,  # <---- добавлено
):
    if vis:
        return beam_search_decode_torch(
            logits=logits,
            beam_size=beam_size,
            eos_id=eos_id,
            pad_id=pad_id,
            temperature=temperature,
            length_penalty=length_penalty,
            normalize_by_length=normalize_by_length,
            diverse_groups=diverse_groups,
            diversity_strength=diversity_strength,
            vis=vis,
            itos=itos,
            lm_model=lm_model,
            lm_tokenizer=lm_tokenizer,
            lm_weight=lm_weight,
            noise_level=noise_level,
            topk_sampling_steps=topk_sampling_steps,
            topk=topk,
            coverage_penalty_weight=coverage_penalty_weight,
            expand_beam_steps=expand_beam_steps,
            seed=seed,  # <---- пробрасываем
        )
    else:
        if logits.dim() != 3:
            raise ValueError(f"Expected (B, T, C), got {tuple(logits.shape)}")

        B, T, C = logits.shape
        device = logits.device
        log_probs = F.log_softmax(logits / temperature, dim=-1)

        results = []

        for b in range(B):
            logp_np = log_probs[b].detach().cpu().numpy().astype(np.float32)

            best_seq = beam_search_numba(
                logp_np,
                beam_size,
                eos_id if eos_id is not None else -1,
                length_penalty,
                normalize_by_length,
                diverse_groups,
                diversity_strength,
            )

            seq = best_seq.tolist()

            if eos_id is not None and eos_id in seq:
                seq = seq[: seq.index(eos_id)]
            results.append(seq)

        max_len = max(len(s) for s in results)
        padded = torch.full(
            (len(results), max_len),
            pad_id,
            dtype=torch.long,
            device=device,
        )
        for i, seq in enumerate(results):
            padded[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)

        return padded


from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
import math


class HF_LMScorer:
    def __init__(
        self, model_name="sberbank-ai/rugpt3small_based_on_gpt2", device="cuda"
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.device = device
        self.model.eval()

    def start(self):
        # Начальное состояние: пустой префикс
        return torch.tensor([self.tokenizer.bos_token_id]).to(self.device)

    def score(self, state, ch: str):
        # Добавляем символ (или токен)
        if ch == "":
            return 0.0, state
        tokens = self.tokenizer.encode(ch, add_special_tokens=False)
        if not tokens:
            return 0.0, state

        input_ids = torch.cat([state, torch.tensor(tokens, device=self.device)], dim=-1)
        with torch.no_grad():
            outputs = self.model(input_ids.unsqueeze(0))
            logits = outputs.logits[0, -2, :]  # логиты предыдущего токена
            probs = F.log_softmax(logits, dim=-1)
            last_token = tokens[0]
            logp = float(probs[last_token].item())

        return logp, input_ids


def decode_predictions(
    logits: torch.Tensor,
    mode: str = "beam",
    beam_size: int = 5,
    eos_id: Optional[int] = None,
    pad_id: int = 0,
    temperature: float = 1.0,
    length_penalty: float = 0.0,
    normalize_by_length: bool = True,
    diverse_groups: int = 1,
    diversity_strength: float = 0.0,
    vis: bool = False,
    itos: Optional[List[str]] = None,
    lm_scorer: Optional[HF_LMScorer] = None,
    lm_weight: float = 0.0,
    noise_level: float = 0.3,
    topk_sampling_steps: int = 3,
    topk: int = 5,
    coverage_penalty_weight: float = 0.1,
    expand_beam_steps: int = 3,
    seed: Optional[int] = None,
):
    mode = mode.lower()
    if mode not in {"beam", "greedy"}:
        raise ValueError(f"Unknown decode mode: {mode}")

    if mode == "greedy":
        preds = greedy_decode(logits)
        if vis and itos is not None:
            B, T, C = logits.shape
            log_probs = F.log_softmax(logits, dim=-1)
            for b in range(B):
                seq = preds[b].tolist()
                beam_history = []
                stopped = False
                for t in range(T):
                    if stopped:
                        break

                    token_id = seq[t]
                    beam_history.append(
                        [
                            {
                                "seq": seq[: t + 1],
                                "score": float(log_probs[b, t, token_id].item()),
                                "kept": True,
                            }
                        ]
                    )
                    if eos_id is not None and token_id == eos_id:
                        stopped = True

                # visualize_decoding(log_probs[b], beam_history, itos)
        return preds

    elif mode == "beam":
        preds = beam_search_decode(
            logits=logits,
            beam_size=beam_size,
            eos_id=eos_id,
            pad_id=pad_id,
            temperature=temperature,
            length_penalty=length_penalty,
            normalize_by_length=normalize_by_length,
            diverse_groups=diverse_groups,
            diversity_strength=diversity_strength,
            vis=vis,
            itos=itos,
            lm_model=lm_scorer.model if lm_scorer is not None else None,
            lm_tokenizer=lm_scorer.tokenizer if lm_scorer is not None else None,
            lm_weight=lm_weight,
        )
        return preds
