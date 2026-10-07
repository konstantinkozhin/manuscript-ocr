"""Batched deterministic beam search, independent of the decoder cache format."""

import numpy as np


def generate_batch(hidden, generation, decode, banned):
    """Decode all samples together; cache rows follow surviving beam parents."""
    g = generation
    beams, limit = g['num_beams'], g['max_length']
    start, eos = int(g['decoder_start_token_id']), g['eos_token_id']
    eos = {int(eos)} if isinstance(eos, int) else set(eos)
    states = [{'active': [([start], 0.0)], 'finished': []} for _ in hidden]
    penalty = float(g['length_penalty'])
    cache = None
    while any(s['active'] for s in states):
        owners, active = [], []
        for owner, state in enumerate(states):
            owners.extend([owner] * len(state['active']))
            active.extend(state['active'])
        if len(active[0][0]) >= limit:
            break
        ids = np.array([item[0] for item in active], np.int64)
        logits, present = decode(ids, hidden[np.asarray(owners)], cache)
        logits = np.asarray(logits, np.float64)
        if (logits.ndim != 3 or logits.shape[0] != len(active)
                or not logits.shape[1] or not logits.shape[2] or not np.isfinite(logits).all()):
            raise ValueError('Decoder emitted invalid or nonfinite logits')
        logits = logits[:, -1]
        logits -= logits.max(axis=1, keepdims=True)
        scores = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
        for i, (tokens, score) in enumerate(active):
            scores[i, banned(tokens, g['no_repeat_ngram_size'])] = -np.inf
            scores[i] += score
        parents = []
        offset = 0
        for state in states:
            count = len(state['active'])
            if not count:
                continue
            flat = scores[offset:offset + count].ravel()
            top = np.argsort(-flat, kind='stable')[:max(2, 1 + len(eos)) * beams]
            next_active, next_parents = [], []
            generated_len = len(state['active'][0][0])
            for rank, index in enumerate(top):
                beam, token = divmod(int(index), scores.shape[1])
                score = float(flat[index])
                if not np.isfinite(score):
                    continue
                tokens = state['active'][beam][0] + [token]
                if token in eos:
                    if rank < beams:
                        state['finished'].append((score / generated_len**penalty, tokens, score))
                        state['finished'] = sorted(state['finished'], key=lambda item: item[0], reverse=True)[:beams]
                else:
                    next_active.append((tokens, score))
                    next_parents.append(offset + beam)
                if len(next_active) == beams:
                    break
            state['active'] = next_active
            if len(state['finished']) >= beams:
                worst = state['finished'][-1][0]
                best = float(flat[top[0]])
                early = g['early_stopping']
                bound = limit - 1 if early == 'never' and penalty > 0 else generated_len
                if early is True or worst >= best / bound**penalty:
                    state['active'] = []
                    next_parents = []
            parents.extend(next_parents)
            offset += count
        if present is not None:
            cache = {name: value[np.asarray(parents, dtype=np.int64)] for name, value in present.items()}
    results = []
    for state in states:
        for tokens, score in state['active']:
            state['finished'].append((score / max(1, len(tokens) - 1)**penalty, tokens, score))
        if not state['finished']:
            raise RuntimeError('Decoder produced no hypotheses')
        _, tokens, score = max(state['finished'], key=lambda item: item[0])
        results.append((tokens, float(np.exp(score / max(1, len(tokens) - 1)))))
    return results
