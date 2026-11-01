"""Batched deterministic beam search, independent of the decoder cache format."""

import numpy as np


class NumpyBackend:
    def ids(self, rows, hidden):
        return np.asarray(rows, np.int64)

    def gather(self, values, indices):
        return values[np.asarray(indices, np.int64)]

    def scores(self, logits, count):
        logits = np.asarray(logits)
        if (logits.ndim != 3 or logits.shape[0] != count or not logits.shape[1]
                or not logits.shape[2] or not np.isfinite(logits).all()):
            raise ValueError('Decoder emitted invalid or nonfinite logits')
        logits = logits[:, -1].astype(np.float64)
        logits -= logits.max(axis=1, keepdims=True)
        return logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))

    def block(self, scores, index, blocked, prior):
        scores[index, blocked] = -np.inf
        scores[index] += prior

    def top(self, scores, k):
        flat = scores.ravel()
        k = min(k, flat.size)
        # Select only top-k, preserving the previous stable flattened-index ties.
        threshold = np.partition(flat, flat.size - k)[flat.size - k]
        greater = np.flatnonzero(flat > threshold)
        ties = np.flatnonzero(flat == threshold)[:k - len(greater)]
        candidates = np.sort(np.concatenate((greater, ties)))
        order = np.argsort(-flat[candidates], kind='stable')
        top = candidates[order]
        return top.tolist(), flat[top].tolist()


def generate_batch(hidden, generation, decode, banned, *, backend=None):
    """Decode all samples together; cache rows follow surviving beam parents."""
    backend = NumpyBackend() if backend is None else backend
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
        ids = backend.ids([item[0] for item in active], hidden)
        logits, present = decode(ids, backend.gather(hidden, owners), cache)
        scores = backend.scores(logits, len(active))
        for i, (tokens, score) in enumerate(active):
            backend.block(scores, i, banned(tokens, g['no_repeat_ngram_size']), score)
        parents = []
        offset = 0
        for state in states:
            count = len(state['active'])
            if not count:
                continue
            top, values = backend.top(scores[offset:offset + count], max(2, 1 + len(eos)) * beams)
            next_active, next_parents = [], []
            generated_len = len(state['active'][0][0])
            for rank, (index, score) in enumerate(zip(top, values)):
                beam, token = divmod(int(index), scores.shape[1])
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
                best = values[0]
                early = g['early_stopping']
                bound = limit - 1 if early == 'never' and penalty > 0 else generated_len
                if early is True or worst >= best / bound**penalty:
                    state['active'] = []
                    next_parents = []
            parents.extend(next_parents)
            offset += count
        if present is not None and parents:
            cache = {name: backend.gather(value, parents) for name, value in present.items()}
    results = []
    for state in states:
        for tokens, score in state['active']:
            state['finished'].append((score / max(1, len(tokens) - 1)**penalty, tokens, score))
        if not state['finished']:
            raise RuntimeError('Decoder produced no hypotheses')
        _, tokens, score = max(state['finished'], key=lambda item: item[0])
        results.append((tokens, float(np.exp(score / max(1, len(tokens) - 1)))))
    return results
