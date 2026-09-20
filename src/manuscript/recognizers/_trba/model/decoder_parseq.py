"""PARSeq decoder and permutation learning adapted to the existing TRBA encoder.

Based on baudm/parseq; Copyright 2022 Darwin Bautista, Apache-2.0.
"""

import math
from itertools import permutations

import torch
from torch import nn
from torch.nn import functional as F

DECODER_DEFAULTS = dict(
    decoder_type="attention",
    decoder_layers=2,
    decoder_heads=8,
    decoder_ffn=1024,
    decoder_dropout=0.1,
    max_len=40,
    parseq_permutations=6,
    parseq_refine_iters=1,
)


def decoder_config(config):
    return {key: config.get(key, value) for key, value in DECODER_DEFAULTS.items()}


class MultiheadAttention(nn.Module):
    def __init__(self, dim, heads, dropout=0.0, scale=None):
        super().__init__()
        if dim % heads:
            raise ValueError("Decoder dimension must be divisible by decoder_heads")
        self.heads, self.head_dim = heads, dim // heads
        self.scale = self.head_dim**-0.5 if scale is None else scale
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def split(self, x):
        return x.reshape(x.shape[0], x.shape[1], self.heads, self.head_dim).transpose(
            1, 2
        )

    def project_kv(self, memory):
        return self.split(self.k(memory)), self.split(self.v(memory))

    def forward(self, query, memory=None, mask=None, padding=None, kv=None):
        q = self.split(self.q(query)) * self.scale
        k, v = self.project_kv(memory) if kv is None else kv
        scores = q @ k.transpose(-1, -2)
        if mask is not None:
            scores = scores.masked_fill(mask[None, None], float("-inf"))
        if padding is not None:
            scores = scores.masked_fill(padding[:, None, None], float("-inf"))
        weights = self.dropout(F.softmax(scores.float(), dim=-1).to(v.dtype))
        out = (weights @ v).transpose(1, 2).reshape(query.shape[0], query.shape[1], -1)
        return self.out(out)


class DecoderBase(nn.Module):
    def __init__(self, dim, num_classes, max_len, sos_id, eos_id, pad_id, blank_id):
        super().__init__()
        if max_len < 1:
            raise ValueError("max_len must be positive")
        self.dim, self.num_classes, self.max_len = dim, num_classes, max_len
        self.sos_id, self.eos_id, self.pad_id, self.blank_id = (
            sos_id,
            eos_id,
            pad_id,
            blank_id,
        )
        banned = torch.zeros(num_classes, dtype=torch.bool)
        banned[sos_id] = banned[pad_id] = True
        if blank_id is not None:
            banned[blank_id] = True
        self.register_buffer("banned_tokens", banned)

    def mask_logits(self, logits):
        return logits.masked_fill(self.banned_tokens, -1e4)

    def steps(self, requested):
        if requested < 1 or requested > self.max_len + 1:
            raise ValueError(
                f"Decoder supports 1..{self.max_len + 1} output steps, got {requested}"
            )
        return requested

    def training_outputs(self, memory, text, max_len):
        return {"attention_logits": self.forward_training(memory, text, max_len)}

    def loss(self, result, targets):
        return F.cross_entropy(
            result["attention_logits"].flatten(0, 1),
            targets.flatten(),
            ignore_index=self.pad_id,
        )


class PARSeqLayer(nn.Module):
    def __init__(self, dim, heads, ffn, dropout):
        super().__init__()
        self.self_attention = MultiheadAttention(dim, heads, dropout)
        self.cross_attention = MultiheadAttention(dim, heads, dropout)
        self.norm_query, self.norm_content = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn, dim)
        )
        self.drop = nn.Dropout(dropout)

    def stream(self, x, normalized, content, memory, mask, padding):
        x = x + self.drop(self.self_attention(normalized, content, mask, padding))
        context = self.cross_attention(self.norm1(x), memory)
        x = x + self.drop(context)
        return x + self.drop(self.ffn(self.norm2(x)))

    def forward(
        self, query, content, memory, query_mask, content_mask, padding, update_content
    ):
        normalized = self.norm_content(content)
        query = self.stream(
            query, self.norm_query(query), normalized, memory, query_mask, padding
        )
        if update_content:
            content = self.stream(
                content, normalized, normalized, memory, content_mask, padding
            )
        return query, content


class PARSeqDecoder(DecoderBase):
    """Adapted from https://github.com/baudm/parseq (Darwin Bautista, Apache-2.0)."""

    def __init__(
        self,
        dim,
        num_classes,
        max_len,
        sos_id,
        eos_id,
        pad_id,
        blank_id,
        layers=2,
        heads=8,
        ffn=1024,
        dropout=0.1,
        perm_num=6,
        refine_iters=1,
    ):
        super().__init__(dim, num_classes, max_len, sos_id, eos_id, pad_id, blank_id)
        if perm_num < 2 or perm_num % 2 or refine_iters < 0:
            raise ValueError("PARSeq needs an even perm_num >= 2 and refine_iters >= 0")
        self.perm_num, self.refine_iters = perm_num, refine_iters
        self.embedding = nn.Embedding(num_classes, dim, padding_idx=pad_id)
        nn.init.trunc_normal_(self.embedding.weight, std=0.02)
        with torch.no_grad():
            self.embedding.weight[pad_id].zero_()
        self.position = nn.Parameter(torch.empty(1, max_len + 1, dim))
        nn.init.trunc_normal_(self.position, std=0.02)
        self.blocks = nn.ModuleList(
            [PARSeqLayer(dim, heads, ffn, dropout) for index in range(layers)]
        )
        self.norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.generator = nn.Linear(dim, num_classes)

    @staticmethod
    def attention_masks(order):
        # rank[i] is the decoding step assigned to token position i.
        rank = torch.argsort(order)
        blocked = rank[None, :] > rank[:, None]
        content_mask = blocked[:-1, :-1]
        query_mask = (
            blocked | torch.eye(order.numel(), device=order.device, dtype=torch.bool)
        )[1:, :-1]
        return content_mask, query_mask

    def generate_orders(self, num_chars, device):
        # CPU torch RNG, saved by the trainer checkpoint, also works on Windows.
        forward = tuple(range(1, num_chars + 1))
        base = [forward]
        pair_budget = min(self.perm_num // 2, max(1, math.factorial(num_chars) // 2))
        seen = {forward, forward[::-1]}
        if num_chars < 5:
            pool = list(permutations(forward))
            candidates = [pool[i] for i in torch.randperm(len(pool)).tolist()]
        else:
            candidates = [
                tuple((torch.randperm(num_chars) + 1).tolist())
                for _ in range(self.perm_num * 2)
            ]
        for candidate in candidates:
            if len(base) >= pair_budget:
                break
            if candidate not in seen:
                base.append(candidate)
                seen.update((candidate, candidate[::-1]))
        orders = []
        for order in base:
            orders.extend(
                ((0, *order, num_chars + 1), (0, *order[::-1], num_chars + 1))
            )
        # Reverse order puts EOS immediately after BOS to learn null-context EOS.
        orders[1] = (0, *range(num_chars + 1, 0, -1))
        return torch.tensor(orders, dtype=torch.long, device=device)

    def decode(
        self,
        memory,
        text,
        content_mask=None,
        query_mask=None,
        padding=None,
        queries=None,
    ):
        embedding = self.embedding(text) * math.sqrt(self.dim)
        content = torch.cat(
            (
                embedding[:, :1],
                embedding[:, 1:] + self.position[:, : text.shape[1] - 1],
            ),
            dim=1,
        )
        content = self.drop(content)
        if queries is None:
            queries = self.position[:, : text.shape[1]].expand(text.shape[0], -1, -1)
        query = self.drop(queries)
        memory = self.memory_norm(memory)
        for index, block in enumerate(self.blocks):
            query, content = block(
                query,
                content,
                memory,
                query_mask,
                content_mask,
                padding,
                update_content=index < len(self.blocks) - 1,
            )
        return self.mask_logits(self.generator(self.norm(query)))

    def forward_training(self, memory, text, batch_max_length):
        steps = self.steps(batch_max_length + 1)
        order = torch.arange(steps + 1, device=text.device)
        content_mask, query_mask = self.attention_masks(order)
        text = text[:, :steps]
        padding = (text == self.pad_id) | (text == self.eos_id)
        return self.decode(memory, text, content_mask, query_mask, padding)

    def training_outputs(self, memory, text, max_len):
        if not self.training:
            return super().training_outputs(memory, text, max_len)
        steps = self.steps(max_len + 1)
        text = text[:, :steps]
        padding = (text == self.pad_id) | (text == self.eos_id)
        outputs = []
        for order in self.generate_orders(max_len, text.device):
            content_mask, query_mask = self.attention_masks(order)
            outputs.append(self.decode(memory, text, content_mask, query_mask, padding))
        return {"attention_logits": outputs[0], "permutation_logits": outputs}

    def loss(self, result, targets):
        if "permutation_logits" not in result:
            return super().loss(result, targets)
        losses, counts = [], []
        for index, logits in enumerate(result["permutation_logits"]):
            # EOS gets supervision in forward/reverse orders only, as upstream.
            target = (
                targets
                if index < 2
                else targets.masked_fill(targets == self.eos_id, self.pad_id)
            )
            losses.append(
                F.cross_entropy(
                    logits.flatten(0, 1),
                    target.flatten(),
                    ignore_index=self.pad_id,
                    reduction="sum",
                )
            )
            counts.append((target != self.pad_id).sum())
        return torch.stack(losses).sum() / torch.stack(counts).sum().clamp_min(1)

    @torch.no_grad()
    def greedy_decode(
        self, memory, batch_max_length, onnx_mode=False, *, return_stages=False
    ):
        """Decode once; optionally retain AR and each refinement's logits for analysis."""
        steps = self.steps(batch_max_length)
        sequence = memory
        bos = torch.full(
            (sequence.shape[0], 1),
            self.sos_id,
            dtype=torch.long,
            device=sequence.device,
        )
        text = bos
        causal = torch.triu(
            torch.ones(steps, steps, device=sequence.device, dtype=torch.bool),
            diagonal=1,
        )
        outputs = []
        # Fixed steps make cloze refinement and ONNX batch sizes agree exactly.
        for index in range(steps):
            end = index + 1
            query = self.position[:, index:end].expand(sequence.shape[0], -1, -1)
            logits = self.decode(
                memory, text, causal[:end, :end], causal[index:end, :end], queries=query
            )
            outputs.append(logits)
            if end < steps:
                text = torch.cat((text, logits.argmax(dim=-1)), dim=1)
        logits = torch.cat(outputs, dim=1)
        stages = [logits] if return_stages else None
        # Hide the character being refined; expose both left and right context.
        cloze = (
            torch.arange(steps, device=sequence.device)[None, :]
            == torch.arange(1, steps + 1, device=sequence.device)[:, None]
        )
        for _ in range(self.refine_iters):
            text = torch.cat((bos, logits[:, :-1].argmax(dim=-1)), dim=1)
            padding = (text == self.eos_id).int().cumsum(dim=1) > 0
            logits = self.decode(memory, text, causal, cloze, padding)
            if return_stages:
                stages.append(logits)
        if return_stages:
            return stages
        return logits, logits.argmax(dim=-1)
