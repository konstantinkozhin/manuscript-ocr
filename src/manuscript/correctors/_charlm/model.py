import torch
import torch.nn as nn


class CharTransformerMLM(nn.Module):
    def __init__(self, vocab_size, emb_size=256, max_len=64, n_layers=6, n_heads=8,
                 ffn_size=1024, dropout=0.1, pad_idx=0):
        super().__init__()
        self.pad_idx = pad_idx
        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=pad_idx)
        self.pos = nn.Embedding(max_len, emb_size)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb_size, nhead=n_heads, dim_feedforward=ffn_size, dropout=dropout,
            batch_first=True, activation="gelu", norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.out = nn.Linear(emb_size, vocab_size)

    def forward(self, x):
        B, T = x.shape
        pos_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.emb(x) + self.pos(pos_ids)
        h = self.encoder(h, src_key_padding_mask=(x == self.pad_idx))
        return self.out(h)
