import torch
import torch.nn as nn
import torch.nn.functional as F

from .seresnet31 import SEResNet31


class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size, hidden_size, bidirectional=True, batch_first=True
        )
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        self.rnn.flatten_parameters()
        h, _ = self.rnn(x)  # [B, T, 2H]
        out = self.linear(h)  # [B, T, D]
        return out


class AttentionCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_embeddings, dropout_p=0.1):
        super().__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.rnn = nn.LSTMCell(input_size + num_embeddings, hidden_size)
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p

    def forward(self, prev_hidden, batch_H, char_onehots):
        # batch_H: [B, Tenc, C]
        proj_H = self.i2h(batch_H)  # [B, Tenc, H]
        proj_h = self.h2h(prev_hidden[0]).unsqueeze(1)
        e = self.score(torch.tanh(proj_H + proj_h))  # [B, Tenc, 1]

        alpha = F.softmax(e, dim=1)  # [B, Tenc, 1]
        alpha = F.dropout(alpha, p=self.dropout_p, training=self.training)

        context = torch.bmm(alpha.transpose(1, 2), batch_H).squeeze(1)  # [B, C]
        x = torch.cat([context, char_onehots], 1)  # [B, C + V]
        cur_hidden = self.rnn(x, prev_hidden)  # (h, c)
        return cur_hidden, alpha


class Attention(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_classes,
        sos_id: int,
        eos_id: int,
        pad_id: int,
        blank_id: int | None = None,
        dropout_p: float = 0.1,
        sampling_prob: float = 0.0,
    ):
        super().__init__()
        self.attention_cell = AttentionCell(
            input_size, hidden_size, num_classes, dropout_p=dropout_p
        )
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.sos_id = sos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.blank_id = blank_id

        self.generator = nn.Linear(hidden_size, num_classes)
        self.dropout_p = dropout_p
        self.sampling_prob = sampling_prob

    def _char_to_onehot(self, input_char, device):
        B = input_char.size(0)
        one_hot = torch.zeros(B, self.num_classes, device=device)
        one_hot.scatter_(1, input_char.unsqueeze(1), 1.0)
        return one_hot

    def _mask_logits(self, logits):
        if self.blank_id is not None:
            if logits.dim() == 3:
                logits[:, :, self.blank_id] = -1e4
            else:
                logits[:, self.blank_id] = -1e4
        return logits

    @torch.no_grad()
    def _greedy_decode(self, batch_H, batch_max_length=25):
        B = batch_H.size(0)
        device = batch_H.device
        steps = batch_max_length + 1

        h = torch.zeros(B, self.hidden_size, device=device)
        c = torch.zeros(B, self.hidden_size, device=device)
        hidden = (h, c)
        targets = torch.full((B,), self.sos_id, dtype=torch.long, device=device)
        probs = torch.zeros(B, steps, self.num_classes, device=device)

        for t in range(steps):
            onehots = self._char_to_onehot(targets, device=device)
            hidden, _ = self.attention_cell(hidden, batch_H, onehots)
            out = F.dropout(hidden[0], p=self.dropout_p, training=self.training)
            logits_t = self.generator(out)  # [B, V]
            logits_t = self._mask_logits(logits_t)
            probs[:, t, :] = logits_t
            targets = logits_t.argmax(1)

        return probs

    def forward(self, batch_H, text=None, is_train=True, batch_max_length=25):
        if not is_train:
            return self._greedy_decode(batch_H, batch_max_length)

        assert (
            text is not None
        ), "For training, `text` with <SOS> at text[:,0] is required"
        device = batch_H.device
        B = batch_H.size(0)
        steps = batch_max_length + 1

        h = torch.zeros(B, self.hidden_size, device=device)
        c = torch.zeros(B, self.hidden_size, device=device)
        hidden = (h, c)

        out_hid = torch.zeros(B, steps, self.hidden_size, device=device)
        targets = text[:, 0]  # <SOS>

        for t in range(steps):
            onehots = self._char_to_onehot(targets, device=device)
            hidden, _ = self.attention_cell(hidden, batch_H, onehots)
            out_hid[:, t, :] = hidden[0]

            out = F.dropout(hidden[0], p=self.dropout_p, training=self.training)
            logits_t = self.generator(out)

            if t < steps - 1:
                if is_train and torch.rand(1).item() < self.sampling_prob:
                    targets = logits_t.argmax(1)
                else:
                    targets = text[:, t + 1]

        logits = self.generator(out_hid)
        logits = self._mask_logits(logits)
        return logits


class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size, hidden_size, bidirectional=True, batch_first=True
        )
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        self.rnn.flatten_parameters()
        h, _ = self.rnn(x)  # [B, T, 2H]
        out = self.linear(h)  # [B, T, D]
        return out


class RCNN(nn.Module):
    def __init__(
        self,
        num_classes,
        hidden_size=256,
        sos_id: int = 1,
        eos_id: int = 2,
        pad_id: int = 0,
        blank_id: int | None = 3,
        enc_dropout_p: float = 0.1,
        dropblock_p: float = 0.0,
        dropblock_block_size: int = 5,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.hidden_size = hidden_size

        self.cnn = SEResNet31(
            in_channels=3,
            out_channels=512,
            dropblock_p=dropblock_p,
            dropblock_block_size=dropblock_block_size,
        )

        self.pool = nn.AdaptiveAvgPool2d((1, None))  # -> [B, C, 1, W]

        enc_dim = self.cnn.out_channels

        self.enc_rnn = nn.Sequential(
            BidirectionalLSTM(enc_dim, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, hidden_size),
        )
        enc_dim = hidden_size

        self.enc_dropout = nn.Dropout(enc_dropout_p)

        self.attn = Attention(
            input_size=enc_dim,
            hidden_size=hidden_size,
            num_classes=num_classes,
            sos_id=sos_id,
            eos_id=eos_id,
            pad_id=pad_id,
            blank_id=blank_id,
            dropout_p=0.1,
            sampling_prob=0.0,
        )

    def encode(self, x):
        f = self.cnn(x)  # [B, C, H, W]
        f = self.pool(f).squeeze(2)  # [B, C, W]
        f = f.permute(0, 2, 1)  # [B, W, C]
        f = self.enc_rnn(f)  # [B, W, H]
        f = self.enc_dropout(f)
        return f

    def forward(self, x, text=None, is_train=True, batch_max_length=25):
        enc = self.encode(x)
        return self.attn(
            enc, text=text, is_train=is_train, batch_max_length=batch_max_length
        )
