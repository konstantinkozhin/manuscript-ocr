from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .seresnet31 import SEResNet31
from .seresnetlite31 import SEResNet31Lite


CNN_BACKBONES = {
    "seresnet31": SEResNet31,
    "seresnet31lite": SEResNet31Lite,
}


class TextContextEncoder(nn.Module):
    """Encodes preceding text context into a fixed-size vector.

    Uses character embeddings followed by a unidirectional LSTM.
    The last hidden state is projected to ``output_dim``.
    When context is empty (all padding), returns a zero vector.
    """

    def __init__(
        self,
        num_chars: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 128,
        pad_id: int = 0,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.output_dim = output_dim
        self.embedding = nn.Embedding(num_chars, embed_dim, padding_idx=pad_id)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, context_ids: torch.Tensor) -> torch.Tensor:
        """Encode context token ids into a vector.

        Args:
            context_ids: ``[B, L]`` token ids (0-padded).

        Returns:
            ``[B, output_dim]`` context vector (zeros when input is all-pad).
        """
        mask = (context_ids != self.pad_id).any(dim=1)  # [B]
        emb = self.embedding(context_ids)  # [B, L, E]
        _, (h_n, _) = self.rnn(emb)  # h_n: [1, B, H]
        out = self.proj(h_n.squeeze(0))  # [B, D]
        # Zero-out context for samples with no context
        out = out * mask.unsqueeze(1).float()
        return out


class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size, hidden_size, bidirectional=True, batch_first=True
        )
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        h, _ = self.rnn(x)  # [B, T, 2H]
        out = self.linear(h)  # [B, T, D]
        return out


class TRBAONNXWrapper(nn.Module):
    """
    ONNX wrapper for TRBA model.
    """
    
    def __init__(self, trba_model, max_length: int = 40, context_window: int = 0):
        """
        Args:
            trba_model: TRBAModel instance
            max_length: Maximum decoding length
            context_window: Context token length (0 = no context)
        """
        super().__init__()
        self.model = trba_model
        self.max_length = max_length
        self.context_window = context_window
        self.use_context = context_window > 0 and trba_model.context_encoder is not None
        
    def forward(self, x: torch.Tensor, context_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] - input images
            context_ids: [B, context_window] - context token ids (optional)
            
        Returns:
            logits: [B, max_length, num_classes] - logits for each position
        """
        logits, _ = self.model.forward_attention(
            x,
            text=None,
            is_train=False,
            batch_max_length=self.max_length,
            onnx_mode=True,
            context_ids=context_ids,
        )
        return logits


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
        """
        Args:
            prev_hidden: (h, c) tuple for LSTM
            batch_H: [B, Tenc, C] encoder output
            char_onehots: [B, V] one-hot encoded character

        Returns:
            cur_hidden: (h, c) tuple
            alpha: [B, Tenc, 1] attention weights
        """
        # Attention mechanism
        proj_H = self.i2h(batch_H)  # [B, Tenc, H]
        proj_h = self.h2h(prev_hidden[0]).unsqueeze(1)
        e = self.score(torch.tanh(proj_H + proj_h))  # [B, Tenc, 1]

        alpha = F.softmax(e, dim=1)  # [B, Tenc, 1]
        alpha = F.dropout(alpha, p=self.dropout_p, training=self.training)

        # Context vector
        context = torch.bmm(alpha.transpose(1, 2), batch_H).squeeze(1)  # [B, C]

        # Concatenate context and character embedding
        x = torch.cat([context, char_onehots], 1)  # [B, C + V]

        # Decoder step
        cur_hidden = self.rnn(x, prev_hidden)  # (h, c)
        return cur_hidden, alpha


class AttentionDecoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        sos_id: int,
        eos_id: int,
        pad_id: int,
        blank_id: Optional[int] = None,
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

    def _char_to_onehot(self, input_char: torch.Tensor) -> torch.Tensor:
        return F.one_hot(input_char, num_classes=self.num_classes).float()

    def _mask_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if self.blank_id is not None:
            if logits.dim() == 3:
                logits[:, :, self.blank_id] = -1e4
            else:
                logits[:, self.blank_id] = -1e4
        return logits

    @torch.no_grad()
    def greedy_decode(
        self, batch_H: torch.Tensor, batch_max_length: int = 25, onnx_mode: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Greedy decoding.

        Args:
            batch_H: [B, W, hidden_size] encoder output
            batch_max_length: maximum sequence length
            onnx_mode: if True, always decodes max_length steps (for ONNX)
                   if False, stops at EOS (for PyTorch)

        Returns:
            probs: [B, T, num_classes] logits
            preds: [B, T] predicted tokens
        """
        B = batch_H.size(0)
        device = batch_H.device

        # Initialize hidden states
        h = torch.zeros(B, self.hidden_size, device=device)
        c = torch.zeros(B, self.hidden_size, device=device)
        hidden = (h, c)

        # Start with SOS token
        targets = torch.full((B,), self.sos_id, dtype=torch.long, device=device)

        all_probs = []
        all_preds = []

        # Decoding
        max_steps = batch_max_length if onnx_mode else (batch_max_length + 1)

        for t in range(max_steps):
            # One-hot encoding (ONNX-friendly)
            onehots = self._char_to_onehot(targets)

            # Attention step
            hidden, _ = self.attention_cell(hidden, batch_H, onehots)
            h_out = hidden[0]

            # Generate logits
            out = F.dropout(h_out, p=self.dropout_p, training=self.training)
            logits_t = self.generator(out)
            logits_t = self._mask_logits(logits_t)

            all_probs.append(logits_t.unsqueeze(1))

            # Greedy choice
            next_tokens = logits_t.argmax(1)
            all_preds.append(next_tokens.unsqueeze(1))

            targets = next_tokens.clone()

            # Early stopping
            if not onnx_mode and (next_tokens == self.eos_id).all():
                break

        probs = torch.cat(all_probs, dim=1)  # [B, T, V]
        preds = torch.cat(all_preds, dim=1)  # [B, T]

        return probs, preds

    def forward_training(
        self, batch_H: torch.Tensor, text: torch.Tensor, batch_max_length: int = 25
    ) -> torch.Tensor:
        """
        Training forward pass with teacher forcing.
        Args:
            batch_H: [B, W, hidden_size] encoder output
            text: [B, T] target tokens (with SOS token at the beginning)
            batch_max_length: maximum length
        Returns:
            logits: [B, T, num_classes]
        """
        device = batch_H.device
        B = batch_H.size(0)
        steps = batch_max_length + 1

        # Initialize hidden states
        h = torch.zeros(B, self.hidden_size, device=device)
        c = torch.zeros(B, self.hidden_size, device=device)
        hidden = (h, c)

        out_hid = torch.zeros(B, steps, self.hidden_size, device=device)
        targets = text[:, 0]  # <SOS>

        for t in range(steps):
            onehots = self._char_to_onehot(targets)
            hidden, _ = self.attention_cell(hidden, batch_H, onehots)

            h_out = hidden[0]
            out_hid[:, t, :] = h_out

            out = F.dropout(h_out, p=self.dropout_p, training=self.training)
            logits_t = self.generator(out)

            # Scheduled sampling
            if t < steps - 1:
                if torch.rand(1).item() < self.sampling_prob:
                    targets = logits_t.argmax(1)
                else:
                    targets = text[:, t + 1]

        logits = self.generator(out_hid)
        logits = self._mask_logits(logits)
        return logits


class TRBAModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        hidden_size: int = 256,
        num_encoder_layers: int = 2,
        img_h: int = 64,
        img_w: int = 256,
        cnn_in_channels: int = 3,
        cnn_out_channels: int = 512,
        cnn_backbone: str = "seresnet31",
        sos_id: int = 1,
        eos_id: int = 2,
        pad_id: int = 0,
        blank_id: Optional[int] = 3,
        enc_dropout_p: float = 0.1,
        use_ctc_head: bool = True,
        context_dim: int = 0,
        context_embed_dim: int = 64,
        context_hidden_dim: int = 128,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_encoder_layers = num_encoder_layers
        self.img_h = img_h
        self.img_w = img_w
        self.use_ctc_head = use_ctc_head
        self.cnn_backbone = cnn_backbone.lower()
        self.context_dim = context_dim

        # ===== CNN Encoder =====
        backbone_cls = CNN_BACKBONES.get(self.cnn_backbone)
        if backbone_cls is None:
            available = ", ".join(sorted(CNN_BACKBONES))
            raise ValueError(
                f"Unsupported cnn_backbone '{cnn_backbone}'. Available: {available}"
            )

        self.cnn = backbone_cls(
            in_channels=cnn_in_channels,
            out_channels=cnn_out_channels,
        )

        # ===== Text Context Encoder (optional) =====
        if context_dim > 0:
            self.context_encoder = TextContextEncoder(
                num_chars=num_classes,
                embed_dim=context_embed_dim,
                hidden_dim=context_hidden_dim,
                output_dim=context_dim,
                pad_id=pad_id,
            )
        else:
            self.context_encoder = None

        # ===== BiRNN Encoder =====
        enc_dim = self.cnn.out_channels + context_dim

        encoder_layers = []
        for i in range(num_encoder_layers):
            input_dim = enc_dim if i == 0 else hidden_size
            encoder_layers.append(
                BidirectionalLSTM(input_dim, hidden_size, hidden_size)
            )

        self.enc_rnn = nn.Sequential(*encoder_layers)
        self.enc_dropout = nn.Dropout(enc_dropout_p)

        # ===== Attention Head =====
        self.attention_decoder = AttentionDecoder(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_classes=num_classes,
            sos_id=sos_id,
            eos_id=eos_id,
            pad_id=pad_id,
            blank_id=blank_id,
            dropout_p=0.1,
            sampling_prob=0.0,
        )

        # ===== CTC Head (optional) =====
        if self.use_ctc_head:
            self.ctc_head = nn.Linear(hidden_size, num_classes)
            self.ctc_loss_fn = nn.CTCLoss(
                blank=blank_id, reduction="mean", zero_infinity=True
            )

    def encode(
        self, x: torch.Tensor, context_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # CNN
        f = self.cnn(x)  # [B, C, H', W']

        # Pooling by height
        f = f.mean(dim=2)  # [B, C, W']

        # Permute for RNN
        f = f.permute(0, 2, 1)  # [B, W', C]

        # Fuse text context if available
        if self.context_encoder is not None and context_ids is not None:
            ctx = self.context_encoder(context_ids)  # [B, D_ctx]
            ctx = ctx.unsqueeze(1).expand(-1, f.size(1), -1)  # [B, W', D_ctx]
            f = torch.cat([f, ctx], dim=2)  # [B, W', C + D_ctx]
        elif self.context_encoder is not None:
            # No context provided — pad with zeros
            B, W_len = f.size(0), f.size(1)
            zeros = torch.zeros(B, W_len, self.context_dim, device=f.device, dtype=f.dtype)
            f = torch.cat([f, zeros], dim=2)  # [B, W', C + D_ctx]

        # BiRNN encoder 
        f = self.enc_rnn(f)  # [B, W', hidden_size]
        f = self.enc_dropout(f)

        return f

    def forward_attention(
        self,
        x: torch.Tensor,
        text: Optional[torch.Tensor] = None,
        is_train: bool = False,
        batch_max_length: int = 25,
        onnx_mode: bool = False,
        context_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for Attention decoding.

        Args:
            x: [B, 3, H, W] input images
            text: [B, T] target tokens (training only)
            is_train: training or inference mode
            batch_max_length: maximum sequence length
            onnx_mode: ONNX mode (always max_length steps)
            context_ids: [B, L] preceding text context token ids (optional)

        Returns:
            logits: [B, T, num_classes]
            preds: [B, T] (inference only) or None
        """
        enc_output = self.encode(x, context_ids=context_ids)  # [B, W, hidden_size]

        if is_train:
            assert text is not None, "text is required for training"
            logits = self.attention_decoder.forward_training(
                enc_output, text, batch_max_length
            )
            return logits, None
        else:
            logits, preds = self.attention_decoder.greedy_decode(
                enc_output, batch_max_length, onnx_mode=onnx_mode
            )
            return logits, preds

    def forward(
        self,
        x: torch.Tensor,
        text: Optional[torch.Tensor] = None,
        is_train: bool = True,
        batch_max_length: int = 25,
        onnx_mode: bool = False,
        context_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Unified forward pass.

        Args:
            x: [B, 3, H, W] input images
            text: [B, T] target tokens (training only)
            is_train: training mode
            batch_max_length: maximum sequence length
            onnx_mode: True for ONNX export (always max_length steps)
            context_ids: [B, L] preceding text context token ids (optional)

        Returns:
            dict with keys:
                "attention_logits": [B, T, num_classes]
                "attention_preds": [B, T] (inference only)
                "ctc_logits": [B, W, num_classes] (if use_ctc_head=True)
        """
        result: Dict[str, Any] = {}
        
        enc_output = self.encode(x, context_ids=context_ids)

        # CTC head
        if self.use_ctc_head:
            result["ctc_logits"] = self.ctc_head(enc_output)

        # Attention head
        if is_train:
            assert text is not None, "text is required for training"
            logits = self.attention_decoder.forward_training(
                enc_output, text, batch_max_length
            )
            result["attention_logits"] = logits
        else:
            logits, preds = self.attention_decoder.greedy_decode(
                enc_output, batch_max_length, onnx_mode=onnx_mode
            )
            result["attention_logits"] = logits
            result["attention_preds"] = preds

        return result

    def compute_ctc_loss(
        self,
        ctc_logits: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute CTC loss.

        Args:
            ctc_logits: [B, W, num_classes]
            targets: [B, T] target tokens
            target_lengths: [B] target sequence lengths

        Returns:
            ctc_loss
        """
        if not self.use_ctc_head:
            return ctc_logits.sum() * 0.0

        ctc_logits = ctc_logits.permute(1, 0, 2)  # [W, B, num_classes]

        B, W = ctc_logits.size(1), ctc_logits.size(0)
        input_lengths = torch.full((B,), W, dtype=torch.long, device=ctc_logits.device)

        log_probs = ctc_logits.log_softmax(2)

        try:
            ctc_loss = nn.functional.ctc_loss(
                log_probs,
                targets,
                input_lengths,
                target_lengths,
                blank=0,
                reduction="mean",
                zero_infinity=True,
            )
        except Exception as e:
            print(f"Warning: CTC loss failed: {e}")
            ctc_loss = ctc_logits.sum() * 0.0

        return ctc_loss
