from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .seresnet31 import SEResNet31
from .seresnetlite31 import SEResNet31Lite
from .seresnetlite31v2 import SEResNet31LiteV2


CNN_BACKBONES = {
    "seresnet31": SEResNet31,
    "seresnet31lite": SEResNet31Lite,
    "seresnetlite31v2": SEResNet31LiteV2,
    "seresnet31lite_linear": SEResNet31LiteV2,  # Legacy saved configurations.
}


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
    
    def __init__(self, trba_model, max_length: int = 40):
        """
        Args:
            trba_model: TRBAModel instance
            max_length: Maximum decoding length
        """
        super().__init__()
        self.model = trba_model
        self.max_length = max_length
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] - input images
            
        Returns:
            logits: [B, max_length, num_classes] - logits for each position
        """
        logits, _ = self.model.forward_attention(
            x,
            text=None,
            is_train=False,
            batch_max_length=self.max_length,
            onnx_mode=True
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

    def forward(self, prev_hidden, batch_H, char_onehots, proj_H=None):
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
        if proj_H is None:
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
        proj_H = self.attention_cell.i2h(batch_H)

        for t in range(max_steps):
            # One-hot encoding (ONNX-friendly)
            onehots = self._char_to_onehot(targets)

            # Attention step
            hidden, _ = self.attention_cell(hidden, batch_H, onehots, proj_H)
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
        proj_H = self.attention_cell.i2h(batch_H)

        for t in range(steps):
            onehots = self._char_to_onehot(targets)
            hidden, _ = self.attention_cell(hidden, batch_H, onehots, proj_H)

            h_out = hidden[0]
            out_hid[:, t, :] = h_out

            # Scheduled sampling
            if t < steps - 1:
                if self.sampling_prob > 0 and torch.rand(1).item() < self.sampling_prob:
                    out = F.dropout(h_out, p=self.dropout_p, training=self.training)
                    logits_t = self.generator(out)
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
        transformation: str = "none",
        tps_num_fiducial: int = 20,
        decoder_type: str = "attention",
        decoder_layers: int = 2,
        decoder_heads: int = 8,
        decoder_ffn: int = 1024,
        decoder_dropout: float = 0.1,
        max_len: int = 40,
        parseq_permutations: int = 6,
        parseq_refine_iters: int = 1,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_encoder_layers = num_encoder_layers
        self.img_h = img_h
        self.img_w = img_w
        self.use_ctc_head = use_ctc_head
        self.cnn_backbone = cnn_backbone.lower()
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

        # ===== BiRNN Encoder =====
        enc_dim = self.cnn.out_channels

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
            # Legacy charsets may use PAD as the CTC blank.
            ctc_blank_id = pad_id if blank_id is None else blank_id
            self.ctc_head = nn.Linear(hidden_size, num_classes)
            self.ctc_loss_fn = nn.CTCLoss(
                blank=ctc_blank_id, reduction="mean", zero_infinity=True
            )

        self.transformation = transformation.lower()
        self.tps = None
        if self.transformation == "tps":
            from .tps import TPSRectifier

            # Preserve shared baseline weights and the subsequent CPU RNG stream.
            with torch.random.fork_rng(devices=[]):
                self.tps = TPSRectifier(
                    (img_h, img_w), cnn_in_channels, tps_num_fiducial
                )
        elif self.transformation != "none":
            raise ValueError(f"Unsupported transformation: {transformation!r}")

        self.decoder_type = decoder_type.lower()
        if self.decoder_type != "attention":
            if decoder_layers < 1 or decoder_heads < 1 or decoder_ffn < 1:
                raise ValueError("Decoder layers, heads and FFN size must be positive")
            from .decoder_parseq import PARSeqDecoder

            if self.decoder_type != "parseq":
                raise ValueError(f"Unsupported decoder_type: {decoder_type!r}")
            # Build after all common modules, preserving baseline CNN/RNN/CTC
            # initialization and the CPU RNG stream for the next training step.
            with torch.random.fork_rng(devices=[]):
                self.attention_decoder = PARSeqDecoder(
                    hidden_size, num_classes, max_len, sos_id, eos_id, pad_id, blank_id,
                    layers=decoder_layers, heads=decoder_heads, ffn=decoder_ffn,
                    dropout=decoder_dropout, perm_num=parseq_permutations, refine_iters=parseq_refine_iters,
                )

    def _cnn_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.tps is not None:
            x = self.tps(x)
        return self.cnn(x)

    def _encode_sequence(self, f: torch.Tensor) -> torch.Tensor:
        # Pooling by height
        f = f.mean(dim=2)  # [B, C, W']

        # Permute for RNN
        f = f.permute(0, 2, 1)  # [B, W', C]

        # BiRNN encoder 
        f = self.enc_rnn(f)  # [B, W', hidden_size]
        f = self.enc_dropout(f)

        return f

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """The original 1D sequence interface, also used by the CTC branch."""
        return self._encode_sequence(self._cnn_features(x))

    def _decoder_memory(self, x: torch.Tensor) -> Any:
        return self.encode(x)

    def evaluation_steps(self, max_chars):
        # New heads reserve an EOS slot, matching their exported ONNX graph.
        # In particular, PARSeq refinement must see the same full context.
        return max_chars if self.decoder_type == "attention" else max_chars + 1

    def forward_attention(
        self,
        x: torch.Tensor,
        text: Optional[torch.Tensor] = None,
        is_train: bool = False,
        batch_max_length: int = 25,
        onnx_mode: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for Attention decoding.

        Args:
            x: [B, 3, H, W] input images
            text: [B, T] target tokens (training only)
            is_train: training or inference mode
            batch_max_length: maximum sequence length
            onnx_mode: ONNX mode (always max_length steps)

        Returns:
            logits: [B, T, num_classes]
            preds: [B, T] (inference only) or None
        """
        enc_output = self._decoder_memory(x)

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
    ) -> Dict[str, Any]:
        """
        Unified forward pass.

        Args:
            x: [B, 3, H, W] input images
            text: [B, T] target tokens (training only)
            is_train: training mode
            batch_max_length: maximum sequence length
            onnx_mode: True for ONNX export (always max_length steps)

        Returns:
            dict with keys:
                "attention_logits": [B, T, num_classes]
                "attention_preds": [B, T] (inference only)
                "ctc_logits": [B, W, num_classes] (if use_ctc_head=True)
        """
        result: Dict[str, Any] = {}
        
        enc_output = self.encode(x)
        memory = enc_output

        # CTC head
        if self.use_ctc_head:
            result["ctc_logits"] = self.ctc_head(enc_output)

        # Attention head
        if is_train:
            assert text is not None, "text is required for training"
            if self.decoder_type == "attention":
                result["attention_logits"] = self.attention_decoder.forward_training(
                    memory, text, batch_max_length
                )
            else:
                result.update(self.attention_decoder.training_outputs(memory, text, batch_max_length))
        else:
            logits, preds = self.attention_decoder.greedy_decode(
                memory, batch_max_length, onnx_mode=onnx_mode
            )
            result["attention_logits"] = logits
            result["attention_preds"] = preds

        return result

    def compute_attention_loss(self, result, targets, criterion=None):
        """Use PARSeq permutation learning or the original attention objective."""
        if self.decoder_type != "attention":
            return self.attention_decoder.loss(result, targets)
        if criterion is None:
            criterion = nn.CrossEntropyLoss(ignore_index=self.attention_decoder.pad_id)
        return criterion(result["attention_logits"].flatten(0, 1), targets.flatten())

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
            targets: [B, T] character or attention-packed targets
            target_lengths: [B] lengths, including EOS if present

        Returns:
            ctc_loss
        """
        if not self.use_ctc_head:
            return ctc_logits.sum() * 0.0

        # Attention packing appends EOS. CTC aligns characters without EOS.
        last_indices = (target_lengths - 1).clamp_min(0).to(targets.device)
        last_tokens = targets.gather(1, last_indices.unsqueeze(1)).squeeze(1)
        has_eos = (target_lengths > 0) & (last_tokens == self.attention_decoder.eos_id)
        target_lengths = target_lengths - has_eos.long()
        ctc_logits = ctc_logits.permute(1, 0, 2)  # [W, B, num_classes]

        B, W = ctc_logits.size(1), ctc_logits.size(0)
        input_lengths = torch.full((B,), W, dtype=torch.long, device=ctc_logits.device)

        # Keep log-softmax and CTC in FP32 under mixed precision. Do not hide
        # runtime failures by silently disabling the auxiliary objective.
        with torch.autocast(device_type=ctc_logits.device.type, enabled=False):
            log_probs = ctc_logits.float().log_softmax(2)
            return self.ctc_loss_fn(log_probs, targets, input_lengths, target_lengths)
