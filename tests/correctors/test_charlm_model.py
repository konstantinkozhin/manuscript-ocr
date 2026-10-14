import pytest

torch = pytest.importorskip("torch")
from manuscript.correctors._charlm.model import CharTransformerMLM


@pytest.mark.parametrize("batch_size", [1, 3])
def test_model_produces_character_logits_and_supports_training(batch_size):
    model = CharTransformerMLM(vocab_size=8, emb_size=8, max_len=4,
                              n_layers=1, n_heads=2, ffn_size=16, dropout=0., pad_idx=0)
    tokens = torch.tensor([[3, 1, 4, 0]]).repeat(batch_size, 1)
    logits = model(tokens)
    assert logits.shape == (batch_size, 4, 8)
    assert torch.isfinite(logits).all()
    loss = torch.nn.functional.cross_entropy(logits[:, 1], torch.full((batch_size,), 5))
    loss.backward()
    assert any(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               and parameter.grad.abs().sum() > 0 for parameter in model.parameters())
