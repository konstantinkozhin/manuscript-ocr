"""Numerical checks for the experiment prerequisites."""

import copy

import pytest

torch = pytest.importorskip("torch")

from manuscript.recognizers._trba.model.model import AttentionDecoder, TRBAModel


def test_cached_attention_preserves_logits_and_gradients():
    torch.manual_seed(7)
    cached = AttentionDecoder(8, 8, 7, 1, 2, 0, 3, dropout_p=0)
    reference = copy.deepcopy(cached)
    original = reference.attention_cell.forward
    reference.attention_cell.forward = lambda h, x, c, proj_H=None: original(h, x, c)
    x = torch.randn(2, 6, 8, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_(True)
    text = torch.tensor([[1, 4, 5, 2], [1, 6, 4, 2]])
    actual = cached.forward_training(x, text, 3)
    expected = reference.forward_training(ref_x, text, 3)
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    expected.square().mean().backward()
    torch.testing.assert_close(x.grad, ref_x.grad)
    for a, b in zip(cached.parameters(), reference.parameters()):
        torch.testing.assert_close(a.grad, b.grad)
    cached.eval()
    reference.eval()
    for onnx_mode in (False, True):
        a, _ = cached.greedy_decode(x.detach(), 4, onnx_mode)
        b, _ = reference.greedy_decode(ref_x.detach(), 4, onnx_mode)
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize("blank_id", [3, None])
def test_ctc_uses_configured_blank_and_excludes_attention_eos(blank_id):
    model = TRBAModel(7, hidden_size=8, cnn_backbone="seresnet31lite", blank_id=blank_id)
    logits = torch.randn(2, 8, 7, requires_grad=True)
    targets = torch.tensor([[4, 4, 5, 2, 0], [6, 4, 2, 0, 0]])
    lengths = torch.tensor([4, 3])
    actual = model.compute_ctc_loss(logits, targets, lengths)
    expected = torch.nn.functional.ctc_loss(
        logits.log_softmax(2).transpose(0, 1), targets,
        torch.tensor([8, 8]), torch.tensor([3, 2]),
        blank=0 if blank_id is None else blank_id, zero_infinity=True,
    )
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0




def test_linear_residual_preserves_weights_but_allows_negative_branch():
    from manuscript.recognizers._trba.model.seresnetlite31 import SEResNet31Lite
    from manuscript.recognizers._trba.model.seresnetlite31v2 import SEResNet31LiteV2

    torch.manual_seed(31)
    base = SEResNet31Lite().eval()
    torch.manual_seed(31)
    linear = SEResNet31LiteV2().eval()
    for key, value in base.state_dict().items():
        assert torch.equal(value, linear.state_dict()[key])
    x = torch.randn(2, 256, 4, 8)
    with torch.no_grad():
        before = base.layer1[0].conv2(x)
        after = linear.layer1[0].conv2(x)
    assert before.min() >= 0
    assert after.min() < 0
    for stage in (linear.layer1, linear.layer2, linear.layer3, linear.layer4):
        for block in stage:
            assert isinstance(block.conv1.act, torch.nn.ReLU)
            assert isinstance(block.conv2.act, torch.nn.Identity)
            assert isinstance(block.relu, torch.nn.ReLU)








def test_cached_decoder_onnx_matches_pytorch_for_dynamic_batch(tmp_path):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")

    class ExportDecoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = AttentionDecoder(8, 8, 7, 1, 2, 0, 3, dropout_p=0)

        def forward(self, features):
            return self.decoder.greedy_decode(features, 4, onnx_mode=True)[0]

    model = ExportDecoder().eval()
    path = tmp_path / "decoder.onnx"
    torch.onnx.export(
        model, torch.randn(1, 6, 8), str(path), opset_version=14,
        input_names=["features"], output_names=["logits"],
        dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}},
        dynamo=False,
    )
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    for batch in (1, 2):
        x = torch.randn(batch, 6, 8)
        actual = torch.from_numpy(session.run(None, {"features": x.numpy()})[0])
        torch.testing.assert_close(actual, model(x), atol=1e-5, rtol=1e-5)
