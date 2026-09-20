"""CPU numerical and integration checks for the optional TPS rectifier."""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from manuscript.recognizers._trba.model.model import TRBAModel
from manuscript.recognizers._trba.model.tps import TPSGridGenerator, TPSRectifier


def test_tps_grid_matches_benchmark_equations_for_non_affine_controls():
    # Independent NumPy implementation of the upstream GridGenerator equations,
    # including its coefficient order and epsilon in the query radial kernel.
    generator = TPSGridGenerator((16, 32), 20)
    control = generator.control_points.double().numpy()
    points = generator.base_grid.double().numpy()
    count = len(control)
    distances = np.linalg.norm(control[:, None] - control[None], axis=-1)
    np.fill_diagonal(distances, 1)
    radial = distances**2 * np.log(distances)
    system = np.block([
        [np.ones((count, 1)), control, radial],
        [np.zeros((2, 3)), control.T],
        [np.zeros((1, 3)), np.ones((1, count))],
    ])
    r = np.linalg.norm(points[:, None] - control[None], axis=-1)
    queries = np.concatenate([np.ones((len(points), 1)), points, r**2 * np.log(r + 1e-6)], axis=1)
    offsets = np.random.default_rng(7).normal(0, 0.08, (2, count, 2)).astype(np.float32)
    expected = np.stack([
        queries @ np.linalg.solve(system, np.concatenate([control + delta, np.zeros((3, 2))]))
        for delta in offsets
    ])
    actual = generator(torch.from_numpy(offsets)).reshape(2, -1, 2)
    torch.testing.assert_close(actual, torch.from_numpy(expected).float(), atol=2e-5, rtol=2e-5)


def test_tps_grid_reproduces_affine_maps_and_propagates_gradients():
    generator = TPSGridGenerator((16, 32), 20)
    matrix = torch.tensor([[0.9, 0.08], [-0.04, 0.95]])
    shift = torch.tensor([0.05, -0.1])
    controls = generator.control_points
    offsets = (controls @ matrix.T + shift - controls).unsqueeze(0).requires_grad_()
    grid = generator(offsets).reshape(1, -1, 2)
    expected = generator.base_grid @ matrix.T + shift
    torch.testing.assert_close(grid[0], expected, atol=2e-6, rtol=2e-6)
    grid.square().mean().backward()
    assert torch.isfinite(offsets.grad).all() and offsets.grad.abs().sum() > 0


@pytest.mark.parametrize("size", [(16, 32), (64, 384)])
def test_rectifier_starts_as_identity_and_handles_autocast(size):
    rectifier = TPSRectifier(size).eval()
    images = torch.randn(2, 3, *size)
    with torch.no_grad():
        actual = rectifier(images)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            mixed = rectifier(images)
    torch.testing.assert_close(actual, images, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(mixed, actual)


def test_rectifier_learns_nonidentity_and_restores_weights(tmp_path):
    torch.manual_seed(4)
    rectifier = TPSRectifier((16, 32)).train()
    optimizer = torch.optim.AdamW(rectifier.parameters(), lr=1e-4, foreach=False)
    images = torch.randn(2, 3, 16, 32)
    target = images.roll(1, dims=-1)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = (rectifier(images) - target).square().mean()
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in rectifier.parameters())
        assert rectifier.localization.fc2.weight.grad.abs().sum() > 0
        if step == 1:
            assert rectifier.localization.conv[0].weight.grad.abs().sum() > 0
        optimizer.step()
    rectifier.eval()
    with torch.no_grad():
        expected = rectifier(images)
        assert (expected - images).abs().max() > 0.01
    weights = tmp_path / "tps.pth"
    torch.save(rectifier.state_dict(), weights)
    restored = TPSRectifier((16, 32)).eval()
    restored.load_state_dict(torch.load(weights, weights_only=True))
    torch.testing.assert_close(restored(images), expected)


def test_tps_preserves_baseline_initialization_rng_and_encoder_output():
    kwargs = dict(num_classes=7, hidden_size=8, num_encoder_layers=1,
                  cnn_out_channels=32, cnn_backbone="seresnet31lite", img_h=32, img_w=64)
    torch.manual_seed(42)
    baseline = TRBAModel(**kwargs).eval()
    baseline_rng = torch.get_rng_state().clone()
    torch.manual_seed(42)
    with_tps = TRBAModel(**kwargs, transformation="tps").eval()
    assert torch.equal(torch.get_rng_state(), baseline_rng)
    for key, value in baseline.state_dict().items():
        assert torch.equal(value, with_tps.state_dict()[key]), key
    assert not any(key.startswith("tps.") for key in baseline.state_dict())
    with torch.no_grad():
        images = torch.randn(2, 3, 32, 64)
        torch.testing.assert_close(with_tps.encode(images), baseline.encode(images))


def test_recognition_loss_trains_tps_and_checkpoint_resumes_optimizer(tmp_path):
    kwargs = dict(num_classes=7, hidden_size=8, num_encoder_layers=1,
                  cnn_out_channels=32, cnn_backbone="seresnet31lite", img_h=32, img_w=64,
                  transformation="tps", enc_dropout_p=0)
    torch.manual_seed(73)
    model = TRBAModel(**kwargs).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=False)
    images = torch.randn(2, 3, 32, 64)
    text = torch.tensor([[1, 4, 5], [1, 5, 4]])
    targets = torch.tensor([[4, 5, 2], [5, 4, 2]])
    lengths = torch.tensor([3, 3])

    def step(network, opt):
        opt.zero_grad(set_to_none=True)
        result = network(images, text=text, batch_max_length=2)
        attn = torch.nn.functional.cross_entropy(result["attention_logits"].flatten(0, 1), targets.flatten())
        ctc = network.compute_ctc_loss(result["ctc_logits"], targets, lengths)
        loss = 0.7 * attn + 0.3 * ctc
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(network.tps.localization.fc2.weight.grad).all()
        assert network.tps.localization.fc2.weight.grad.abs().sum() > 0
        opt.step()
        return loss.detach()

    step(model, optimizer)
    checkpoint = tmp_path / "checkpoint.pth"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, checkpoint)
    restored = TRBAModel(**kwargs).train()
    resumed_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-4, foreach=False)
    saved = torch.load(checkpoint, weights_only=True)
    restored.load_state_dict(saved["model"], strict=True)
    resumed_optimizer.load_state_dict(saved["optimizer"])
    # Replay the same dropout randomness on both sides of the checkpoint.
    rng = torch.get_rng_state()
    expected_loss = step(model, optimizer)
    torch.set_rng_state(rng)
    actual_loss = step(restored, resumed_optimizer)
    torch.testing.assert_close(actual_loss, expected_loss)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key])


def test_nonidentity_tps_onnx_matches_pytorch_for_dynamic_batch(tmp_path):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    rectifier = TPSRectifier((16, 32)).eval()
    with torch.no_grad():
        rectifier.localization.fc2.weight.normal_(std=0.002)
        rectifier.localization.fc2.bias.normal_(std=0.03)
    path = tmp_path / "rectifier.onnx"
    torch.onnx.export(
        rectifier, torch.randn(1, 3, 16, 32), str(path), opset_version=16,
        input_names=["images"], output_names=["rectified"],
        dynamic_axes={"images": {0: "batch"}, "rectified": {0: "batch"}}, dynamo=False,
    )
    graph = onnx.load(str(path))
    onnx.checker.check_model(graph)
    assert any(node.op_type == "GridSample" for node in graph.graph.node)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    for batch in (1, 2):
        images = torch.randn(batch, 3, 16, 32)
        with torch.no_grad():
            expected = rectifier(images)
        actual = torch.from_numpy(session.run(None, {"images": images.numpy()})[0])
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


def test_public_export_includes_tps_and_rejects_wrong_config(tmp_path):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from manuscript.recognizers._trba import TRBA
    from manuscript.recognizers._trba.model.model import TRBAONNXWrapper

    cfg = dict(hidden_size=8, num_encoder_layers=1, img_h=32, img_w=64,
               cnn_out_channels=32, cnn_backbone="seresnet31lite", transformation="tps",
               tps_num_fiducial=20)
    model = TRBAModel(6, **cfg).eval()
    with torch.no_grad():
        model.tps.localization.fc2.bias.normal_(std=0.03)
    weights = tmp_path / "checkpoint.pth"
    torch.save({"model_state": model.state_dict()}, weights)
    config = tmp_path / "config.json"
    config.write_text(json.dumps(dict(cfg, max_len=2)), encoding="utf-8")
    charset = tmp_path / "charset.txt"
    charset.write_text("<PAD>\n<SOS>\n<EOS>\n<BLANK>\na\nb\n", encoding="utf-8")
    path = tmp_path / "recognizer.onnx"
    TRBA.export(weights, config, charset, path, simplify=False)
    graph = onnx.load(str(path))
    assert graph.opset_import[0].version >= 16
    assert any(node.op_type == "GridSample" for node in graph.graph.node)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    wrapper = TRBAONNXWrapper(model, max_length=3).eval()
    for batch in (1, 2):
        images = torch.randn(batch, 3, 32, 64)
        actual = torch.from_numpy(session.run(None, {"input": images.numpy()})[0])
        with torch.no_grad():
            expected = wrapper(images)
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
    config.write_text(json.dumps(dict(cfg, max_len=2, transformation="none")), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Unexpected key"):
        TRBA.export(weights, config, charset, tmp_path / "wrong.onnx", simplify=False)


@pytest.mark.parametrize("count", [0, 3, 5])
def test_invalid_fiducial_counts_fail_early(count):
    with pytest.raises(ValueError, match="num_fiducial"):
        TPSGridGenerator((16, 32), count)
