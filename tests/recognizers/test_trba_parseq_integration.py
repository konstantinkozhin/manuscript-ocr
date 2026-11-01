"""Checkpoint/resume and ONNX integration for retained PARSeq."""
import json
import pytest
torch = pytest.importorskip("torch")
from manuscript.recognizers._trba.model.model import TRBAModel, TRBAONNXWrapper

OPTIONS = ("linear",)

def config(name="linear"):
    return dict(cnn_backbone="seresnetlite31v2", hidden_size=16, cnn_out_channels=32,
                num_encoder_layers=2, img_h=32, img_w=64, max_len=4,
                decoder_type="parseq", decoder_heads=2, decoder_ffn=32,
                decoder_layers=2)

@pytest.mark.parametrize("name", OPTIONS)
def test_actual_objective_and_exact_cpu_resume_next_update(tmp_path, name):
    from manuscript.recognizers._trba.training.utils import save_checkpoint, load_checkpoint
    model = TRBAModel(8, **config(name)).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=False)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-4, total_steps=8)
    images = torch.randn(2, 3, 32, 64)
    text = torch.tensor([[1, 4, 5, 4, 4], [1, 5, 0, 0, 0]])
    targets = torch.tensor([[4, 5, 4, 4, 2], [5, 2, 0, 0, 0]])
    def step(network, opt, sched):
        opt.zero_grad(set_to_none=True)
        result = network(images, text, batch_max_length=4)
        value = .7 * network.compute_attention_loss(result, targets) + .3 * network.compute_ctc_loss(result["ctc_logits"], targets, torch.tensor([5, 2]))
        value.backward()
        assert torch.isfinite(value)
        for key, p in network.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), key
        opt.step()
        sched.step()
        return value.detach()
    step(model, optimizer, scheduler)
    path = tmp_path / "last_ckpt.pth"
    save_checkpoint(path, model, optimizer, scheduler, None, 1, 1, 1, 0, [], {}, config(name), str(tmp_path))
    rng = torch.get_rng_state().clone()
    restored = TRBAModel(8, **config(name)).train()
    opt2 = torch.optim.AdamW(restored.parameters(), lr=1e-4, foreach=False)
    sched2 = torch.optim.lr_scheduler.OneCycleLR(opt2, max_lr=1e-4, total_steps=8)
    metadata = load_checkpoint(path, restored, opt2, sched2, map_location="cpu")
    assert metadata["config"] == config(name)
    assert torch.equal(rng, torch.get_rng_state())
    expected = step(model, optimizer, scheduler)
    torch.set_rng_state(rng)
    actual = step(restored, opt2, sched2)
    torch.testing.assert_close(expected, actual, atol=0, rtol=0)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], atol=0, rtol=0)
    assert scheduler.state_dict() == sched2.state_dict()
    with pytest.raises(RuntimeError, match="does not match"):
        load_checkpoint(path, TRBAModel(8, **dict(config(), transformation="tps")), map_location="cpu")


@pytest.mark.parametrize("name", OPTIONS)
def test_public_export_preserves_variant_and_dynamic_batch(tmp_path, name):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from manuscript.recognizers._trba import TRBA
    cfg = config(name)
    model = TRBAModel(8, **cfg).eval()
    weights, settings, charset, output = [tmp_path / p for p in ("weights.pth", "config.json", "charset.txt", "model.onnx")]
    torch.save(model.state_dict(), weights)
    settings.write_text(json.dumps(cfg), encoding="utf-8")
    charset.write_text("<PAD>\n<SOS>\n<EOS>\n<BLANK>\na\nb\nc\nd\n", encoding="utf-8")
    TRBA.export(weights, settings, charset, output, simplify=False)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(output), options, providers=["CPUExecutionProvider"])
    wrapper = TRBAONNXWrapper(model, max_length=5).eval()
    for batch in (1, 2):
        x = torch.randn(batch, 3, 32, 64)
        with torch.no_grad():
            expected = wrapper(x)
        actual = torch.from_numpy(session.run(None, {"input": x.numpy()})[0])
        torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
