"""Numerical, causal, objective, resume and ONNX checks for new TRBA heads."""

import json

import pytest

torch = pytest.importorskip("torch")
from torch.nn import functional as F

from manuscript.recognizers._trba.model.model import TRBAModel, TRBAONNXWrapper
from manuscript.recognizers._trba.model.decoder_parseq import PARSeqDecoder

HEADS = {"parseq": PARSeqDecoder}


def head(name, **kwargs):
    return HEADS[name](16, 8, 4, 1, 2, 0, 3, heads=2, ffn=32, dropout=0, **kwargs)


@pytest.mark.parametrize("name", HEADS)
def test_head_objective_has_finite_gradients_including_auxiliary_tasks(name):
    decoder = head(name).train()
    memory = torch.randn(2, 8, 16, requires_grad=True)
    text = torch.tensor([[1, 4, 5, 4, 4], [1, 5, 0, 0, 0]])
    targets = torch.tensor([[4, 5, 4, 4, 2], [5, 2, 0, 0, 0]])
    result = decoder.training_outputs(memory, text, 4)
    loss = decoder.loss(result, targets)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(memory.grad).all() and memory.grad.abs().sum() > 0
    for key, parameter in decoder.named_parameters():
        assert parameter.grad is not None, key
        assert torch.isfinite(parameter.grad).all(), key
    if name == "parseq":
        assert len(result["permutation_logits"]) == 6



@pytest.mark.parametrize("name", ["parseq"])
def test_autoregressive_training_cannot_read_future_or_current_answer(name):
    decoder = head(name).eval()
    memory = torch.randn(2, 8, 16)
    text = torch.tensor([[1, 4, 5, 6, 7], [1, 5, 4, 7, 6]])
    changed = text.clone()
    changed[:, 2:] = 4
    with torch.no_grad():
        before = decoder.forward_training(memory, text, 4)
        after = decoder.forward_training(memory, changed, 4)
    # Output at position 1 predicts text[:,2]; that answer must be hidden too.
    torch.testing.assert_close(before[:, :2], after[:, :2], atol=1e-6, rtol=1e-6)


def test_parseq_permuted_masks_block_direct_and_indirect_answer_leakage():
    decoder = head("parseq", layers=3).eval()
    order = torch.tensor([0, 3, 1, 4, 2, 5])
    content_mask, query_mask = decoder.attention_masks(order)
    reference = torch.zeros(6, 6, dtype=torch.bool)
    for index, token in enumerate(order):
        reference[token, order[index + 1:]] = True
    assert torch.equal(content_mask, reference[:-1, :-1])
    reference.fill_diagonal_(True)
    assert torch.equal(query_mask, reference[1:, :-1])
    memory, text = torch.randn(1, 8, 16), torch.tensor([[1, 4, 5, 6, 7]])
    for query in range(5):
        changed = text.clone()
        changed[:, query_mask[query]] = 7 - (text[:, query_mask[query]] % 4)
        with torch.no_grad():
            before = decoder.decode(memory, text, content_mask, query_mask)
            after = decoder.decode(memory, changed, content_mask, query_mask)
        torch.testing.assert_close(before[:, query], after[:, query], atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("name", ["parseq"])
def test_teacher_forcing_matches_stepwise_greedy_on_its_own_prefix(name):
    kwargs = {"refine_iters": 0} if name == "parseq" else {}
    decoder = head(name, **kwargs).eval()
    memory = torch.randn(2, 8, 16)
    actual, predictions = decoder.greedy_decode(memory, 5, onnx_mode=True)
    text = torch.cat((torch.ones(2, 1, dtype=torch.long), predictions[:, :-1]), dim=1)
    # Explicit decode avoids masking EOS used inside fixed-step AR prefixes.
    if name == "parseq":
        content_mask, query_mask = decoder.attention_masks(torch.arange(6))
        expected = decoder.decode(memory, text, content_mask, query_mask)
    else:
        expected = decoder.forward_training(memory, text, 4)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)




def test_parseq_loss_supervises_eos_only_for_first_two_permutations():
    decoder = head("parseq").train()
    targets = torch.tensor([[4, 5, 2, 0, 0]])
    outputs = [torch.randn(1, 5, 8, requires_grad=True) for _ in range(6)]
    actual = decoder.loss({"permutation_logits": outputs}, targets)
    expected = sum(F.cross_entropy(x[0, :3 if i < 2 else 2], targets[0, :3 if i < 2 else 2], reduction="sum")
                   for i, x in enumerate(outputs)) / 14
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert outputs[0].grad[0, 2].abs().sum() > 0
    assert outputs[2].grad[0, 2:].abs().sum() == 0




@pytest.mark.parametrize("name", HEADS)
def test_shared_encoder_ctc_weights_and_rng_match_baseline(name):
    kwargs = dict(num_classes=8, cnn_backbone="seresnet31lite", hidden_size=16,
                  cnn_out_channels=32, num_encoder_layers=1, img_h=32, img_w=64, max_len=4,
                  decoder_heads=2, decoder_ffn=32)
    torch.manual_seed(8)
    baseline = TRBAModel(**kwargs)
    baseline_rng = torch.get_rng_state()
    torch.manual_seed(8)
    model = TRBAModel(**kwargs, decoder_type=name)
    assert torch.equal(torch.get_rng_state(), baseline_rng)
    for key, value in baseline.state_dict().items():
        if not key.startswith("attention_decoder."):
            assert torch.equal(model.state_dict()[key], value), key


@pytest.mark.parametrize("name", HEADS)
def test_real_model_backward_and_checkpoint_resume(tmp_path, name):
    from manuscript.recognizers._trba.training.utils import save_checkpoint, load_checkpoint

    kwargs = dict(num_classes=8, cnn_backbone="seresnet31lite", hidden_size=16,
                  cnn_out_channels=32, num_encoder_layers=1, img_h=32, img_w=64, max_len=4,
                  decoder_type=name, decoder_heads=2, decoder_ffn=32)
    model = TRBAModel(**kwargs).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=False)
    images = torch.randn(2, 3, 32, 64)
    text = torch.tensor([[1, 4, 5, 4, 4], [1, 5, 0, 0, 0]])
    targets = torch.tensor([[4, 5, 4, 4, 2], [5, 2, 0, 0, 0]])
    output = model(images, text, batch_max_length=4)
    loss = .7 * model.compute_attention_loss(output, targets) + .3 * model.compute_ctc_loss(output["ctc_logits"], targets, torch.tensor([5, 2]))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    path = tmp_path / "last_ckpt.pth"
    save_checkpoint(path, model, optimizer, None, None, 1, 1, float(loss), 0, [], {}, kwargs, str(tmp_path))
    state = torch.get_rng_state().clone()
    restored = TRBAModel(**kwargs)
    opt2 = torch.optim.AdamW(restored.parameters(), lr=1e-4, foreach=False)
    metadata = load_checkpoint(path, restored, opt2, map_location="cpu")
    assert metadata["epoch"] == 1 and torch.equal(state, torch.get_rng_state())
    assert len(opt2.state) == len(optimizer.state)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key])

    def step(network, opt):
        opt.zero_grad(set_to_none=True)
        result = network(images, text, batch_max_length=4)
        value = network.compute_attention_loss(result, targets)
        value.backward()
        opt.step()
        return value.detach()

    expected_loss = step(model, optimizer)
    torch.set_rng_state(state)
    actual_loss = step(restored, opt2)
    torch.testing.assert_close(actual_loss, expected_loss)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key])
    wrong = TRBAModel(**dict(kwargs, transformation="tps"))
    with pytest.raises(RuntimeError, match="does not match"):
        load_checkpoint(path, wrong, map_location="cpu")


@pytest.mark.parametrize("name", HEADS)
def test_public_onnx_export_dynamic_batch_and_logits_match(tmp_path, name):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from manuscript.recognizers._trba import TRBA

    cfg = dict(cnn_backbone="seresnet31lite", hidden_size=16, cnn_out_channels=32,
               num_encoder_layers=1, img_h=32, img_w=64, max_len=4,
               decoder_type=name, decoder_heads=2, decoder_ffn=32, decoder_layers=2)
    model = TRBAModel(8, **cfg).eval()
    assert model.evaluation_steps(cfg["max_len"]) == 5
    weights, config, charset, onnx = [tmp_path / p for p in ("weights.pth", "config.json", "charset.txt", "model.onnx")]
    torch.save(model.state_dict(), weights)
    config.write_text(json.dumps(cfg), encoding="utf-8")
    charset.write_text("<PAD>\n<SOS>\n<EOS>\n<BLANK>\na\nb\nc\nd\n", encoding="utf-8")
    TRBA.export(weights, config, charset, onnx, simplify=False)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(onnx), options, providers=["CPUExecutionProvider"])
    wrapper = TRBAONNXWrapper(model, max_length=5).eval()
    for batch in (1, 2):
        x = torch.randn(batch, 3, 32, 64)
        expected = wrapper(x)
        evaluated = model(x, is_train=False, batch_max_length=model.evaluation_steps(cfg["max_len"]))["attention_logits"]
        torch.testing.assert_close(evaluated, expected[:, :evaluated.shape[1]])
        actual = torch.from_numpy(session.run(None, {"input": x.numpy()})[0])
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
