import numpy as np
import pytest

from manuscript.recognizers import TrOCR


def test_ngram_bans():
    assert TrOCR._banned([1, 2, 1, 2, 1], 3) == [2]
    assert TrOCR._banned([1, 2, 1, 2], 3) == [1]
    assert TrOCR._banned([1, 2, 1], 1) == [1, 2, 1]
    assert TrOCR._banned([1], 3) == []


@pytest.mark.parametrize(
    "beams,penalty,early",
    [(1, 1.0, False), (4, 2.0, False), (4, 1.0, True), (4, 2.0, "never")],
)
def test_generation_matches_transformers(beams, penalty, early):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.set_num_threads(2)
    torch.manual_seed(7)
    encoder = transformers.ViTModel(
        transformers.ViTConfig(
            image_size=16,
            patch_size=8,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=24,
        )
    )
    decoder = transformers.TrOCRForCausalLM(
        transformers.TrOCRConfig(
            vocab_size=9,
            d_model=16,
            decoder_layers=1,
            decoder_attention_heads=2,
            decoder_ffn_dim=24,
            cross_attention_hidden_size=16,
            decoder_start_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
        )
    )
    model = transformers.VisionEncoderDecoderModel(
        encoder=encoder, decoder=decoder
    ).eval()
    generation = dict(
        decoder_start_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        max_length=12,
        num_beams=beams,
        length_penalty=penalty,
        early_stopping=early,
        no_repeat_ngram_size=3,
        use_cache=False,
    )
    pixels = torch.zeros(1, 3, 16, 16)
    with torch.no_grad():
        hidden = encoder(pixels).last_hidden_state
        reference = model.generate(pixels, **generation)[0].tolist()

    class Item:
        def __init__(self, name):
            self.name, self.type = name, "tensor(float)"

    class Session:
        def get_inputs(self):
            return [Item("input_ids"), Item("encoder_hidden_states")]

        def run(self, outputs, feed):
            with torch.no_grad():
                return [
                    decoder(
                        input_ids=torch.from_numpy(feed["input_ids"]),
                        encoder_hidden_states=torch.from_numpy(
                            feed["encoder_hidden_states"]
                        ),
                        use_cache=False,
                    ).logits.numpy()
                ]

    recognizer = TrOCR.__new__(TrOCR)
    recognizer.decoder_session = Session()
    recognizer.generation = generation
    result, _ = recognizer._generate(hidden.numpy())
    assert result == reference


@pytest.mark.parametrize("shape,target", [((11, 23), (8, 16)), ((8, 16), (15, 29))])
def test_resize_matches_torchvision_rounding(shape, target):
    torch = pytest.importorskip("torch")
    functional = pytest.importorskip("torchvision.transforms.functional")
    from manuscript.recognizers._trocr import _resize_antialiased

    image = np.random.default_rng(17).integers(0, 256, (*shape, 3), np.uint8)
    source = (
        functional.resize(
            torch.from_numpy(image.transpose(2, 0, 1)),
            list(target),
            interpolation=functional.InterpolationMode.BICUBIC,
            antialias=True,
        )
        .numpy()
        .transpose(1, 2, 0)
    )
    candidate = _resize_antialiased(image, *target, 3)
    assert np.abs(source.astype(float) - candidate).max() <= 1


def test_real_onnx_bundle_and_page(tmp_path):
    import json

    onnx = pytest.importorskip("onnx")
    tokenizers = pytest.importorskip("tokenizers")
    from manuscript.data import Page, Block, Line, TextSpan

    info = lambda name, typ, shape: onnx.helper.make_tensor_value_info(name, typ, shape)
    enc = onnx.helper.make_graph(
        [
            onnx.helper.make_node(
                "Constant",
                [],
                ["hidden"],
                value=onnx.numpy_helper.from_array(np.zeros((1, 1, 2), np.float32)),
            )
        ],
        "encoder",
        [info("pixels", onnx.TensorProto.FLOAT, [1, 3, 8, 16])],
        [info("hidden", onnx.TensorProto.FLOAT, [1, 1, 2])],
    )
    weights = np.full((5, 5), -20, np.float32)
    weights[1, 3] = 20
    weights[3, 2] = 20
    dec = onnx.helper.make_graph(
        [onnx.helper.make_node("Gather", ["weights", "input_ids"], ["logits"], axis=0)],
        "decoder",
        [
            info("input_ids", onnx.TensorProto.INT64, ["B", "T"]),
            info("encoder_hidden_states", onnx.TensorProto.FLOAT, ["B", 1, 2]),
        ],
        [info("logits", onnx.TensorProto.FLOAT, ["B", "T", 5])],
        [onnx.numpy_helper.from_array(weights, "weights")],
    )
    for name, graph in [("encoder", enc), ("decoder", dec)]:
        onnx.save(
            onnx.helper.make_model(
                graph, opset_imports=[onnx.helper.make_opsetid("", 17)], ir_version=9
            ),
            tmp_path / f"{name}.onnx",
        )
    tokenizer = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(
            {"<pad>": 0, "<s>": 1, "</s>": 2, "текст": 3, "<unk>": 4}, unk_token="<unk>"
        )
    )
    tokenizer.add_special_tokens(["<pad>", "<s>", "</s>"])
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "encoder.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "algorithm": "autoregressive",
                "decoder": "decoder.onnx",
                "tokenizer": "tokenizer.json",
                "preprocess": {
                    "height": 8,
                    "width": 16,
                    "mean": [0, 0, 0],
                    "std": [1, 1, 1],
                },
                "generation": {"decoder_start_token_id": 1, "eos_token_id": 2},
            }
        )
    )
    model = TrOCR(str(tmp_path / "encoder.onnx"), device="cpu")
    image = np.full((16, 32, 3), 255, np.uint8)
    assert model._predict_word_images([image])[0]["text"] == "текст"
    span = TextSpan(
        polygon=[[1, 1], [20, 1], [20, 12], [1, 12]], detection_confidence=1.0
    )
    page = Page(blocks=[Block(lines=[Line(text_spans=[span])])])
    result = model.predict(page, image)
    assert result.blocks[0].lines[0].text_spans[0].text == "текст"
    assert page.blocks[0].lines[0].text_spans[0].text is None
    raw = model.predict(page, image, return_raw=True)
    assert raw["predictions"][0]["metadata"]["token_ids"]
    assert raw["predictions"][0]["metadata"]["encoder_output"].shape == (1, 1, 2)
    for override in ({"num_beams": True}, {"max_length": 2.5},
                     {"length_penalty": float("nan")}, {"no_repeat_ngram_size": -1}):
        with pytest.raises(ValueError):
            TrOCR(str(tmp_path / "encoder.onnx"), generation=override)
    # A legacy fixed-batch encoder remains usable with a larger requested batch.
    legacy = TrOCR(str(tmp_path / "encoder.onnx"), batch_size=2)
    assert [x["text"] for x in legacy._predict_word_images([image, image])] == ["текст", "текст"]


def test_registry_companions_are_resolved_without_model_registration(
    tmp_path, monkeypatch
):
    import json
    from manuscript import models

    files = {}
    for role in ("weights", "decoder", "tokenizer"):
        p = tmp_path / (role + ".onnx" if role != "tokenizer" else "tokenizer.json")
        p.write_bytes(b"placeholder")
        files[role] = p
    config = tmp_path / "settings.json"
    config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "algorithm": "autoregressive",
                "decoder": "unavailable.onnx",
                "tokenizer": "unavailable.json",
                "preprocess": {
                    "height": 8,
                    "width": 16,
                    "mean": [0, 0, 0],
                    "std": [1, 1, 1],
                },
                "generation": {"decoder_start_token_id": 1, "eos_token_id": 2},
            }
        )
    )
    files["config"] = config

    def resolve(name, class_name, **kwargs):
        assert name == "future-model" and class_name == "TrOCR"
        return files

    monkeypatch.setattr(models, "resolve", resolve)
    instance = TrOCR("future-model", device="cpu")
    assert instance.config_path == str(config)
    assert instance.decoder_path == str(files["decoder"])
    assert instance.tokenizer_path == str(files["tokenizer"])
    with pytest.raises(ValueError, match="do_sample"):
        TrOCR("future-model", generation={"do_sample": True})
