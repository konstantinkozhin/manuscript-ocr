"""Export local HF encoder/decoder models; training dependencies stay optional."""

import json
from pathlib import Path
import types


def export(source, output, *, use_cache=True, fp16=True):
    """Export dynamic image batching and optionally a two-stage KV-cache decoder.

    The source is a local Hugging Face model directory. Set use_cache=False for
    a full-prefix bundle. torch/transformers are needed only during export.
    """
    try:
        import torch
        from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel
        from transformers.cache_utils import EncoderDecoderCache
        from transformers.image_processing_utils_fast import BaseImageProcessorFast
    except ImportError as exc:
        raise ImportError('TrOCR export requires torch and transformers; install them in the export environment') from exc
    from manuscript.utils._onnx_export import half_model

    source, output = Path(source), Path(output)
    if not source.is_dir():
        raise FileNotFoundError(source)
    output.mkdir(parents=True, exist_ok=True)
    model = VisionEncoderDecoderModel.from_pretrained(
        source, attn_implementation="eager", local_files_only=True
    ).eval()
    processor = AutoImageProcessor.from_pretrained(source, local_files_only=True)
    if (getattr(processor, "do_center_crop", False) or not processor.do_resize
            or not processor.do_normalize or not processor.do_rescale):
        raise ValueError('Exporter requires RGB resize/rescale/normalize without center crop')
    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    height, width = processor.size['height'], processor.size['width']
    pixels = torch.zeros(1, 3, height, width)
    # Freeze spatial interpolation only. The image batch stays dynamic.
    if model.config.encoder.model_type == 'dinov2':
        embeddings = model.encoder.embeddings
        tokens = 1 + (height // model.config.encoder.patch_size) * (width // model.config.encoder.patch_size)
        with torch.no_grad():
            position = embeddings.interpolate_pos_encoding(torch.zeros(1, tokens, model.config.encoder.hidden_size), height, width)
        embeddings.register_buffer('onnx_position_embeddings', position)

        def fixed_position(self, embeddings, height, width):
            return self.onnx_position_embeddings

        embeddings.interpolate_pos_encoding = types.MethodType(fixed_position, embeddings)

    def flatten_cache(cache):
        if hasattr(cache, 'self_attention_cache'):
            return tuple(tensor for self_layer, cross_layer in zip(
                cache.self_attention_cache.layers, cache.cross_attention_cache.layers
            ) for tensor in (self_layer.keys, self_layer.values, cross_layer.keys, cross_layer.values))
        return tuple(tensor for layer in cache for tensor in layer)

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = model.encoder
            self.projection = getattr(model, 'enc_to_dec_proj', None)

        def forward(self, pixel_values):
            kwargs = {'interpolate_pos_encoding': True} if model.config.encoder.model_type in ('dinov2', 'vit', 'deit') else {}
            hidden = self.encoder(pixel_values=pixel_values, return_dict=False, **kwargs)[0]
            return self.projection(hidden) if self.projection is not None else hidden

    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = model.decoder

        def forward(self, input_ids, encoder_hidden_states, *past):
            cache = None
            if past:
                legacy = tuple(tuple(past[i:i + 4]) for i in range(0, len(past), 4))
                if hasattr(EncoderDecoderCache, 'from_legacy_cache'):
                    cache = EncoderDecoderCache.from_legacy_cache(legacy)
                else:
                    cache = EncoderDecoderCache(legacy)
            result = self.decoder(input_ids=input_ids, encoder_hidden_states=encoder_hidden_states,
                                  past_key_values=cache, use_cache=use_cache, return_dict=True)
            return (result.logits, *flatten_cache(result.past_key_values)) if use_cache else result.logits

    encoder, decoder = Encoder().eval(), Decoder().eval()
    export_options = dict(dynamo=False, opset_version=17)
    with torch.no_grad():
        hidden = encoder(pixels)
        torch.onnx.export(encoder, (pixels,), output / 'encoder.fp32.onnx',
                          input_names=['pixel_values'], output_names=['encoder_hidden_states'],
                          dynamic_axes={'pixel_values': {0: 'batch'}, 'encoder_hidden_states': {0: 'batch'}},
                          **export_options)
        print('encoder exported', flush=True)
        ids = torch.ones(2, 2, dtype=torch.long)
        repeated = hidden.repeat(2, 1, 1)
        first = decoder(ids, repeated)
        cache_count = len(first) - 1 if use_cache else 0
        if use_cache and (not cache_count or cache_count % 4):
            raise ValueError('Expected four self/cross attention cache tensors per layer')
        mapping = [{'input': f'past.{i}', 'output': f'present.{i}'} for i in range(cache_count)]
        outputs = ['logits', *(x['output'] for x in mapping)]
        axes = {'input_ids': {0: 'batch', 1: 'sequence'}, 'encoder_hidden_states': {0: 'batch'},
                'logits': {0: 'batch', 1: 'sequence'}}
        for i, entry in enumerate(mapping):
            axes[entry['output']] = {0: 'batch', 2: 'total_sequence' if i % 4 < 2 else 'encoder_sequence'}
        torch.onnx.export(decoder, (ids, repeated), output / 'decoder.fp32.onnx',
                          input_names=['input_ids', 'encoder_hidden_states'], output_names=outputs,
                          dynamic_axes=axes, **export_options)
        print('initial decoder exported', flush=True)
        if use_cache:
            axes = dict(axes)
            for i, entry in enumerate(mapping):
                axes[entry['input']] = {0: 'batch', 2: 'past_sequence' if i % 4 < 2 else 'encoder_sequence'}
            torch.onnx.export(decoder, (ids[:, :1], repeated, *first[1:]), output / 'decoder_with_past.fp32.onnx',
                              input_names=['input_ids', 'encoder_hidden_states', *(x['input'] for x in mapping)],
                              output_names=outputs, dynamic_axes=axes, **export_options)
            print('cached decoder exported', flush=True)
    parts = ['encoder', 'decoder'] + (['decoder_with_past'] if use_cache else [])
    if fp16:
        for part in parts:
            # Keep encoder states and KV-cache in FP16 between decoding steps.
            half_model(output / f'{part}.fp32.onnx', keep_io_types=False)
            print(part + ' FP16 exported', flush=True)
    tokenizer.backend_tokenizer.save(str(output / 'tokenizer.json'))
    native_uint8 = int(processor.resample) == 3 or (
        int(processor.resample) == 2 and torch.backends.cpu.get_cpu_capability() in ('AVX2', 'AVX512')
    )
    resize_backend = ('torchvision_aa_uint8' if native_uint8 else 'torchvision_aa') if isinstance(processor, BaseImageProcessorFast) else 'pil'
    preprocess = dict(height=height, width=width, resample=int(processor.resample),
                      resize_backend=resize_backend,
                      mean=processor.image_mean, std=processor.image_std,
                      rescale_factor=processor.rescale_factor, do_resize=processor.do_resize)
    for precision in ('fp32', 'fp16') if fp16 else ('fp32',):
        config = dict(schema_version=1, decoder_format='kv_cache' if use_cache else 'full_prefix',
                      algorithm='autoregressive', task='text_recognition', decoder=f'decoder.{precision}.onnx',
                      tokenizer='tokenizer.json', preprocess=preprocess,
                      generation={**model.generation_config.to_dict(), 'use_cache': use_cache},
                      source_model_type=dict(encoder=model.config.encoder.model_type, decoder=model.config.decoder.model_type))
        if use_cache:
            config.update(decoder_with_past=f'decoder_with_past.{precision}.onnx', cache_mapping=mapping)
        (output / f'encoder.{precision}.json').write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return output
