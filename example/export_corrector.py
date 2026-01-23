from manuscript import Pipeline, CharLM

CharLM.export(
    weights_path=r"exp_last_modern\checkpoints\charlm_epoch_25.pt",
      vocab_path=r"exp_last_modern\vocab.json",
        output_path=r'exp_last_modern\checkpoints\charlm_epoch_25.onnx')

