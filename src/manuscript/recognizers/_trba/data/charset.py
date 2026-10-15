def load_charset(charset_path: str):
    """
    Load one token per line, preserving token order as model output indices.

    PAD, SOS and EOS are required by TRBA; BLANK is optional. Literal spaces
    are preserved and empty lines are ignored. Example:
        <PAD>
        <SOS>
        <EOS>
        <BLANK>
        a
        b
        ...
    Returns (itos, stoi).
    """
    itos = []
    with open(charset_path, "r", encoding="utf-8") as f:
        for line in f:
            tok = line.rstrip("\n")
            if tok == "":
                continue
            itos.append(tok)
    stoi = {s: i for i, s in enumerate(itos)}
    return itos, stoi
