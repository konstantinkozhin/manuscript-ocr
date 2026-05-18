def load_charset(charset_path: str):
    """
    Loads the character vocabulary from a file of the format:
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
