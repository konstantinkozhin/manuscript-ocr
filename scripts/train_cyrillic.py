"""Run: python train_cyrillic.py parseq_linear [--resume]."""

import sys

from scripts.cyrillic_experiments import main

if __name__ == "__main__":
    main(["train", *sys.argv[1:]])
