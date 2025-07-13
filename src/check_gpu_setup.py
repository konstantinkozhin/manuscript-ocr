import torch
import torchvision
import torchaudio
import numpy as np


def main():
    print("PyTorch:", torch.__version__)
    print("Torchvision:", torchvision.__version__)
    print("Torchaudio:", torchaudio.__version__)
    print("NumPy:", np.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("CUDA version:", torch.version.cuda)
        print("Device count:", torch.cuda.device_count())
        print("Current device:", torch.cuda.current_device())
        print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

        # Simple tensor test on GPU
        a = torch.randn(1000, 1000, device="cuda")
        b = torch.randn(1000, 1000, device="cuda")
        c = torch.matmul(a, b)
        print("Matrix multiplication (1000x1000) succeeded on GPU.")
    else:
        print("⚠️ CUDA is NOT available. Make sure you installed the correct GPU build.")


if __name__ == "__main__":
    main()
