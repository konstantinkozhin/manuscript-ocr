import os
from setuptools import setup, find_packages


def parse_requirements(fname="requirements.txt"):
    here = os.path.dirname(__file__)
    with open(os.path.join(here, fname), encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#") and not ln.startswith("--")]


# Базовые зависимости (CPU версия)
base_requirements = [
    "torch>=1.11.0,<2.8.0",
    "torchvision>=0.12.0,<0.23.0", 
    "torchaudio>=0.11.0,<2.8.0",
    "numpy>=1.21.0,<2.3.0",
    "opencv-python>=4.5.0,<5.0.0",
    "Pillow>=9.0.0",
    "shapely>=1.8.0,<3.0.0",
    "numba>=0.56.0,<1.0.0",
    "pydantic>=2.0.0,<3.0.0",
    "gdown>=4.4.0,<6.0.0",
]

# GPU и обучение зависимости
gpu_requirements = [
    "tensorboard>=2.8.0,<3.0.0",
    "scikit-image>=0.19.0,<1.0.0",
    "torch-optimizer>=0.1.0,<1.0.0",
]

setup(
    name="manuscript-ocr",
    version="0.1.7",
    description="EAST-based OCR detector API (CPU/GPU versions)",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="",
    author_email="sherstpasha99@gmail.com",
    url="https://github.com/konstantinkozhin/manuscript-ocr",
    license="MIT",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=base_requirements,
    extras_require={
        "gpu": gpu_requirements,
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9", 
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
