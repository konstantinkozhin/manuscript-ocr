import os
from setuptools import setup, find_packages


def parse_requirements(fname="requirements.txt"):
    here = os.path.dirname(__file__)
    with open(os.path.join(here, fname), encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#") and not ln.startswith("--")]


base_requirements = parse_requirements("requirements.txt")

setup(
    name="manuscript-ocr",
    version="0.1.8",
    description="Manuscript",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="",
    author_email="sherstpasha99@gmail.com",
    url="https://github.com/konstantinkozhin/manuscript-ocr",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=base_requirements,
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9", 
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
