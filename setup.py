import os
from setuptools import setup, find_packages


def parse_requirements(fname="requirements.txt"):
    here = os.path.dirname(__file__)
    with open(os.path.join(here, fname), encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]


setup(
    name="manuscript-ocr",
    version="0.1.6",
    description="EAST-based OCR detector API",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="",
    author_email="sherstpasha99@gmail.com",
    url="https://github.com/konstantinkozhin/manuscript-ocr",
    license="MIT",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=parse_requirements(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
