import os
from setuptools import setup, find_packages

def parse_requirements(fname="requirements.txt"):
    here = os.path.dirname(__file__)
    with open(os.path.join(here, fname)) as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]

setup(
    name="manuscript-ocr",
    version="0.1.1",
    description="manuscript-ocr",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",

    author="",
    author_email="",

    url="",
    license="MIT",

    packages=find_packages(where="src"),
    package_dir={"": "src"},

    python_requires=">=3.8",
    install_requires=parse_requirements(),

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    include_package_data=True,
)
