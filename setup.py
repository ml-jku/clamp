import os

import pkg_resources
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="clamp",
    py_modules=["clamp"],
    version="1.0",
    author="Philipp Seidl",
    author_email="ph.seidl92@gmail.com",
    packages=find_packages(exclude=["tests*"]),
    install_requires="""mhnreact@git+https://github.com/ml-jku/mhn-react.git
        clip@git+https://github.com/openai/CLIP.git
        numpy
        tqdm
        scikit-learn
        scipy
        pandas
        nltk
        rdkit
        transformers
        matplotlib
        torch
        torchvision
        loguru
        wandb
        mlflow
        swifter""".split(),
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ml-jku/clamp",
    python_requires='>=3.7',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD-2-Clause License",
        "Operating System :: linux-64",
    ],
    include_package_data=True,
    extras_require={'dev': ['pytest']},
)
