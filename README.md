<div align="center">

[![logo](https://raw.githubusercontent.com/Leikoe/GPT-EIRB/main/docs/logo.png)](https://github.com/Leikoe/gpt-eirb/)

<h3>

[Homepage](https://github.com/Leikoe/gpt-eirb/edit/main) | [Documentation](/docs) | [Examples](/examples)

</h3>

![GitHub Repo stars](https://img.shields.io/github/stars/Leikoe/gpt-eirb)

</div>

---
# GPT-EIRB

A Minimal Language Model based on KNN and gzip compression.

## Install dependencies

> Note: for now, the project only runs on a computer with a CUDA capable gpu and CUDA installed.

```shell
# Create a conda venv (we used Mambaforge while developping the project)
conda create -n gpt-eirb
conda activate gpt-eirb

# Install kvikio
mamba install -c rapidsai -c conda-forge kvikio

# Install the rest of the dependencies
pip install -r requirements.txt
```

## Usage


> Note: The default input.txt is from the Tinyshakespeare dataset

```shell
python3 download.py # Optionnal: downloads the TinyStories dataset and creates input.txt
python3 train.py # launch the training and generate a few tokens to stdin + write them into output.txt
```
