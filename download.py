# Taken from llama2.c code and heavily modified
# Copyright (c) 2023 Andrej

import glob
import json
import os
import requests
from tqdm import tqdm

DATA_CACHE_DIR = "data"


def download_file(url: str, fname: str, chunk_size=1024):
    """Helper function to download a file from a given url"""
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
            desc=fname,
            total=total,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


def download_dataset():
    """Downloads the TinyStories dataset to DATA_CACHE_DIR"""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    # download the TinyStories dataset, unless it's already downloaded
    data_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
    data_filename = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data.tar.gz")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)
    else:
        print(f"{data_filename} already exists, skipping download...")

    # unpack the tar.gz file into all the data shards (json files)
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        print(f"Unpacking {data_filename}...")
        os.system(f"tar -xzf {data_filename} -C {data_dir}")
    else:
        print(f"{data_dir} already exists, skipping unpacking...")

    data_filename = os.path.join(DATA_CACHE_DIR, "TinyStories.txt")
    if not os.path.exists(data_filename):
        print(f"Creating {data_filename}...")
        shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
        # only shard 0 is used for now
        with open(shard_filenames[0], "r") as f:
            data = json.load(f)
            with open(data_filename, "w") as out_f:
                for story in data:
                    out_f.write("\n\n" + story["story"].strip() + "\n")
    else:
        print(f"{data_filename} already exists, skipping txt creation...")

    print("Download done.")


download_dataset()


def download_tokenizer():
    # download the TinyStories dataset, unless it's already downloaded
    data_url = "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K/tok512.model?download=true"
    data_filename = os.path.join(DATA_CACHE_DIR, "tok512.model")
    if not os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)
    else:
        print(f"{data_filename} already exists, skipping download...")


download_tokenizer()
