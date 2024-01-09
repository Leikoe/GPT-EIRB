#
#   takes a dataset generated from download.py (.txt) and creates a model
#

from itertools import islice
from typing import Any, Sequence
from numcodecs.compat import ensure_contiguous_ndarray_like
from kvikio._lib.libnvcomp_ll import SUPPORTED_ALGORITHMS
from kvikio.nvcomp_codec import NvCompBatchCodec
import kvikio
import cupy as cp
import numpy as np
import numcodecs
import matplotlib.pyplot as plt
from pprint import pprint
import time

import numpy as np
import tqdm
from sklearn.neighbors import KNeighborsClassifier
from utils import Timing
import humanize
from more_itertools import flatten

def batched(iterable, n):
    "Batch data into tuples of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while (batch := tuple(islice(it, n))):
        yield batch


print(f"{SUPPORTED_ALGORITHMS=}")


NVCOMP_CODEC_ID = "nvcomp_batch"

LZ4_ALGO = "LZ4"
GDEFLATE_ALGO = "Gdeflate"
SNAPPY_ALGO = "snappy"
ZSTD_ALGO = "zstd"

USED_ALGO = LZ4_ALGO

gpu_compressor = numcodecs.registry.get_codec(dict(id=NVCOMP_CODEC_ID, algorithm=LZ4_ALGO))
cpu_compressor = numcodecs.registry.get_codec({"id": USED_ALGO.lower()})


# hyperparameters
n_train = 5000
n_ctx = 16  # what is the maximum context length for predictions?
# ------------


# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
n_vocab = len(stoi)

# Train and test splits
data = np.array(encode(text), dtype=np.uint8)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


print(f"n_ctx   = {n_ctx}")
print(f"n_vocab = {n_vocab}")
print(f"train_data.shape = {train_data.shape}")
print(f"val_data.shape   = {val_data.shape}")


def get_example(split):
    """
    Get a random example from the dataset.
    """
    data = train_data if split == 'train' else val_data
    i = np.random.randint(0, len(data) - n_ctx)
    x = data[i:i+n_ctx]
    y = data[i+1:i+n_ctx+1]
    return x, y

   
def encode_batch_size(codec: NvCompBatchCodec, bufs: Sequence[Any]) -> np.ndarray: 
    """
    Compresses a batch of buffers using the given codec.
    returns the compressed sizes of each buffer in the batch, in bytes.
    """
    num_chunks = len(bufs)
    # chunks_size = sum(map(len, bufs)) # map each buffer to it's size, then sum
    # print(f"[encode_batch_size] chunk_size = {humanize.naturalsize(chunks_size)}")
    # print(f"[encode_batch_size] num_chunks = {num_chunks}")
    if num_chunks == 0:
        return []

    print("converting to contiguous arrays")
    buf_sizes = [b.size * b.itemsize for b in bufs]
    buf = cp.asarray(np.concatenate([b for b in bufs]))

    max_chunk_size = max(buf_sizes)

    # Get temp and output buffer sizes.
    print("getting temp sizes")
    temp_size = codec._algo.get_compress_temp_size(num_chunks, max_chunk_size)
    comp_chunk_size = codec._algo.get_compress_chunk_size(max_chunk_size)

    # Prepare data and size buffers.
    # uncomp_chunks is used as a container that stores pointers to actual chunks.
    # nvCOMP requires this and sizes buffers to be in GPU memory.
    print("preparing data and size buffers")
    uncomp_chunks = cp.array(np.cumsum(buf_sizes) + buf.data.ptr, dtype=cp.)
    uncomp_chunk_sizes = cp.array(buf_sizes, dtype=cp.uint64)

    temp_buf = cp.empty(temp_size, dtype=cp.uint8)

    comp_chunks = cp.empty((num_chunks, comp_chunk_size), dtype=cp.uint8)
    # Array of pointers to each compressed chunk.
    comp_chunk_ptrs = cp.array([c.data.ptr for c in comp_chunks], dtype=cp.uintp)
    # Resulting compressed chunk sizes.
    comp_chunk_sizes = cp.empty(num_chunks, dtype=cp.uint64)

    print("calling compress")
    codec._algo.compress(
        uncomp_chunks,
        uncomp_chunk_sizes,
        max_chunk_size,
        num_chunks,
        temp_buf,
        comp_chunk_ptrs,
        comp_chunk_sizes,
        codec._stream,
    )

    # Copy to host to subsequently avoid many smaller D2H copies.
    comp_chunk_sizes = cp.asnumpy(comp_chunk_sizes, codec._stream) # copy gpu -> cpu
    codec._stream.synchronize()

    return comp_chunk_sizes[:num_chunks]


def ncd(len_x1, len_x2, len_x1x2):
    return (len_x1x2 - min(len_x1, len_x2)) / max(len_x1, len_x2)


# creating training set
XS = []
YS = []
for i in range(n_train):
    x, y = get_example("train")
    for token_idx in range(n_ctx):
        context = x[:token_idx + 1]
        target = y[token_idx]
        # print(f"when context is '{decode(context.tolist())}', target is '{decode([target.tolist()])}'")
        XS.append(context)
        YS.append(target)

with Timing("compressing examples"):
    XS_compressed_lens = encode_batch_size(gpu_compressor, XS)

print(f"Total examples: {len(XS_compressed_lens)}")


# making the model

compressed_pairs = np.empty((len(XS), len(XS)), dtype=np.float16) # those are the compressed lengths
STEP = 10
print(f"compressing {STEP} lines of {len(XS)} elements at a time (total {STEP*len(XS)} pairs each time)")
with Timing("compressing pairs.."):
    for i in tqdm.tqdm(range(0, len(XS), STEP)):
        chunk = XS[i: i + STEP]
        data = [np.concatenate((x1, np.array([0], dtype=np.uint8), x2)) for x1 in chunk for x2 in XS]

        compressed_chunk = encode_batch_size(gpu_compressor, data)
        # compressed_chunk = cp.array(list(encode_batch_size_chunked(gpu_compressor, data, batch_size=10000)))
        compressed_pairs[i: i + STEP] = compressed_chunk.reshape((STEP, len(XS)))

#ncd_scores = np.empty((len(XS), len(XS)), dtype=np.float32) # those are the normalized compression distances
ncd_scores = compressed_pairs
with Timing("computing ncds"):
    for i in range(len(compressed_pairs)):
        for j in range(len(compressed_pairs)):
            compressed_pairs[i, j] = ncd(XS_compressed_lens[i], XS_compressed_lens[j], compressed_pairs[i, j])
            # ncd_scores[i, j] = ncd(XS_compressed_lens[i], XS_compressed_lens[j], compressed_pairs[i, j])

print(ncd_scores)


def generate(knn: KNeighborsClassifier, context: np.ndarray, max_new_tokens: int, streaming=False, temperature=1.0,
             top_k=None):
    if streaming:
        print(decode(context), end="")
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
        # crop idx to the last block_size tokens
        idx_cond = context[-n_ctx:]
        idx_cond_compressed = encode_batch_size(gpu_compressor, [idx_cond.tobytes()])[0]

        cmps = encode_batch_size(gpu_compressor, [np.concatenate((idx_cond.tobytes(), np.array([0], dtype=np.uint8), x2)) for x2 in XS])
        _ncd_scores = np.array([ncd(idx_cond_compressed, XS_compressed_lens[i], cmps[i]) for i in range(len(XS_compressed_lens))])

        # get the predictions
        probs: np.ndarray = knn.predict_proba([_ncd_scores])

        # pluck the logits at the final step and scale by desired temperature
        idx_next = np.random.choice(knn.classes_, 1, p=probs[0])  # (B, 1)
        # append sampled index to the running sequence
        context = np.concatenate((context, idx_next))  # (B, T+1)
        if streaming:
            print(decode(idx_next), end="", flush=True)
    if streaming:
        print("\n")
    return context


# ncd_scores = np.array(ncd_scores, dtype=np.float32)
YS = np.array(YS, dtype=np.uint8)

neigh = KNeighborsClassifier(n_neighbors=7)
neigh.fit(ncd_scores, YS)
np.save("model_ncd_scores", ncd_scores)
np.save("model_ys", YS)

neigh.classes_ = np.arange(n_vocab, dtype=np.int64)

prompt = "Citi"
context = np.array(encode(prompt), dtype=np.uint8)

print(f"prompt: {prompt}")
print(f"number of tokens in the prompt = {len(context)}")
for token in context:
    print(f"{token:5} -> '{decode([token])}'")
print()

open('output.txt', 'w').write(decode(generate(neigh, context, max_new_tokens=200, streaming=True)))