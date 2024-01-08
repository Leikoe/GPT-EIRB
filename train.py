#
#   takes a dataset generated from download.py (.txt) and creates a model
#

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

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


print(f"{SUPPORTED_ALGORITHMS=}")

LZ4_ALGO = "LZ4"
GDEFLATE_ALGO = "Gdeflate"
SNAPPY_ALGO = "snappy"
ZSTD_ALGO = "zstd"

def _get_codec(algo: str, **kwargs):
    codec_args = {"id": "nvcomp_batch", "algorithm": algo, "options": kwargs}
    return numcodecs.registry.get_codec(codec_args)

gpu_compressor = _get_codec(GDEFLATE_ALGO)



# hyperparameters
n_train = 10000
n_batch = 1
n_ctx = 64  # what is the maximum context length for predictions?
temp = 0.800000
top_k = 40
top_p = 0.950000
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

# Train and test splits
data = cp.array(encode(text), dtype=cp.uint8)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


print(f"n_ctx   = {n_ctx}")
print(f"n_vocab = {len(chars)}")
print(f"train_data.shape = {train_data.shape}")
print(f"val_data.shape   = {val_data.shape}")


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = np.random.randint(0, len(data) - n_ctx, (n_batch,))
    x = np.stack([data[i:i+n_ctx] for i in ix])
    y = np.stack([data[i+1:i+n_ctx+1] for i in ix])
    return x, y

def get_example(split):
    # generate an example of input x and target y
    data = train_data if split == 'train' else val_data
    i = np.random.randint(0, len(data) - n_ctx)
    x = data[i:i+n_ctx]
    y = data[i+1:i+n_ctx+1]
    return x, y

   
def encode_batch_size(codec: NvCompBatchCodec, bufs: Sequence[Any]) -> Sequence[Any]: 
    num_chunks = len(bufs)
    # chunks_size = sum(map(len, bufs)) # map each buffer to it's size, then sum
    # print(f"[encode_batch_size] chunk_size = {humanize.naturalsize(chunks_size)}")
    # print(f"[encode_batch_size] num_chunks = {num_chunks}")
    if num_chunks == 0:
        return []

    bufs = [cp.asarray(ensure_contiguous_ndarray_like(b)) for b in bufs]
    buf_sizes = [b.size * b.itemsize for b in bufs]

    max_chunk_size = max(buf_sizes)

    # Get temp and output buffer sizes.
    temp_size = codec._algo.get_compress_temp_size(num_chunks, max_chunk_size)
    comp_chunk_size = codec._algo.get_compress_chunk_size(max_chunk_size)

    # Prepare data and size buffers.
    # uncomp_chunks is used as a container that stores pointers to actual chunks.
    # nvCOMP requires this and sizes buffers to be in GPU memory.
    uncomp_chunks = cp.array([b.data.ptr for b in bufs], dtype=cp.uintp)
    uncomp_chunk_sizes = cp.array(buf_sizes, dtype=cp.uint64)

    temp_buf = cp.empty(temp_size, dtype=cp.uint8)

    comp_chunks = cp.empty((num_chunks, comp_chunk_size), dtype=cp.uint8)
    # Array of pointers to each compressed chunk.
    comp_chunk_ptrs = cp.array([c.data.ptr for c in comp_chunks], dtype=cp.uintp)
    # Resulting compressed chunk sizes.
    comp_chunk_sizes = cp.empty(num_chunks, dtype=cp.uint64)

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

    res = []
    # Copy to host to subsequently avoid many smaller D2H copies.
    # comp_chunks = cp.asnumpy(comp_chunks, codec._stream)
    comp_chunk_sizes = cp.asnumpy(comp_chunk_sizes, codec._stream)
    codec._stream.synchronize()

    for i in range(num_chunks):
        # res.append(comp_chunks[i, : comp_chunk_sizes[i]].tobytes())
        res.append(comp_chunk_sizes[i])
    return res


def encode_batch_size_chunked(codec: NvCompBatchCodec, bufs: Sequence[Any]) -> Sequence[Any]:
    return flatten(map(lambda chunk: encode_batch_size(codec, chunk), chunks(bufs, 100)))


def ncd_fast(x: bytes, x_compressed: int, x2: bytes, x2_compressed: int):  # NCD with compressed lengths
    xx2 = len(gpu_compressor.encode(b" ".join([x, x2])))
    return (xx2 - min(x_compressed, x2_compressed)) / max(x_compressed, x2_compressed)


XS = []
YS = []
for i in range(n_train):
    x, y = get_example("train")
    for token_idx in range(n_ctx):
        context = x[:token_idx + 1]
        target = y[token_idx]
        print(f"when context is '{decode(context.tolist())}', target is '{decode([target.tolist()])}'")
        XS.append(context.tobytes())
        YS.append(target)

exit(0)

with Timing("compressing examples: "):
    XS_compressed_lens = list(encode_batch_size_chunked(gpu_compressor, XS))
print(f"Total examples: {len(XS_compressed_lens)}")

print("compressing pairs..")
compressed_pairs = []

for x1 in XS_compressed_lens:
    print(x1)

exit(0)

for chunk in tqdm.tqdm(chunks(XS, 10), desc="compressing pairs"):
    ee = [b"".join([x1, x2]) for x1 in chunk for x2 in XS]
    compressed_pairs.extend(encode_batch_size_chunked(gpu_compressor, ee))

print(compressed_pairs[:10])

exit(0)

aa = zip(map(len, compressed_pairs), [(x_compressed, x2_compressed) for x_compressed in map(len, XS_compressed) for x2_compressed in map(len, XS_compressed)])
train_ncd = [(xx2 - min(x_compressed, x2_compressed)) / max(x_compressed, x2_compressed) for (xx2, (x_compressed, x2_compressed)) in aa]
print(len(train_ncd))
print("DONEEEEE")

# def nomnom_fast(i):
    # return [ncd_fast(*X[i], *X[j]) for j in range(len(X))]


# train_ncd = [nomnom_fast(i) for i in tqdm.tqdm(range(len(X)), desc="creating model")]

# remote_tqdm = ray.remote(tqdm_ray.tqdm)
# bar = remote_tqdm.remote(total=len(X), desc="creating model")
# train_ncd = ray.get([nomnom_fast.remote(bar, i) for i in range(len(X))])
# bar.close.remote()
# time.sleep(0.1)
# ray.shutdown()


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def inv_softmax(x, C):
    return np.log(x) + C


def generate(knn: KNeighborsClassifier, context: np.ndarray, max_new_tokens: int, streaming=False, temperature=1.0,
             top_k=None):
    if streaming:
        print(tokenizer.decode(context), end="")
    # idx is (B, T) array of indices in the current context
    for _ in range(max_new_tokens):
        # crop idx to the last block_size tokens
        idx_cond = context[-n_ctx:]
        idx_cond_compressed = len(gpu_compressor.encode(idx_cond.tobytes()))

        # get the predictions
        ncd_scores = np.array([ncd_fast(idx_cond.tobytes(), idx_cond_compressed, *x2) for x2 in X])
        probs: np.ndarray = knn.predict_proba([ncd_scores])

        # pluck the logits at the final step and scale by desired temperature
        probs = inv_softmax(probs + 0.1, 0)
        probs /= temperature
        probs = softmax(probs)
        # sample from the distribution
        if top_k is not None:
            topk_idxs = (-probs).argsort()[:top_k]
            idx_next = np.random.choice(knn.classes_[topk_idxs], 1, p=probs[0][topk_idxs])  # (B, 1)
        else:
            idx_next = np.random.choice(knn.classes_, 1, p=probs[0])  # (B, 1)
        # append sampled index to the running sequence
        context = np.concatenate((context, idx_next))  # (B, T+1)
        if streaming:
            print(tokenizer.decode(idx_next), end=" ")
    return context


neigh = KNeighborsClassifier(n_neighbors=7)
neigh.fit(train_ncd, Y)
np.save("model", [train_ncd, Y])
# neigh.classes_ = np.arange(n_vocab, dtype=np.int64)

prompt = "Speak"
context = np.array(tokenizer.encode(prompt), dtype=np.int64)

print(f"prompt: {prompt}")
print(f"number of tokens in the prompt = {len(context)}")
for token in context:
    print(f"{token:5} -> '{tokenizer.decode(token)}'")
print()
print(f"sampling parameters: temp = {temp:6f}, top_k = {top_k}, top_p = {top_p:6f}")
print()
print()

open('output.txt', 'w').write(tokenizer.decode(generate(neigh, context, max_new_tokens=200, streaming=True)))