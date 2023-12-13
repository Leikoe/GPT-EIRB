#!/usr/bin/env python3
import argparse
from typing import Optional, Tuple
import numpy as np
from model import GPTeirb
from utils import Timing


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None) -> Tuple[int, np.ndarray[np.float32]]:
    probs = None
    idx_next = int(np.argmax(logits))
    return idx_next, probs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run GPTeirb",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--count", type=int, default=30, help="Max number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature in the softmax")
    parser.add_argument("--timing", action="store_true", help="Print timing per token")
    parser.add_argument("--weights", type=str, default="./gpteirb.bin", help="Path to the downloaded weights")
    args = parser.parse_args()

    model = GPTeirb([[]], [])   # TODO: remove this once we have a model to run
    # model = GPTeirb.from_weights(args.weights)

    toks = [model.tokenizer.bos_id()]
    start_pos = 0
    for i in range(args.count):
        with Timing("total ", enabled=args.timing, on_exit=lambda x: f", {1e9 / x:.2f} tok/sec"):
            tok, _ = sample(model(toks[start_pos:]), temperature=args.temperature)
        toks.append(tok)
        start_pos += 1
        print(model.tokenizer.decode(toks))
