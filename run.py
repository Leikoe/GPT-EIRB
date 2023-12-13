#!/usr/bin/env python3
import argparse
from model import GPTeirb
from utils import Timing

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run GPTeirb",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--count", type=int, default=30, help="Max number of tokens to generate")
    parser.add_argument("--timing", action="store_true", help="Print timing per token")
    parser.add_argument("--weights", type=str, default="./gpteirb.bin", help="Path to the downloaded weights")
    args = parser.parse_args()

    model = GPTeirb()
    # model = GPTeirb.from_weights(args.weights)

    toks = [model.tokenizer.bos_id()]
    start_pos = 0
    for i in range(args.count):
        with Timing("total ", enabled=args.timing, on_exit=lambda x: f", {1e9 / x:.2f} tok/sec"):
            tok = model(toks[start_pos:])
        toks.append(tok)
        start_pos += 1
        print(model.tokenizer.decode(toks))
