import pickle
import time
from typing import Self, List
from tokenizer import tokenizer


class GPTeirb:
    tokenizer = tokenizer
    def __init__(self):
        pass

    @classmethod
    def from_weights(cls, path: str) -> Self:
        with open(path, "rb") as f:
            return pickle.load(f)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def __call__(self, tokens: List[int], *args, **kwargs) -> int:
        # TODO: implement forward
        time.sleep(0.01)
        return tokenizer.encode("owo")[1]
