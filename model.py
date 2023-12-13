import pickle
import time
from typing import Self, List
from sklearn.neighbors import KNeighborsClassifier

from tokenizer import tokenizer


class GPTeirb:
    tokenizer = tokenizer

    def __init__(self, train_ncd: List[List[float]], Y: List[int]):
        self.neigh = KNeighborsClassifier(n_neighbors=7)
        self.neigh.fit(train_ncd, Y)

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
