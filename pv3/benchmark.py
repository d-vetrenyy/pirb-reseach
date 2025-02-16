import numpy as np
import pandas as pd


class Benchmark:
    def __init__(self, rel_true: list[str], similarities: np.ndarray, threshold=0.5):
        self.rel_true = rel_true
        self.rel_pred = np.where(similarities > threshold)[0]
