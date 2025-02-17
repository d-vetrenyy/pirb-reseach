from .dataset import Dataset
from . import evals
import numpy as np
import pandas as pd


class Benchmark:
    def __init__(self, rel_true, similarity_scores, threshold=0.5):
        self.rel_true = rel_true
        self.rel_pred = np.where(similarity_scores > threshold)[0]
