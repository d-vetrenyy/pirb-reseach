from .dataset import Dataset
from . import evals
import numpy as np
import pandas as pd


class Benchmark:
    def __init__(self, dset: Dataset, fact_id: str, rel_pred: np.ndarray):
        rel_true_ids = dset.rel_true_for(fact_id)

        rel_true_sentences = [dset.sentences[sid] for sid in rel_true_ids if sid in dset.sentences]
        rel_pred_sentences = [dset.all_sentences[i] for i in rel_pred if i < len(dset.all_sentences)]

        self.precision = evals.precision(rel_pred_sentences, rel_true_sentences)
        self.recall = evals.recall(rel_pred_sentences, rel_true_sentences)
        self.f1_score = evals.fbeta(1.0, self.precision, self.recall)



    def rel_pred(similarity_scores: np.ndarray, threshold=0.5):
        return np.where(similarity_scores > threshold)[0]
