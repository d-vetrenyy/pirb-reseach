import torch
import numpy as np


class Evaluator:
    def __init__(self, embeddings_by_documents: np.ndarray, query_embeddings: np.ndarray):
        self.embeds_by_docs = embeddings_by_documents
        self.query_embeds = query_embeddings

    def dot_product(self) -> np.ndarray:
        return np.dot(self.embeds_by_docs, self.query_embeds.T)
        # return torch.mm(self.embeds_by_docs, self.query_embeds.T)

    def cosine_similarity(self) -> np.ndarray:
        norm_doc = np.linalg.norm(self.embeds_by_docs, axis=1, keepdims=True)
        norm_query = np.linalg.norm(self.query_embeds)
        # norm_doc = self.embeds_by_docs.norm(dim=1, keepdim=True)
        # norm_query = self.query_embeds.norm()
        return self.dot_product() / (norm_doc * norm_query)

    def euclidean_distance(self) -> np.ndarray:
        return np.linalg.norm(self.embeds_by_docs - self.query_embeds, axis=1)
        # return torch.norm(self.embeds_by_docs - self.query_embeds, dim=1)
