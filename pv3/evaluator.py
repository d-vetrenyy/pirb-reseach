import torch
import numpy as np


class Evaluator:
    def __init__(self, embeddings_by_documents: torch.Tensor, query_embeddings: torch.Tensor):
        self.embeds_by_docs = embeddings_by_documents
        self.query_embeds = query_embeddings

    def dot_product(self) -> torch.Tensor:
        return torch.mm(self.embeds_by_docs, self.query_embeds.T)

    def cosine_similarity(self) -> torch.Tensor:
        norm_doc = self.embeds_by_docs.norm(dim=1, keepdim=True)
        norm_query = self.query_embeds.norm()
        return self.dot_product() / (norm_doc * norm_query)

    def euclidean_distance(self) -> torch.Tensor:
        return torch.norm(self.embeds_by_docs - self.query_embeds, dim=1)
