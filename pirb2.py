import torch
import numpy as np
import pandas as pd
import json
from dataclasses import dataclass
from pathlib import Path
from transformers import AutoTokenizer, AutoModel


@dataclass
class Dataset:
    facts: dict[str, str]
    sentences: dict[str, str]
    rels: pd.DataFrame # [fact_id, sent_id, score]

    @staticmethod
    def from_filesystem(path: str|Path):
        path = Path(path)
        facts_path = path / "facts.json"
        sentences_path = path / "sentences.json"
        rels_path = path / "rels.tsv"

        with open(facts_path, 'r') as facts_file:
            facts = json.load(facts_file)
        with open(sentences_path, 'r') as sentences_file:
            sentences = json.load(sentences_file)
        rels = pd.read_csv(rels_path, delimiter='\t', header=None, names=['fact', 'sentence', 'score'])
        return Dataset(facts, sentences, rels)

    @property
    def documents(self) -> list[str]:
        return list(self.sentences.values())

    def get_sentences(self, sent_ids: list[str]):
        result = {}
        for sent_id in sent_ids:
            result[sent_id] = self.sentences[sent_id]
        return result

    def sample(self, n: int = 1, random_state: int | None = None) -> pd.DataFrame:
        return self.rels.sample(n, random_state=random_state)['fact'].array

    def random_fact(self) -> str:
        return self.facts[self.sample(n=1)[0]]

    def expected_for(self, fact_id: str) -> pd.DataFrame:
        return self.rels[self.rels['fact'] == fact_id][['sentence', 'score']]


class RetrieverModel:
    def __init__(self, model_name: str, documents: list[str]|None = None, similarity_func = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        self._embeddings = np.array([])
        if documents != None:
            self.embeddings = self.encode_documents(documents)

        self._similarity_func = similarity_func

    def encode_documents(self, documents: list[str]) -> np.ndarray:
        inputs = self.tokenizer(documents, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            embeddings = self.model(**inputs).last_hidden_state[:, 0, :]
        return embeddings.numpy()

    def embed_dataset(self, dataset: Dataset):
        self._embeddings = self.encode_documents(list(dataset.sentences.values()))

    def encode_query(self, query: str) -> np.ndarray:
        inputs = self.tokenizer(query, return_tensors='pt')
        with torch.no_grad():
            embedding = self.model(**inputs).last_hidden_state[:, 0, :]
        return embedding.numpy()

    def search_query(self, query: str):
        query_embeddings = self.encode_query(query)
        similarities = self._similarity_func(query_embeddings, self._embeddings)
        return similarities
