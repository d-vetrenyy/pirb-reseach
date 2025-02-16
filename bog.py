from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
import pandas as pd
import numpy as np
from pathlib import Path
import json
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class Dataset:
    facts: dict[str, str]
    sentences: dict[str, str]
    rels: pd.DataFrame

    def from_filesystem(path: str|Path):
        path = Path(path)
        with open(path / 'facts.json', 'r') as facts_file:
            facts = json.load(facts_file)
        with open(path / 'sentences.json') as sentences_file:
            sentences = json.load(sentences_file)
        rels = pd.read_csv(path / 'rels.tsv', delimiter='\t', header=None, names=['fact', 'sentence', 'score'])
        return Dataset(facts, sentences, rels)

    @property
    def documents(self) -> list[str]:
        return list(self.sentences.values())

    def sample(self, n: int = 1, random_state: int | None = None) -> pd.DataFrame:
        return self.rels.sample(n, random_state=random_state)['fact'].array

    def relevant_for(self, fact_id: str) -> pd.DataFrame:
        return self.rels[self.rels['fact'] == fact_id][['sentence', 'score']]


dataset: Dataset = Dataset.from_filesystem('./corpus')

query_id = dataset.sample(n=1)[0]
query_text = dataset.facts[query_id]
known_relevant = set(dataset.relevant_for(query_id))

print("Query:", query_text)
print("Known relevant:", known_relevant)

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', token=HF_TOKEN, trust_remote_code=True)
doc_embeds = model.encode(dataset.documents, show_progress_bar=True, output_value='sentence_embedding', convert_to_numpy=True)
query_embed = model.encode(query_text, show_progress_bar=True, output_value='sentence_embedding', convert_to_numpy=True)

similarity = cosine_similarity([query_embed], doc_embeds)[0]
retrieved_relevant = set(np.where(similarity > 0.5)[0])

print("Retrieved relevant: ", retrieved_relevant)

tp = len(retrieved_relevant.intersection(known_relevant))
precision = tp / len(retrieved_relevant) if len(retrieved_relevant) > 0 else 0
recall = tp / len(known_relevant) if len(known_relevant) > 0 else 0

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
