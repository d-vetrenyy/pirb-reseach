import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dataclasses import dataclass
from pathlib import Path
import json


MODEL_LIST = ['sentence-transformers/all-MiniLM-L6-v2']


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
        rels = pd.read_csv(rels_path, delimiter='\t', header=None, names=['fact_id', 'sentence_id', 'score'])
        return Dataset(facts, sentences, rels)

    @property
    def documents(self) -> list[str]:
        return list(self.sentences.values())

    def sample(self, n: int = 1, random_state: int | None = None) -> pd.DataFrame:
        return self.rels.sample(n, random_state=random_state)['fact_id'].array

    def relevant_for(self, query_id: str) -> pd.DataFrame:
        return self.rels[self.rels['fact_id'] == query_id][['sentence_id', 'score']]


dataset = Dataset.from_filesystem('./corpus')

query_id = dataset.sample()[0]
query = dataset.facts[query_id]
relevant_true_ids = dataset.relevant_for(query_id)['sentence_id'].tolist()
print('Query:', query)
print('Known relevant:')
for relevant_true_id in relevant_true_ids:
    print('\t', dataset.sentences[relevant_true_id])

for model_name in MODEL_LIST:
    model = SentenceTransformer(model_name)
    doc_embeddings = model.encode(dataset.documents, show_progress_bar=True, output_value='sentence_embedding', convert_to_tensor=True)
    query_embedding = model.encode(query, show_progress_bar=True, output_value='sentence_embedding', convert_to_tensor=True)

    similarities = util.pytorch_cos_sim(query_embedding, doc_embeddings)[0]
    relevant_pred = {sid: dataset.sentences[sid] for sid, score in zip(dataset.sentences.keys(), similarities) if score > 0.7}

    print("Model retrieved:")
    for retrieved_relevant in relevant_pred.values():
        print(retrieved_relevant)

    relevant_pred_ids = list(relevant_pred.keys())

    tp = len(set(relevant_pred_ids).intersection(set(relevant_true_ids)))
    fp = len(set(relevant_pred_ids) - set(relevant_true_ids))
    fn = len(set(relevant_true_ids) - set(relevant_pred_ids))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    print(f"{precision = }")
    print(f"{recall = }")
