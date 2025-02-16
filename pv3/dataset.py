from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import json


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
    def documents(self):
        return list(self.sentences.items())

    @property
    def all_sentences(self) -> list[str]:
        return [sentence for _, sentence in self.documents]

    @property
    def all_sentence_ids(self) -> list[str]:
        return [sid for sid, _ in self.documents]

    def sample(self, n: int = 1, random_state: int | None = None) -> pd.DataFrame:
        return self.rels.sample(n, random_state=random_state)['fact_id'].array

    def rel_true_for(self, query_id: str) -> list[str]:
        return self.rels[self.rels['fact_id'] == query_id]['sentence_id'].tolist()
