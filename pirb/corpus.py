from pathlib import Path
import json
import random
import pandas as pd


class CorpusManipulator:
    FACTS_SUFFIX     = "facts.json"
    SENTENCES_SUFFIX = "sentences.json"
    RELS_SUFFIX      = "rels.tsv"
    def __init__(self, corpus_path: Path | str):
        self.corpus_path = Path(corpus_path)

    def _open_file(self, suffix, mode):
        return open(self.corpus_path / Path(suffix), mode)


class CorpusBuilder(CorpusManipulator):
    def __init__(self, corpus_path: Path | str):
        super().__init__(corpus_path)
        self.facts_file = self._open_file(super().FACTS_SUFFIX, "r+")
        self.sentences_file = self._open_file(super().SENTENCES_SUFFIX, "r+")
        self.rels_file = self._open_file(super().RELS_SUFFIX, "r+")

    def __del__(self):
        self.facts_file.close()
        self.sentences_file.close()
        self.rels_file.close()


class CorpusLoader(CorpusManipulator):
    def __init__(self, corpus_path: Path | str):
        super().__init__(corpus_path)
        facts_file = self._open_file(super().FACTS_SUFFIX, "r")
        sentences_file = self._open_file(super().SENTENCES_SUFFIX, "r")
        rels_file = self._open_file(super().RELS_SUFFIX, "r")
        self.facts = json.load(facts_file)
        self.sentences = json.load(sentences_file)
        # self.rels = tuple(csv.reader(rels_file, delimiter='\t'))
        self.rels = pd.read_csv(rels_file, delimiter='\t', header=None, names=['fact', 'sentence', 'score'])


    def find_fact(self, fact: str) -> str:
        return [k for k, v in self.facts.items() if v == fact][0]


    def get_facts(self, fact_ids: list[str]):
        result = {}
        for fact_id in fact_ids:
            result[fact_id] = self.facts[fact_id]
        return result


    def sample(self, n: int = 1, random_state: int | None = None) -> pd.DataFrame:
        return self.rels.sample(n, random_state=random_state)['fact'].array


    def expected_for(self, fact_id: str) -> pd.DataFrame:
        return self.rels[self.rels['fact'] == fact_id][['sentence', 'score']]
