from pirb.corpus import CorpusLoader
import pirb.evals as evals
from sentence_transformers import SentenceTransformer
import numpy as np
import sys

# FAISS -- библиотека facebook AI
# сравнительная таблица модели/методы сравнения
# ACL.ANTHOLOGY.ORG: acl, emnlp -- publications
# ARXIV.ORG -- benchmark (latest years)
# посмотреть как представляются бенчмарки
# sk-learn посмотреть реализацию меткрики (report)
# [!!] 10 моделей * sin, dot, euclean + TF-IDF + OkamiB25 (rrcall, precision, fall-out)

SIMILARITY_THRESHOD = 0.7


corpus = CorpusLoader("./corpus")

model = SentenceTransformer(
    "thenlper/gte-large",
    token=sys.argv[1])


if __name__ == '__main__':
    corpus_embeddings = model.encode(list(corpus.sentences.values()))

    sample = corpus.sample(n=1)[0]
    query = corpus.facts[sample]
    print('Query:', query)
    y = corpus.expected_for(sample)
    print('Expected results:', y)
    query_embeddings = model.encode(query)

    similarities = model.similarity(corpus_embeddings, query_embeddings).numpy()
    relevant_ids = similarities >= SIMILARITY_THRESHOD
    relevant = list(corpus.get_sentences(relevant_ids).values())
    precision = evals.precision(relevant, y)
    recall = evals.recall(relevant, y)

    # Как осуществить оценку ответов модели задействуя назначенную ей оценку: не совпадает с оценкой присвоенной в таблице

    print(relevant)
    print(precision, recall)
