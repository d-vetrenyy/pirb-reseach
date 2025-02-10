from pirb.corpus import CorpusLoader
from sentence_transformers import SentenceTransformer
import numpy as np
import torch.cuda
import sys


corpus = CorpusLoader("./corpus")

model = SentenceTransformer(
    "thenlper/gte-large",
    token=sys.argv[1])


if __name__ == '__main__':
    corpus_embeddings = model.encode(list(corpus.sentences.values()))

    query = corpus.facts[corpus.sample(n=1)[0]]
    query_embeddings = model.encode(query)

    similarities = model.similarity(corpus_embeddings, query_embeddings)
    print(similarities)
