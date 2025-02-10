# from sentence_transformers import SentenceTransformer
# import sys

# model = SentenceTransformer(
#     "thenlper/gte-large",
#     token=sys.argv[1])

# sentences = [
#     "That is a happy person",
#     "That is a happy dog",
#     "That is a very happy person",
#     "Today is a sunny day"
# ]
# embeddings = model.encode(sentences)
# print(embeddings, '\n', embeddings.shape)
# query = model.encode(["who is very happy?"])
# print(query, '\n', query.shape)

# similarities = model.similarity(embeddings, query)
# print(similarities.numpy(force=True))

from pirb.corpus import CorpusLoader
from sentence_transformers import SentenceTransformer
import numpy as np
import sys


corpus = CorpusLoader("./corpus")

model = SentenceTransformer(
    "thenlper/gte-large",
    token=sys.argv[1])


if __name__ == '__main__':
    corpus_embeddings = model.encode(list(corpus.sentences.values()))

    query = corpus.facts[corpus.sample(n=1)[0]]
    query_embeddings = model.encode(query)

    similarities = model.similarity(corpus_embeddings, query_embeddings).numpy()
    print(similarities)
