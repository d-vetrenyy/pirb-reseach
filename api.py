from sentence_transformers import SentenceTransformer
import sys

model = SentenceTransformer(
    "thenlper/gte-large",
    token=sys.argv[1])

sentences = [
    "That is a happy person",
    "That is a happy dog",
    "That is a very happy person",
    "Today is a sunny day"
]
embeddings = model.encode(sentences)
print(embeddings, '\n', embeddings.shape)
query = model.encode(["who is very happy?"])
print(query, '\n', query.shape)

similarities = model.similarity(embeddings, query)
print(similarities.numpy(force=True))
