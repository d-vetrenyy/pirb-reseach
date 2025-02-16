from sentence_transformers import SentenceTransformer
import numpy as np
import pirb.evals as evals
from pirb.corpus import CorpusLoader
from sklearn.metrics.pairwise import cosine_similarity


corpusloader = CorpusLoader('./corpus')
print("[INFO]\tCorpus was loaded")

model = SentenceTransformer('all-MiniLM-L6-v2', backend='torch')
print("[INFO]\tsentence transformer was initialized")

documents = list(corpusloader.sentences.values())
doc_embeddings = model.encode(documents, show_progress_bar=True, output_value='sentence_embedding', convert_to_numpy=True, batch_size=64)

query_id = corpusloader.sample()[0]
query_text = corpusloader.facts[query_id]
y_true = corpusloader.expected_for(query_id)
print(f"[INFO] Query - {query_text}")

for i, sent_id in enumerate(y_true['sentence']):
    sent = corpusloader.sentences[sent_id]
    print(f"Expected sentence[{i}]: {sent}")

query_embedding = model.encode([query_text], show_progress_bar=True, output_value='sentence_embedding', convert_to_numpy=True)

similarities = cosine_similarity(query_embedding, doc_embeddings)
closest = np.argmax(similarities)

print(documents[closest])
