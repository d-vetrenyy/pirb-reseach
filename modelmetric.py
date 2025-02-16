import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from transformers import AutoTokenizer, AutoModel
import torch

class CorpusLoader:
    def __init__(self, facts, sentences, relations):
        self.facts = facts  # dict of fact_id to fact
        self.sentences = sentences  # dict of sentence_id to sentence
        self.relations = relations  # DataFrame with fact_id, sentence_id, score

class InformationRetrieval:
    def __init__(self, corpus_loader):
        self.corpus_loader = corpus_loader
        self.model_name = 'sentence-transformers/all-MiniLM-L6-v2'  # Example model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.sentence_embeddings = self._generate_embeddings()

    def _generate_embeddings(self):
        sentences = list(self.corpus_loader.sentences.values())
        inputs = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            embeddings = self.model(**inputs).last_hidden_state[:, 0, :]  # Use [CLS] token
        return embeddings.numpy()

    def query(self, fact):
        fact_embedding = self._embed_fact(fact)
        similarities = self._compute_similarities(fact_embedding)
        return similarities

    def _embed_fact(self, fact):
        inputs = self.tokenizer(fact, return_tensors='pt')
        with torch.no_grad():
            embedding = self.model(**inputs).last_hidden_state[:, 0, :]
        return embedding.numpy()

    def _compute_similarities(self, fact_embedding):
        # Choose your similarity method here
        # Example: Cosine Similarity
        similarities = cosine_similarity(fact_embedding, self.sentence_embeddings)
        return similarities.flatten()

    def evaluate(self, retrieved_sentences, threshold=0.5):
        # Convert retrieved sentences to a set for evaluation
        retrieved_set = set(retrieved_sentences)
        true_positive = 0
        false_positive = 0
        false_negative = 0

        for _, row in self.corpus_loader.relations.iterrows():
            if row['sentence_id'] in retrieved_set:
                if row['score'] >= threshold:
                    true_positive += 1
                else:
                    false_positive += 1
            else:
                if row['score'] >= threshold:
                    false_negative += 1

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return precision, recall, f1_score

# Example usage
facts = {'fact1': 'This is a fact.', 'fact2': 'This is another fact.'}
sentences = {'sentence1': 'This is a related sentence.', 'sentence2': 'This is not related.'}
relations = pd.DataFrame({'fact_id': ['fact1', 'fact1', 'fact2'], 'sentence_id': ['sentence1', 'sentence2', 'sentence1'], 'score': [1, 0, 1]})

corpus_loader = CorpusLoader(facts, sentences, relations)
info_retrieval = InformationRetrieval(corpus_loader)

# Example query with a random fact
fact_to_query = 'This is a fact.'
similarities = info_retrieval.query(fact_to_query)

# Get the top N sentences based on similarity scores
N = 2  # Number of top sentences to retrieve
top_indices = np.argsort(similarities)[-N:][::-1]  # Get indices of top N sentences
top_sentences = [list(corpus_loader.sentences.values())[i] for i in top_indices]

print("Top retrieved sentences:")
for sentence in top_sentences:
    print(sentence)

# Evaluate the retrieved sentences against the relations DataFrame
# For simplicity, we will consider a threshold of 0.5 for relevance
threshold = 0.5
retrieved_sentences = [list(corpus_loader.sentences.keys())[i] for i in top_indices]  # Get sentence IDs
precision, recall, f1_score = info_retrieval.evaluate(retrieved_sentences, threshold)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1_score:.2f}")
