## Recall
a fundamental evaluation metric in information retrieval, measuring the ability of a system to retrieve all relevant documents or items from a collection

R = number_of_relevant_documents_retrieved / Total_number_of_relevant_documents_in_the_database
or
R = |relevant_docs & retrieved_docs| / |relevant_docs|

## Precision

P = |relevant_docs & retrieved_docs| / |retrieved_docs| where |X| — length of collection/set

### Presicion at k
P(k) — precision at top k documents (we need to sort all retrieved docs by similarity_score and take first k, then take P of that)

### Avarage precision (AP)
AP = 1/RD * Sigma(k=1, n, P(k) * r(k))
    where RD — number of relevant documents for the query,
    where n — total number of documents,
    where P(k) — precision at k
    where r(k) — relevance of k-th retrieved document (usually mapped to binary relevance (0 — not relevant, 1 — relevant))

## Mean Avarage Precision (MAP)

MAP = 1/N * Sigma(i=1, N, AP(i))
    where N — total number of queries
    where AP(i) — Avarage Precision of query

## Normalised Discounted Cumulative Gain (NDCG)

*Cumulative Gain* — the sum of the graded relevance values of all result in a search result list

CG(p) = Sigma(i=1, p, rel(i)) — where rel(i) = graded relevance of the result at position i

(!) This DOES NOT account for order of relevancy, so there is no penalty for putting more relevant results father in the list then less relevant ones

*To penalize highly relevant results appearing low on the list, it can be devided by logarithmically growing function of its position*

DCG(p)
    = Sigma(i=1, p, rel(i) / log2(i + 1))
    = rel(1) + Sigma(i=2, p, rel(i) / log2(i + 1))

Alternatively

DCG(p) = Sigma(i=1, p, ((2 ^ rel(i) - 1) / log2(i + 1))

And then normalisation is often applied:

*Search result lists vary in length depending on the query. Comparing a search engine's performance from one query to the next cannot be consistently achieved using DCG alone, so the cumulative gain at each position for a chosen value of p should be normalized across queries. This is done by sorting all relevant documents in the corpus by their relative relevance, producing the maximum possible DCG through position p, also called Ideal DCG (IDCG) through that position*

NDCG(p) = DCG(p) / IDCG(p)
    — where IDCG(p) = Sigma(i=1, |REL(p)|, ((2 ^ rel(i)) - 1) / log2(i + 1))
    — where REL(p) — list of relevant documents (ordered by their relevance) in the corpus up to position p

## fall-out
The proportion of non-relevant documents that are retrieved, out of all non-relevant documents available

fall-out = |non-relevant-docs & retrieved documents| / |non-relevant-docs|

## F-score / F-measure
The weighted harmonic mean of precision and recall

F(b) = ((1 + b^2) * (precision * recall)) / (b^2 * precision + recall)
    where b — weight: Real, determining recall/precision

## Dense Information Retrieval Model
Unlike sparse vector space models, such as TF-IDF or BM25, which represent documents as bag-of-words*, DIR models learn 'dense vector'* representations of documents and queries. These dense vectors capture the semantic relationships between queries and documents, enabling more accurate and robust retrieval

*bag-of-words (BoW) — model of text which uses a representation of text that is based on an unordered collection (a "bag") of words
e.g. "John likes to watch movies. Mary likes movies too. Mary also likes to watch football games."
    ==> {"John":1,"likes":3,"to":2,"watch":2,"movies":2,"Mary":2,"too":1,"also":1,"football":1,"games":1}

*dense vector — high-dimensional vector where most elements (features) hold significant values, unlike sparse vectors, which are predominantly composed of zeroes.
popular algorithms for generation: word2vec, BERT (bidirectional encoder representation for transformers) (both by Google)

## Lexical Model in IR
 a theoretical framework or mathematical representation that describes how words and their meanings are related and used in text documents. The goal of a lexical model is to capture the semantic relationships between words, enabling IR systems to better understand the meaning of queries and documents, and thus improve retrieval accuracy
Lexical models can be categorized into two main types:
1. Statistical models: These models represent word meanings based on statistical patterns in large text corpora. Examples include:
    - Bag-of-Words (BoW): represents documents as vectors of word frequencies.
    - Term Frequency-Inverse Document Frequency (TF-IDF): weights word frequencies by their rarity across documents.
2. Semantic models: These models incorporate knowledge about word meanings, often drawn from lexical resources such as dictionaries, thesauri, or ontologies. Examples include:
    - WordNet: a lexical database that organizes words into synonym sets (synsets) and hyponymy/hypernymy relationships.
    - Latent Semantic Analysis (LSA): represents word meanings as vectors in a high-dimensional space, based on co-occurrence patterns in a large corpus.


## Okapi BM25 (Best Matching)
ranking function used by search engines to estimate the relevance of documents given search query. It is based on the probabilistic retrieval framework developed in the 1970s and 1980s by Stephen E. Robertson, Karen Spärck Jones, and others.
BM25 is a bag-of-words retrieval function that ranks a set of documents based on the query terms appearing in each document, regardless of their proximity within the document. It is a family of scoring functions with slightly different components and parameters

given q[1] .. q[n] — keywords in Q — query and D — document:
score(D, Q) = Sigma(i=1, n, IDF(q[i])) * ((f(q[i], D) * (k[1] + 1)) / f(q[i], D) + k[1] * (1 - b + b * (|D|/avgdl)))
    where f(q[i], D) — frequency of q[i] in D
    where |D| — count of words in D
    where avgdl — average document length in words in corpus
    where k[1], b — free parameters, usually chosen; k in [1.2, 2.0], b = 0.75
    where IDF(q[i]) — inverse document frequency

IDF(q[i]) = ln((N - n(q[i]) + 0.5) / (n(q[i]) + 0.5) + 1)
    where N — total number of documents in corpus
    where n(q[i]) — number of documents containing q[i]

## Sparse Retrieval
A family of neural network-based information retrieval methods that transform queries and documents into sparse weight vectors aligned with a vocabulary. These vectors are designed to capture the most relevant terms or features for a given query-document pair, rather than dense, high-dimensional embeddings.
Typically used techniques:
1. Bag-of-Words (Bow): weighted sum of vocabulary terms, with non-zero weights indicating the presence of each term
2. Neural Sparse Models: Using neural networks to infer which vocabulary terms are relevant to a document, even if they’re not explicitly mentioned

## K-values in IR
a set of evaluation metrics used to assess the performance of an IR system. Specifically, K-values are used to measure the precision and recall of a system at different levels of retrieval, typically denoted as P@K or R@K

## Sparse Transformer
a modified version of the standard Transformer architecture, designed to reduce the time and memory complexity of attention mechanisms. This is achieved through sparse factorizations of the attention matrix, which decrease the computational cost from O(n^2) to O(n*sqrt(n))

## Dense Passage Retrieval (DPR)
 a technique for retrieving relevant passages from a large corpus of text based on a given query -- (??)

## Reranking
 enhancing the quality of search results by reordering documents based on their relevance to a query. It’s a second-stage evaluation, refining the initial retrieval results to prioritize the most relevant items

## Cross-Encoder
a type of neural network architecture used in natural language processing (NLP) tasks, particularly in information retrieval (IR). Its primary purpose is to evaluate and provide a single score or representation for a pair of input sentences, indicating the relationship or similarity between them
