from pv3.dataset import Dataset
from pv3.util import get_dotenv
from pv3.evaluator import Evaluator
from pv3.benchmark import Benchmark
from sentence_transformers import SentenceTransformer


MODEL_LIST = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "nlplabtdtu/sbert-all-MiniLM-L6-v2",
    "Alibaba-NLP/gte-multilingual-base",
]

hf_token = get_dotenv()["HF_TOKEN"]

dset = Dataset.from_filesystem("./corpus")

fact_id = dset.sample(n=1)[0]
fact = dset.facts[fact_id]
print(f"Query: {fact} [{fact_id}]")

for model_name in MODEL_LIST:
    model = SentenceTransformer(model_name, token=hf_token, trust_remote_code=True)

    embeddings_by_docs = model.encode(dset.all_sentences, convert_to_tensor=True, show_progress_bar=True)
    fact_embeddings = model.encode(fact, convert_to_tensor=True)

    evaluator = Evaluator(embeddings_by_docs, fact_embeddings)
    rel_true = dset.rel_true_for(fact_id)

    dot_sims = evaluator.dot_product()
    dot_bench = Benchmark(rel_true, dot_sims, threshold=0.8)

    cos_sims = evaluator.cosine_similarity()
    cos_bench = Benchmark(rel_true, cos_sims, threshold=0.8)

    euc_sims = evaluator.euclidean_distance()
    euc_bench = Benchmark(rel_true, euc_sims, threshold=0.8)
