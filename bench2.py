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

    embeddings_by_docs = model.encode(dset.all_sentences, convert_to_numpy=True, show_progress_bar=True)
    fact_embeddings = model.encode(fact, convert_to_numpy=True)

    evaluator = Evaluator(embeddings_by_docs, fact_embeddings)
    rel_true = [i for i, k in enumerate(dset.documents) if k[0] in dset.rel_true_for(fact_id)]
    print(f"INFO - benchmarking {model_name}...")

    dot_sims = evaluator.dot_product()
    dot_bench = Benchmark(rel_true, dot_sims, threshold=0.8)
    print(f"\t+ dot_sim\n\t\t{dot_bench.precision=}\n\t\t{dot_bench.recall=}\n\t\t{dot_bench.f1_score=}")

    cos_sims = evaluator.cosine_similarity()
    cos_bench = Benchmark(rel_true, cos_sims, threshold=0.8)
    print(f"\t+ cos_sim\n\t\t{cos_bench.precision=}\n\t\t{cos_bench.recall=}\n\t\t{cos_bench.f1_score=}")

    euc_sims = evaluator.euclidean_distance()
    euc_bench = Benchmark(rel_true, euc_sims, threshold=0.8)
    print(f"\t+ euc_sim\n\t\t{euc_bench.precision=}\n\t\t{euc_bench.recall=}\n\t\t{euc_bench.f1_score=}")
