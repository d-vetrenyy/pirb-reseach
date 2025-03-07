from pv3.dataset import Dataset
from pv3.benchmark import Benchmark
from pv3.util import get_dotenv
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, safe_sparse_dot
import sys


MODEL_LIST = [
    "sentence-transformers/all-MiniLM-L6-v2",
    # "Snowflake/snowflake-arctic-embed-l-v2.0",
    # "jinaai/jina-embeddings-v3",
    # "ai-forever/sbert_large_nlu_ru",
    "nlplabtdtu/sbert-all-MiniLM-L6-v2",
    "Alibaba-NLP/gte-multilingual-base",
]

hf_token = get_dotenv()["HF_TOKEN"]

dset = Dataset.from_filesystem("./corpus")

fact_id = dset.sample(n=1)[0]
fact = dset.facts[fact_id]
print(f"Query: {fact} [{fact_id}]")

for model_name in MODEL_LIST:
    try:
        model = SentenceTransformer(model_name, token=hf_token, device='cpu', trust_remote_code=True)

        tensor_embeddings_by_docs = model.encode(
            dset.all_sentences,
            # normalize_embeddings=True,
            show_progress_bar=True,
            output_value='sentence_embedding',
            convert_to_tensor=True)
        tensor_fact_embeddings = model.encode(fact, output_value='sentence_embedding', convert_to_tensor=True)

        print(f"INFO - benchmarking: {model_name.split('/')[1]} + cosine_similarity...")
        try:
            # cos_sims = util.pytorch_cos_sim(tensor_fact_embeddings, tensor_embeddings_by_docs)[0].numpy(force=True)
            cos_sims = cosine_similarity(tensor_fact_embeddings, tensor_embeddings_by_docs)[0]
            benchmark = Benchmark(dset, fact_id, Benchmark.rel_pred(cos_sims, threshold=0.85))
            print(f"{benchmark.precision=}\n{benchmark.recall=}\n{benchmark.f1_score=}\n")
        except Exception as e:
            print(f"ERROR - {e}", file=sys.stderr)

        print(f"INFO - benchmarking: {model_name.split('/')[1]} + euclidean...")
        try:
            # euc_sims = util.euclidean_sim(tensor_fact_embeddings, tensor_embeddings_by_docs)[0].numpy(force=True)
            euc_sims = euclidean_distances(tensor_fact_embeddings, tensor_embeddings_by_docs)[0]
            benchmark = Benchmark(dset, fact_id, Benchmark.rel_pred(euc_sims, threshold=0.8))
            print(f"{benchmark.precision=}\n{benchmark.recall=}\n{benchmark.f1_score=}\n")
        except Exception as e:
            print(f"ERROR - {e}", file=sys.stderr)

        print(f"INFO - benchmarking: {model_name.split('/')[1]} + dot_product...")
        try:
            # dot_sims = util.dot_score(tensor_fact_embeddings, tensor_embeddings_by_docs)[0].numpy(force=True)
            dot_sims = safe_sparse_dot(tensor_fact_embeddings, tensor_embeddings_by_docs)[0]
            benchmark = Benchmark(dset, fact_id, Benchmark.rel_pred(dot_sims, threshold=0.8))
            print(f"{benchmark.precision=}\n{benchmark.recall=}\n{benchmark.f1_score=}\n")
        except Exception as e:
            print(f"ERROR - {e}", file=sys.stderr)

    except Exception as e:
        print(f"ERROR - {e}", file=sys.stderr)
        continue
