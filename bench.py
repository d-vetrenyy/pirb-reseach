from pv3.dataset import Dataset
from pv3.benchmark import Benchmark
from pv3.util import get_dotenv
from sentence_transformers import SentenceTransformer, util


MODEL_LIST = ["sentence-transformers/all-MiniLM-L6-v2"]

hf_token = get_dotenv()["HF_TOKEN"]

dset = Dataset.from_filesystem("./corpus")

fact_id = dset.sample(n=1)[0]
fact = dset.facts[fact_id]
print(f"Query: {fact} [{fact_id}]")

for model_name in MODEL_LIST:
    model = SentenceTransformer(model_name, token=hf_token)

    tensor_embeddings_by_docs = model.encode(
        dset.all_sentences,
        show_progress_bar=True,
        output_value='sentence_embedding',
        convert_to_tensor=True)
    tensor_fact_embeddings = model.encode(fact, output_value='sentence_embedding', convert_to_tensor=True)

    cos_sims = util.pytorch_cos_sim(tensor_fact_embeddings, tensor_embeddings_by_docs)[0].numpy(force=True)
    benchmark = Benchmark(fact_id, cos_sims)
    print(f"INFO - benchmarking: {model_name.split('/')[1]} + cosine_similarity...")

