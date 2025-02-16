import pirb2


MODEL_LIST = ['BAAI/bge-m3']

dataset = pirb2.Dataset.from_filesystem('./corpus')

query_id = dataset.sample()[0]
query_text = dataset.facts[query_id]
expected = dataset.expected_for(query_id)


for model_name in MODEL_LIST:
    print("INFO - testing model:", model_name)
    retriever = pirb2.RetrieverModel(model_name, dataset.documents)
    query_vector = retriever.search_query(query_text)
    print(query_vector)
