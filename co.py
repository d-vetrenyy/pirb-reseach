from pirb.corpus import CorpusLoader


corpus = CorpusLoader("./corpus")

print(corpus.get_facts(corpus.sample(2)))
