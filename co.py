from pirb.corpus import CorpusLoader


corpus = CorpusLoader("./corpus")

samples = corpus.sample(2)
facts = corpus.get_facts(samples)
print(facts)
