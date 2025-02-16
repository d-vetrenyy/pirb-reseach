from pirb2.corpus import CorpusLoader
from transformers import pipeline


corpus = CorpusLoader.from_filesystem('./corpus')

pipel = pipeline(model='BAAI/bge-m3')
