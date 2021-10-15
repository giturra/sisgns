from river import datasets
from river.datasets import SMSSpam
from nltk import word_tokenize

from incremental_sk import IncrementalSG

from vocab import Vocabulary

# dataset = [
#     "he is a aking",
#     "she is a queen",
#     "he is a man",
#     "she is a woman",
#     "warsaw is poland capital",
#     "berlin is germany capital",
#     "paris is rance capital"
# ]

dataset = SMSSpam()

#isg = IncrementalSG(max_vocab_size=10, unigram_table_size=20, window_size=2, neg_sample_num=2, subsampling_threshold=1, tokenizer=word_tokenize)
isg = IncrementalSG(on='body', tokenizer=word_tokenize)
for xi, yi in dataset:
    isg.learn_one(xi)

# print(isg.vocab.inverse_table)
# print(isg.unigram_table.table)
#print(len(isg.counts))
#print(isg.vocab.inverse_table)
#print(isg.unigram_table.table)
#print(isg.unigram_table.current_size)

# for val in isg.counts:
#     print(val)

# from numpy.random import Generator, MT19937, SeedSequence
# sg = SeedSequence(1234)
# bit_generator = MT19937(sg)
# rg = []
# for _ in range(10):
#    rg.append(Generator(bit_generator))
#    # Chain the BitGenerators
#    bit_generator = bit_generator.jumped()
#    print(bit_generator.state)