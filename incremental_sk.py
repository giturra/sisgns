import torch
import numpy as np

from nltk.probability import RandomProbDist
from river.base.transformer import Transformer
from river.feature_extraction.vectorize import VectorizerMixin

from .vocab import Vocabulary
from .unigram_table import UnigramTable
from .rand import RandomNum
from .skipgram import SkipGram


class IncrementalSG(Transformer, VectorizerMixin):

    def __init__(
        self,
        vec_size=100, 
        max_vocab_size=1e6,
        unigram_table_size=1e8,
        window_size=5,
        neg_sample_num=5,
        alpha=0.75,
        subsampling_threshold=1e-3,
        on=None, 
        tokenizer=None):

        super().__init__(on=on, tokenizer=tokenizer)
        
        self.vec_size = int(vec_size)

        self.max_vocab_size = int(2 * max_vocab_size)
        self.vocab = Vocabulary(int(self.max_vocab_size * 2))

        self.unigram_table_size = unigram_table_size
        self.unigram_table = UnigramTable(self.unigram_table_size)

        self.counts = np.zeros(int(2 * self.max_vocab_size))
        self.total_count = 0

        self.neg_sample_num = neg_sample_num

        self.alpha = alpha
        self.window_size = window_size
        self.subsampling_threshold = subsampling_threshold
        
        self.randomizer = RandomNum(1234)

        self.model = SkipGram(self.max_vocab_size, self.vec_size)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.5, momentum=0.9)
        self.criterion = torch.nn.BCEWithLogitsLoss()


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def learn_one(self, x: dict, **kwargs):
        tokens = self.process_text(x)
        n = len(tokens)
        neg_samples = np.zeros(self.neg_sample_num)
        for target, w in enumerate(tokens):
            self.update_unigram_table(w)
            target_index = self.vocab[w]
            if target_index == -1:
                continue
            random_window_size = self.randomizer.uniform(1, self.window_size + 1)
            #print(random_window_size)
            for offset in range(int(-random_window_size), int(random_window_size)):
                #print("hola")
                if offset == 0 or (target + offset) < 0:
                    #print("1")
                    continue
                if (target + offset) == n:
                    #print("2")
                    break
                context_index = self.vocab[tokens[target + offset]]
                if context_index == -1:
                    #print("3")
                    continue
                if 0 < self.counts[context_index] and np.sqrt(
                    (self.subsampling_threshold * self.total_count) / self.counts[context_index]
                ) < self.randomizer.uniform(0, 1):
                    #print("4")
                    continue
                for k in range(0, self.neg_sample_num):
                    neg_samples[k] = int(self.unigram_table.sample(self.randomizer))
                
                input_nn, labels = create_input(target_index, context_index, neg_samples)
                input_nn.to(self.device)
                labels.to(self.device)

                print(self.model.parameters())

                pred = self.model(input_nn)
                self.model.zero_grad()

                self.criterion(pred, labels)

                self.optimizer.step()

                print(self.model.parameters())

        return self

    def transform_one(self, x: dict) -> dict:
        return super().transform_one(x)

    def update_unigram_table(self, word: str):
        word_index = self.vocab.add(word)
        self.total_count += 1
        if word_index != -1:
            self.counts[word_index] += 1
            F = np.power(self.counts[word_index], self.alpha) - np.power(self.counts[word_index] - 1, self.alpha)
            self.unigram_table.update(word_index, F, self.randomizer)


def create_input(target_index, context_index, neg_samples):
    input = [[int(target_index), int(context_index)]]
    labels = [1]
    for neg_sample in neg_samples:
        input.append([target_index, int(neg_sample)])
        labels.append(0)
    return torch.LongTensor([input]), torch.LongTensor([labels])