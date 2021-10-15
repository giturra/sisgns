import random
import math
from numpy.random import Generator, MT19937, SeedSequence


class RandomNum:

    def __init__(self, seed):
        self.sg = SeedSequence(seed)
        self.bit_generator = MT19937(self.sg)
        self.rg = []
        for _ in range(10):
            self.rg.append(Generator(self.bit_generator))
            self.bit_generator = self.bit_generator.jumped()
        self.state = self.bit_generator.state['state']['key']

    def uniform(self, min_num, max_num):
        random_index = random.randint(0, len(self.state) - 1)
        result = min_num + (max_num - min_num) * self.state[random_index] / (self.state.max() - self.state.min())
        return result
    
    def round(self, x):
        c = math.ceil(x)
        f = math.floor(x)
        if self.uniform(0, 1) < (x - f):
            return c
        else: 
            return f
