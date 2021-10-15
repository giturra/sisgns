from numpy.random import Generator, MT19937, SeedSequence

class RandomNum:

    def __init__(self, seed):
        self.sg = SeedSequence(seed)
        self.bit_generator = MT19937(self.sg)
        self.rg = []
        for _ in range(10):
            self.rg.append(Generator(self.bit_generator))
            self.bit_generator = self.bit_generator.jumped()
        self.state = self.bit_generator.state


r = RandomNum(1234)
print(r.state)
