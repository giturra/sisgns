import random
import numpy as np
 
class UnigramTable:

    def __init__(self, max_size) -> None:
        self.max_size = max_size
        self.current_size = 0
        self.table = np.zeros(int(self.max_size))
        self.weight_sum = 0
    
    def sample(self, rand):
        rand_num = int(rand.uniform(0, self.current_size))
        output = self.table[rand_num]
        return output

    def update(self, word_index, weight, rand):
        self.weight_sum += weight
        if self.current_size < self.max_size:
            #print(weight)
            new_size = min(rand.round(weight) + self.current_size, self.max_size)
            #print(new_size)
            #print(self.current_size, new_size)
            for i in range(self.current_size, new_size):
                self.table[i] = word_index
            self.current_size = new_size
            #print(self.current_size)
        else:
            n = rand.round(weight / self.weight_sum) * self.max_size
            #print(f"n {n}")
            for i in range(n):
                self.table[i] = word_index