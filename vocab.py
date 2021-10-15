from collections import defaultdict


class Vocabulary:

    def __init__(self, max_size):
        self.max_size = max_size
        self.current_size = 0
        self.table = dict()
        self.inverse_table = dict()

    def add(self, word) -> int:
        if word not in self.inverse_table and not self.is_full():
            word_index = self.current_size
            self.inverse_table[word] = word_index
            self.table[word_index] = word
            self.current_size += 1
            return word_index

        elif word in self.inverse_table:
            word_index = self.inverse_table[word]
            return word_index
        else:
            return -1
                
    def is_full(self) -> bool:
        return self.current_size == self.max_size
    
    def __getitem__(self, word: str):
        if word in self.inverse_table:
            word_index = self.inverse_table[word] 
            return word_index
        return -1