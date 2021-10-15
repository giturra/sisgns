from unigram_table import UnigramTable
from rand import RandomNum


rn = RandomNum(1234)

ut = UnigramTable(1000)
ut.current_size = 100
ut.sample(rn)