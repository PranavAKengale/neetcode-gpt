import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        attached=positive+negative


        voc=sorted({word for sen in attached for word in sen.split()})
        word_to_id= {word:idx+1 for idx,word in enumerate(voc)}

        encoded=[torch.tensor([word_to_id[w] for w in s.split()]) for s in attached]

        return nn.utils.rnn.pad_sequence(encoded,batch_first=True)