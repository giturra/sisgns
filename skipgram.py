import torch

class SkipGram(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGram, self).__init__()
        
        # embeddings
        self.embedding_u = torch.nn.Embedding(vocab_size, embedding_dim)
        self.embedding_v = torch.nn.Embedding(vocab_size, embedding_dim)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def forward(self, x):
        #print(x)
        # input should be of shape [batch_size, 1+k, 2]
        # split positive and negative sample
        x_pos_1, x_pos_2 = x[:, 0, :].T
        x_neg_1, x_neg_2 = x[:, 1:, :].T
        # print(x_pos_1, x_pos_2)
        
        # log-likelihood w.r.t. x_pos
        u = self.embedding_u(x_pos_1).to('cuda')
        
        v = self.embedding_v(x_pos_2).to('cuda')
        x_pos = (u * v).sum(dim=1).view(1, -1)
        
        # print(x_pos)
        # log-likelihood w.r.t. x_neg
        u = self.embedding_u(x_neg_1).to('cuda')
        v = self.embedding_v(x_neg_2).to('cuda')
        x_neg = (u * v).sum(dim=2)
        # print(x_neg)
        x = torch.cat((x_pos, x_neg)).T
        # print(x)
        return x