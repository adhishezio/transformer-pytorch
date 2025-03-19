import torch 
import torch.nn
import math

# Define the model
class InputEmbeddings(nn.Model):

    def __init__(self, d_model: int, vocab_size: int)--> None:
        super().__init__()
        self.d_model = d_model # embedding dimension
        self.vocab_size = vocab_size  # vocabulary size
        self.embeddings = nn.Embedding(vocab_size, d_model) # embedding layer

    def forward(self, x):
        return self.embeddings(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float, seq_len: int) --> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        sef.seq_len = seq_len
        sef.d_model = d_model

        # Create a positional encoding
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # a tensor with values from 0 to seq_len - 1
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)) 

        # apply sin to even indices in the array; 2i
        pe[:, 0::2] = torch.sin(position * div_term)
        # apply cos to odd indices in the array; 2i+1
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) #(1, seq_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].requires_grad_(False)
        return self.dropout(x)

# define the layer normalization
class LayerNorm(nn.Module):

    def __init__(self, eps: float = 1e-6) --> None:
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(d_model)) # alpha multiplicative parameter
        self.b_2 = nn.Parameter(torch.zeros(d_model)) # beta additive parameter
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

# define the feedforward network
class FeedForward(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) --> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        return self.linear_2(self.dropout(F.relu(self.linear_1(x))))