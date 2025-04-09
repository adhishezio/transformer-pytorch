import torch 
import torch.nn
import math

# Define the model
class InputEmbeddings(nn.Model):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model # embedding dimension
        self.vocab_size = vocab_size  # vocabulary size
        self.embeddings = nn.Embedding(vocab_size, d_model) # embedding layer

    def forward(self, x):
        return self.embeddings(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float, seq_len: int) -> None:
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
class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model # embedding dimension
        self.h = h
        assert d_model % h == 0, "d_model must be divisible by h"
        self.d_k = d_model // h # dimension of each head
        self.dropout = nn.Dropout(dropout)

        self.w_q = nn.Linear(d_model, d_model) # wq
        self.w_k = nn.Linear(d_model, d_model) #wk
        self.w_v = nn.Linear(d_model, d_model) #wv
        self.w_o = nn.Linear(d_model, d_model) #wo

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Droupout):
        d_k = query.size(-1) 
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k) 
        if mask is not None:
            attention_scores = attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # (batch_size, h, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        return (attention_scores @ value), attention_scores # (batch_size, h, seq_len, d_k), (batch_size, h, seq_len, seq_len)

    def forward(self, q, k, v, mask=None):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        query = self.w_q(q) 
        key = self.w_k(k)
        value = self.w_v(v)

        # split into h heads
        # (batch_size, seq_len, d_model) -> (batch, seq_len, h,d_k) -> (batch_size, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2) 
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # (batch_size, h, seq_len, d_k) -> (batch_size, seq_len, h, d_k) -> (batch_size, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        return self.w_o(x) # (batch_size, seq_len, d_model)
    
class ResidualConnection(nn.Module):

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
class EncodeerBlock(nn.Module):
    def __init__(self, attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.attention_block = attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) 
                                                   for _ in range(2)]) # two residual connections

    def forward(self, x, src_mask=None):
        x = self.residual_connections[0](x, lambda x: self.attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm() 

    def forward(self, x, src_mask=None):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)
    
    


        