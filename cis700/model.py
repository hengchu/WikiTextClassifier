import torch
import math
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

class WordEmbedding(nn.Module):
  def __init__(self, vocab_size, dim_embedding):
    super(WordEmbedding, self).__init__()
    self.embed = nn.Embedding(vocab_size, dim_embedding)
  def forward(self, x):
    return self.embed(x)

class PositionEncoding(nn.Module):
  def __init__(self, dim_embedding, max_seq_len):
    super(PositionEncoding, self).__init__()
    self.dim_embedding = dim_embedding

    pe = torch.zeros(max_seq_len, dim_embedding)
    for pos in range(max_seq_len):
      for i in range(0, dim_embedding, 2):
        pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / dim_embedding)))
        pe[pos, i+1] = math.cos(pos / (10000 ** ((2 * (i+1)) / dim_embedding)))

    pe = pe.unsqueeze(0)
    self.register_buffer('pe', pe)

  def forward(self, x):
    x = x * math.sqrt(self.dim_embedding)
    x_len = x.size(1)
    if False: #torch.cuda.is_available():
      x = x + Variable(self.pe[:,:x_len], requires_grad=False).cuda()
    else:
      x = x + Variable(self.pe[:,:x_len], requires_grad=False)
    return x

class MultiheadAttention(nn.Module):
  def __init__(self, num_heads, dim_embedding, dropout = 0.1):
    super(MultiheadAttention, self).__init__()

    self.dim_embedding = dim_embedding
    self.dim_k = dim_embedding / num_heads
    if int(self.dim_k) != self.dim_k:
      raise ValueError('num_heads should divide dim_embedding evenly! num_heads = %d, dim_embedding = %d' \
                       % (num_heads, dim_embedding))
    self.dim_k = int(self.dim_k)
    self.num_heads = num_heads

    self.q_linear = nn.Linear(dim_embedding, dim_embedding)
    self.v_linear = nn.Linear(dim_embedding, dim_embedding)
    self.k_linear = nn.Linear(dim_embedding, dim_embedding)

    self.dropout = nn.Dropout(dropout)

    self.out = nn.Linear(dim_embedding, dim_embedding)

  def attention(self, q, v, k, mask):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dim_k)
    # print('q.size = %s' % str(q.size()))
    # print('scores.size = %s' % str(scores.size()))
    mask = mask.unsqueeze(1).unsqueeze(1)
    scores = scores.masked_fill(mask == 0, -1e9)

    # print(scores.size())
    scores = F.softmax(scores, dim=-1)
    # print(scores.size())
    scores = self.dropout(scores)

    return torch.matmul(scores, v)

  def forward(self, q, v, k, mask):
    batch_size = q.size(0)

    q = self.q_linear(q).view(batch_size, -1, self.num_heads, self.dim_k)
    v = self.v_linear(v).view(batch_size, -1, self.num_heads, self.dim_k)
    k = self.k_linear(k).view(batch_size, -1, self.num_heads, self.dim_k)

    q = q.transpose(1, 2)
    v = v.transpose(1, 2)
    k = k.transpose(1, 2)

    scores = self.attention(q, v, k, mask)
    scores = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.dim_embedding)
    return scores

class FeedForward(nn.Module):
  def __init__(self, dim_embedding, num_features, dropout = 0.1):
    super(FeedForward, self).__init__()
    self.fc1 = nn.Linear(dim_embedding, num_features)
    self.dropout = nn.Dropout(dropout)
    self.fc2 = nn.Linear(num_features, dim_embedding)

  def forward(self, x):
    x = self.fc1(x)
    x = F.relu(x)
    x = self.dropout(x)
    return self.fc2(x)

class Normalization(nn.Module):
  def __init__(self, dim_embedding, eps = 1e-6):
    super(Normalization, self).__init__()

    self.dim_embedding = dim_embedding
    self.alpha = nn.Parameter(torch.ones(self.dim_embedding))
    self.bias = nn.Parameter(torch.zeros(self.dim_embedding))
    self.eps = eps

  def forward(self, x):
    norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
    return norm

class EncoderLayer(nn.Module):
  def __init__(self, num_heads, dim_embedding, ff_num_features, dropout=0.1):
    super(EncoderLayer, self).__init__()
    self.attention = MultiheadAttention(num_heads, dim_embedding)
    self.norm1 = Normalization(dim_embedding)
    self.ff = FeedForward(dim_embedding, ff_num_features)
    self.norm2 = Normalization(dim_embedding)
    self.drop1 = nn.Dropout(dropout)
    self.drop2 = nn.Dropout(dropout)

  def forward(self, x, mask):
    x_ = self.norm1(x)
    x = x + self.drop1(self.attention(x_, x_, x_, mask))
    x_ = self.norm2(x)
    x = x + self.drop2(self.ff(x_))
    return x

class Encoder(nn.Module):
  def __init__(self, vocab_size, dim_embedding, num_heads, ff_num_features, num_encoder_layers, max_seq_len):
    super(Encoder, self).__init__()
    self.num_encoder_layers = num_encoder_layers
    self.embed = WordEmbedding(vocab_size, dim_embedding)
    self.pe = PositionEncoding(dim_embedding, max_seq_len)
    self.encoder_layers = \
      nn.ModuleList([EncoderLayer(num_heads, dim_embedding, ff_num_features) for _ in range(num_encoder_layers)])
    self.norm = Normalization(dim_embedding)

  def forward(self, x, mask):
    x = self.embed(x)
    x = self.pe(x)
    for i in range(self.num_encoder_layers):
      x = self.encoder_layers[i](x, mask)
    return self.norm(x)

class Classifier(nn.Module):
  def __init__(self,
               vocab_size, dim_embedding, num_heads,
               ff_num_features, num_encoder_layers,
               max_seq_len, num_classes):
    super(Classifier, self).__init__()
    self.dim_embedding = dim_embedding
    self.max_seq_len = max_seq_len
    self.encoder = Encoder(vocab_size, dim_embedding, num_heads, ff_num_features, num_encoder_layers, max_seq_len)
    self.fc = nn.Linear(dim_embedding * max_seq_len, num_classes)

  def forward(self, x, mask):
    x = self.encoder(x, mask)
    return self.fc(x.view(-1, self.dim_embedding * self.max_seq_len))
