import torch.nn as nn
from utils import Initializer

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.initializer = Initializer() #初始化，这个utils的实现？

        self.drop = nn.Dropout(dropout)  #两个参数，dropout 随机失活概率；inplace:默认是False

        self.encoder = nn.Embedding(ntoken, ninp) #ninp是每一个embedding向量的维数  ntoken是size of the dictionary of embeddings
        self.initializer.init_embedding_(self.encoder)  #初始化embedding

        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout, batch_first=True) #getattr就是获得某属性的值，这里就是获取nn的网络类型值，如获得nn的LSTM
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]  #获取字典的这个rnn_type值
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout, batch_first=True) #激活函数的选择

        self.decoder = nn.Linear(nhid, ntoken)
        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')  #若是softmax和word embedding 权重绑定。则hidden层和emsize相等
            self.decoder.weight = self.encoder.weight  #若绑定就是embedding的encoding.weight 和 decoder.weight相等
        else:
            self.initializer.init_linear_(self.decoder, nhid, dropout=dropout)  #对线性层进行初始化

        # self.init_weights()


    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)


    def forward(self, input, words_lengths):
        ''' Forward pass:
            - word embedding
            - 输入循环神经网络
            - 一个线性层从hidden state转化为输出单词表
        '''
        emb = self.drop(self.encoder(input)) # batch_size * seq_len * emb_dim   #将输入向量嵌入后随机置0
        emb = nn.utils.rnn.pack_padded_sequence(emb, words_lengths, batch_first=True)

        output_packed, _ = self.rnn(emb)  #通过网络得到输出
        output, _ = nn.utils.rnn.pad_packed_sequence(output_packed, batch_first=True) # output: batch_size * seq_len * num_direct*hidden_dim,  num_direct = 1

        output = self.drop(output)   #对输出进行随机置0
        decoded = self.decoder(output.contiguous().view(output.size(0)*output.size(1), output.size(2))) #decoder，output.contiguous()的一种可能解释是因为view要基于整块内存，而Tensor可能是零碎的，所以用contiguous变成连续
        return decoded.view(output.size(0), output.size(1), decoded.size(1)) # batch_size * seq_len * vocab_size