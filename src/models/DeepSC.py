# -*- coding: utf-8 -*-

"""
Transformer includes:
    Encoder
        1. Positional coding
        2. Multihead-attention
        3. PositionwiseFeedForward
    Decoder
        1. Positional coding
        2. Multihead-attention
        3. Multihead-attention
        4. PositionwiseFeedForward
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
from communication.channel_transformation import ChannelDecoder,ChannelEncoder


class PositionalEncoding(nn.Module):
    """Implement the PE function"""

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))  # math.log(math.exp(1)) = 1
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        x = self.dropout(x)
        return x


class MultiHeadedAttention(nn.Module):

    def __init__(self, num_heads, d_model, dropout=0.1):
        """Take in model size and number of heads"""
        super(MultiHeadedAttention, self).__init__()
        assert d_model % num_heads == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.dense = nn.Linear(d_model, d_model)

        # self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query = self.wq(query).view(nbatches, -1, self.num_heads, self.d_k)
        query = query.transpose(1, 2)

        key = self.wk(key).view(nbatches, -1, self.num_heads, self.d_k)
        key = key.transpose(1, 2)

        value = self.wv(value).view(nbatches, -1, self.num_heads, self.d_k)
        value = value.transpose(1, 2)

        #        query, key, value = \
        #            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        #             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = self.attention(query, key, value, mask=mask)

        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.num_heads * self.d_k)

        x = self.dense(x)
        x = self.dropout(x)

        return x

    def attention(self, query, key, value, mask=None):
        """Compute 'Scaled Dot Product Attention' """
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores += (mask * -1e9)  # Mask with -1e9
        p_attn = F.softmax(scores, dim=-1)
        # print('p_attn', p_attn.size())  # Attention weights
        # print('value', value.size())
        return torch.matmul(p_attn, value), p_attn


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = F.relu(x)
        x = self.w_2(x)
        x = self.dropout(x)
        return x


class EncoderLayer(nn.Module):
    """Encoder is made up of self-attn and feed forward (defined below)"""

    def __init__(self, d_model, num_heads, dff, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadedAttention(num_heads, d_model, dropout=0.1)
        self.ffn = PositionwiseFeedForward(d_model, dff, dropout=0.1)

        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, mask):
        """Follow Figure 2 (left) for connections., that is transimitter"""
        attn_output = self.mha(x, x, x, mask)
        x = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(x)
        x = self.layernorm2(x + ffn_output)

        return x


class DecoderLayer(nn.Module):
    """Decoder is made of self-attn, src-attn, and feed forward (defined below)"""

    def __init__(self, d_model, num_heads, dff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_mha = MultiHeadedAttention(num_heads, d_model, dropout=0.1)
        self.src_mha = MultiHeadedAttention(num_heads, d_model, dropout=0.1)
        self.ffn = PositionwiseFeedForward(d_model, dff, dropout=0.1)

        self.layernorm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm3 = nn.LayerNorm(d_model, eps=1e-6)

        # self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, look_ahead_mask, trg_padding_mask):
        """Follow Figure 2 (right) for connections., that is receiver"""
        # m = memory

        attn_output = self.self_mha(x, x, x, look_ahead_mask)
        x = self.layernorm1(x + attn_output)

        src_output = self.src_mha(x, memory, memory, trg_padding_mask)  # q, k, v
        x = self.layernorm2(x + src_output)

        fnn_output = self.ffn(x)
        x = self.layernorm3(x + fnn_output)
        return x

#############################################
##### steps with additional logging

# Encoder with logging of the input and output shapes
class Encoder(nn.Module):
    """Core encoder is a stack of N layers"""

    def __init__(self, num_layers, src_vocab_size, max_len,
                 d_model, num_heads, dff, dropout=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(src_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)
        self.enc_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, dff, dropout)
                                         for _ in range(num_layers)])

    def forward(self, x, src_mask):
        """Pass the input (and mask) through each layer in turn"""
        # The input size of x is [batch_size, seq_len]
        #shape
        logging.info(f"Input to Encoder shape: {x.shape}")
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        for enc_layer in self.enc_layers:
            x = enc_layer(x, src_mask)
            # print('Encoder output', x.size())
            logging.info(f"Encoder layer output shape: {x.shape}")
        logging.info(f"Final Encoder output shape: {x.shape}")
        torch.save(x, 'checkpoints/encoder_output.pt')
        return x


# Decoder with logging of input/output shapes
class Decoder(nn.Module):
    def __init__(self, num_layers, trg_vocab_size, max_len,
                 d_model, num_heads, dff, dropout=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(trg_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)
        self.dec_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, dff, dropout)
                                         for _ in range(num_layers)])

    def forward(self, x, memory, look_ahead_mask, trg_padding_mask):
        logging.info(f"Input to Decoder shape: {x.shape}")
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        for dec_layer in self.dec_layers:
            x = dec_layer(x, memory, look_ahead_mask, trg_padding_mask)
            logging.info(f"Decoder layer output shape: {x.shape}")
        
        logging.info(f"Final Decoder output shape: {x.shape}")
        torch.save(x, 'checkpoints/decoder_output.pt')
        return x
    


#############################################################

class DeepSC(nn.Module):
    def __init__(self, num_layers, src_vocab_size, trg_vocab_size, src_max_len,
                 trg_max_len, d_model, num_heads, dff, dropout=0.1):
        super(DeepSC, self).__init__()

        """
        1. Transmitter (Semantic Encoder ---> Channel Encoder)
        2. Receiver (Channel Decoder ---> Semantic Decoder)
        Note: The overall system only have one Tx and one big Rx, and
              the Tx_sig comes from the transmitter must be same. This 
              big decoder should decode all things whose receives. That
              is because this is wireless communication system.
        """

        self.encoder = Encoder(num_layers, src_vocab_size, src_max_len,
                               d_model, num_heads, dff, dropout)

        # Set inplace=True in nn.Relu, meaning replace the previous values
        # from the last layer in order to save computing spaces.
        self.channel_encoder = ChannelEncoder(input_dim=d_model, hidden_dim=256, output_dim=16)
        self.channel_decoder = ChannelDecoder(16, d_model, 512)
        self.decoder = Decoder(num_layers, trg_vocab_size, trg_max_len,
                               d_model, num_heads, dff, dropout)

        self.dense = nn.Linear(d_model, trg_vocab_size)
