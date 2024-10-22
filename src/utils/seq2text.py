#this is the process transfer result sequence into text for 
#1. semantic communication
#2. point 2 point

import torch
import torch.nn as nn
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SeqtoText:
    def __init__(self, vocb_dictionary, end_idx):
        self.reverse_word_map = dict(zip(vocb_dictionary.values(), vocb_dictionary.keys()))
        self.end_idx = end_idx

    def sequence_to_text(self, list_of_indices):
        """ Looking up words in dictionary """
        words = []
        for idx in list_of_indices:
            if idx == self.end_idx:
                break
            else:
                words.append(self.reverse_word_map.get(idx))
        words = ' '.join(words)
        return words


def initNetParams(model):
    """ Init net parameters """
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


def subsequent_mask(size):
    """ Mask out subsequent positions """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask)


def create_masks(src, trg, padding_idx):
    src_mask = (src == padding_idx).unsqueeze(-2).type(torch.FloatTensor)  # [batch, 1, seq_len]

    trg_mask = (trg == padding_idx).unsqueeze(-2).type(torch.FloatTensor)  # [batch, 1, seq_len]
    look_ahead_mask = subsequent_mask(trg.size(-1)).type_as(trg_mask.data)
    combined_mask = torch.max(trg_mask, look_ahead_mask)
    return src_mask.to(device), combined_mask.to(device)


def loss_function(x, trg, padding_idx, criterion):
    loss = criterion(x, trg)
    mask = (trg != padding_idx).type_as(loss.data)
    loss *= mask
    return loss.mean()


# # Power normalization at the transmitter
# def PowerNormalize(x):
#     x_square = torch.mul(x, x)
#     power = math.sqrt(2) * torch.mean(x_square).sqrt()
#     x = torch.div(x, power)
#     return x

def PowerNormalize(x):
    x_square = torch.mul(x, x)
    power = torch.mean(x_square).sqrt()
    if power > 1:
        x = torch.div(x, power)
    return x


# --------------------------------------------------------------------------------------------------------------------
# ------------------------------------------- Point-to-point System [START] ------------------------------------------

# -------------------- Point-to-point System --------------------
def total_rx_sig(Tx_sig, n_var):

    X = Tx_sig
    bs, sent_len, d_model = X.shape
    X = torch.reshape(X, (bs, -1, 2))
    X_real = X[:, :, 0]
    X_imag = X[:, :, 1]
    X_complex = torch.complex(X_real, X_imag)

    N = torch.normal(0, n_var, size=X.shape).to(device)
    N_real = N[:, :, 0]
    N_imag = N[:, :, 1]
    N_complex = torch.complex(N_real, N_imag)

    # -------------------- Perfect CSI --------------------
    Y_complex = X_complex + N_complex
    X_est_complex = Y_complex

    X_est_real = torch.real(X_est_complex)
    X_est_img = torch.imag(X_est_complex)
    X_est_real = torch.unsqueeze(X_est_real, -1)
    X_est_img = torch.unsqueeze(X_est_img, -1)
    X_est = torch.cat([X_est_real, X_est_img], axis=-1)
    X_est = torch.reshape(X_est, (bs, sent_len, -1))

    return X_est


# -------------------- Point-to-point System --------------------
def train_step(model, src, trg, n_var, pad, opt, criterion):
    model.train()

    trg_inp = trg[:, :-1]
    trg_real = trg[:, 1:]

    opt.zero_grad()
    #torch.save(src, './Intermediates/Inputtensor.pth')  # Save input tensors after embedding

    src_mask, look_ahead_mask = create_masks(src, trg_inp, pad)

    enc_output = model.encoder(src, src_mask)
    channel_enc_output = model.channel_encoder(enc_output)
    Tx_sig = PowerNormalize(channel_enc_output)
    #torch.save(Tx_sig, './Intermediates/Outputtensor.pth')  # Save output tensors before entering wireless channels (power normalized already)

    Rx_sig = total_rx_sig(Tx_sig, n_var)
    channel_dec_output = model.channel_decoder(Rx_sig)
    dec_output = model.decoder(trg_inp, channel_dec_output, look_ahead_mask, src_mask)
    pred = model.dense(dec_output)
    ntokens = pred.size(-1)
    loss = loss_function(pred.contiguous().view(-1, ntokens), trg_real.contiguous().view(-1), pad, criterion)
    loss.backward()
    opt.step()
    return loss


# -------------------- Point-to-point System --------------------
def val_step(model, src, trg, n_var, pad, criterion):
    trg_inp = trg[:, :-1]
    trg_real = trg[:, 1:]

    src_mask, look_ahead_mask = create_masks(src, trg_inp, pad)

    enc_output = model.encoder(src, src_mask)
    channel_enc_output = model.channel_encoder(enc_output)
    Tx_sig = PowerNormalize(channel_enc_output)

    Rx_sig = total_rx_sig(Tx_sig, n_var)

    channel_dec_output = model.channel_decoder(Rx_sig)
    dec_output = model.decoder(trg_inp, channel_dec_output, look_ahead_mask, src_mask)
    pred = model.dense(dec_output)
    ntokens = pred.size(-1)

    loss = loss_function(pred.contiguous().view(-1, ntokens), trg_real.contiguous().view(-1), pad, criterion)
    return loss.item()


# -------------------- Point-to-point System --------------------
def greedy_decode(model, src, n_var, max_len, padding_idx, start_symbol):
    src_mask = (src == padding_idx).unsqueeze(-2).type(torch.FloatTensor).to(device)  # Create src_mask

    enc_output = model.encoder(src, src_mask)
    channel_enc_output = model.channel_encoder(enc_output)
    Tx_sig = PowerNormalize(channel_enc_output)
    Rx_sig = total_rx_sig(Tx_sig, n_var)
    memory = model.channel_decoder(Rx_sig)
    outputs = torch.ones(src.size(0), 1).fill_(start_symbol).type_as(src.data)

    for i in range(max_len - 1):
        trg_mask = (outputs == padding_idx).unsqueeze(-2).type(torch.FloatTensor)
        look_ahead_mask = subsequent_mask(outputs.size(1)).type(torch.FloatTensor)
        combined_mask = torch.max(trg_mask, look_ahead_mask)
        combined_mask = combined_mask.to(device)

        dec_output = model.decoder(outputs, memory, combined_mask, None)
        pred = model.dense(dec_output)
        prob = pred[:, -1:, :]
        _, next_word = torch.max(prob, dim=-1)
        outputs = torch.cat([outputs, next_word], dim=1)

    return outputs
# ------------------------------------------- Point-to-point System [END] --------------------------------------------
# --------------------------------------------------------------------------------------------------------------------
