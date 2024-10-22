# -*- coding: utf-8 -*-
#this file in charge of training model
import os
import argparse
import time
import torch
import random
import torch.nn as nn
import numpy as np
from utils.seq2text import initNetParams, train_step, val_step
from utils.dataset import EurDataset, collate_data
from models.DeepSC import DeepSC
from torch.utils.data import DataLoader
from tqdm import tqdm
from communication.SNR2noise import SNR_to_noise
import logging
from utils.config import load_vocab



# Set up logging
logging.basicConfig(filename='deepsc_transmission.log', level=logging.INFO, format='%(asctime)s - %(message)s')


parser = argparse.ArgumentParser()
parser.add_argument('--vocab-file', default='europarl/vocab.json', type=str)
parser.add_argument('--checkpoint-path', default='checkpoints/AWGN (1e-4)', type=str)
parser.add_argument('--MAX-LENGTH', default=30, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)
parser.add_argument('--d-model', default=128, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=4, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--epochs', default=80, type=int)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

noise_std1 = (SNR_to_noise(10),)  # Direct path system


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def validate(epoch, args, net):
    test_eur = EurDataset('test')
    test_iterator = DataLoader(test_eur, batch_size=args.batch_size, num_workers=0,
                               pin_memory=True, collate_fn=collate_data)
    net.eval()
    pbar = tqdm(test_iterator)
    total = 0
    with torch.no_grad():
        for sents in pbar:
            sents = sents.to(device)

            loss = val_step(net, sents, sents, noise_std1[0], pad_idx, criterion)  # Direct path system
            total += loss
            pbar.set_description(
                'Epoch: {}; Type: VALIDATE; Loss: {:.5f}'.format(
                    epoch + 1, loss
                )
            )

    return total / len(test_iterator)


def train(epoch, args, net):
    train_eur = EurDataset('train')
    train_iterator = DataLoader(train_eur, batch_size=args.batch_size, num_workers=0,
                                pin_memory=True, collate_fn=collate_data)
    pbar = tqdm(train_iterator)

    for sents in pbar:
        sents = sents.to(device)
        loss = train_step(net, sents, sents, noise_std1[0], pad_idx, optimizer, criterion)  # Direct path system

        pbar.set_description(
            'Epoch: {};  Type: TRAIN; Loss: {:.5f}'.format(
                epoch + 1, loss
            )
        )


if __name__ == '__main__':
    args = parser.parse_args()
    args.vocab_file = 'data/' + args.vocab_file

    # -------------------- Preparing the dataset --------------------
    #vocab = json.load(open(args.vocab_file, 'rb'))
    #token_to_idx = vocab['token_to_idx']
    #num_vocab = len(token_to_idx)
    #pad_idx = token_to_idx["<PAD>"]
    #start_idx = token_to_idx["<START>"]
    #end_idx = token_to_idx["<END>"]

    token_to_idx, pad_idx, start_idx, end_idx = load_vocab(args.vocab_file)
    num_vocab = len(token_to_idx)

    # -------------------- Define optimizer and loss function --------------------
    deepsc = DeepSC(args.num_layers, num_vocab, num_vocab,
                    num_vocab, num_vocab, args.d_model, args.num_heads,
                    args.dff, 0.1).to(device)
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(deepsc.parameters(),
                                 lr=1e-4, betas=(0.9, 0.98), eps=1e-8, weight_decay=5e-4)
    initNetParams(deepsc)
    for epoch in range(args.epochs):
        start = time.time()
        record_acc = 10

        train(epoch, args, deepsc)
        avg_acc = validate(epoch, args, deepsc)

        if avg_acc < record_acc:
            if not os.path.exists(args.checkpoint_path):
                os.makedirs(args.checkpoint_path)
            with open(args.checkpoint_path + '/checkpoint_{}.pth'.format(str(epoch + 1).zfill(2)), 'wb') as f:
                torch.save(deepsc.state_dict(), f)
            record_acc = avg_acc
    record_loss = []