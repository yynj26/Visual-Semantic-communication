# !usr/bin/env python
# -*- coding:utf-8 _*-
#modified to have 2gram
import os
import torch
import numpy as np
from utils.dataset import EurDataset, collate_data
from torch.utils.data import DataLoader
from eval.BleuScore import BleuScore
from utils.seq2text import greedy_decode, SeqtoText
from tqdm import tqdm
from utils.config import load_vocab
from communication.SNR2noise import SNR_to_noise

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
token_to_idx, pad_idx, start_idx, end_idx = load_vocab('data/europarl/vocab.json')

def performance(args, SNR, net):
    bleu_score_1gram = BleuScore(1, 0, 0, 0)  # 1-gram
    #bleu_score_2gram = BleuScore(0, 1, 0, 0)  # 2-grams
    #bleu_score_3gram = BleuScore(0, 0, 1, 0)  # 3-grams
    #bleu_score_4gram = BleuScore(0, 0, 0, 1)  # 4-grams

    test_eur = EurDataset('test')
    test_iterator = DataLoader(test_eur, batch_size=args.batch_size, num_workers=0,
                               pin_memory=True, collate_fn=collate_data)

    StoT = SeqtoText(token_to_idx, end_idx)

    score_1gram = []
    #score_2gram = []
    net.eval()
    with torch.no_grad():
        for epoch in range(args.epochs):
            Tx_word = []
            Rx_word = []

            for snr in tqdm(SNR):
                word = []
                target_word = []
                noise_std = SNR_to_noise(snr)

                for sents in test_iterator:
                    sents = sents.to(device)
                    target = sents

                    out = greedy_decode(net, sents, noise_std, args.MAX_LENGTH, pad_idx, start_idx)  # Direct path system

                    sentences = out.cpu().numpy().tolist()
                    result_string = list(map(StoT.sequence_to_text, sentences))
                    word = word + result_string

                    target_sent = target.cpu().numpy().tolist()
                    result_string = list(map(StoT.sequence_to_text, target_sent))
                    target_word = target_word + result_string

                Tx_word.append(word)
                Rx_word.append(target_word)

            os.makedirs('/log', exist_ok=True)
            with open('/log/transmitted_text.txt', 'w') as f:
                for sent1, sent2 in zip(Tx_word, Rx_word):
                    f.write(f"Transmitted: {sent1}\nReceived: {sent2}\n\n")

            bleu_score_1 = []
            #bleu_score_2 = []
            for sent1, sent2 in zip(Tx_word, Rx_word):
                bleu_score_1.append(bleu_score_1gram.compute_blue_score(sent1, sent2))  # 1-gram
                #bleu_score_2.append(bleu_score_2gram.compute_blue_score(sent1, sent2))  # 2-grams
                # bleu_score.append(bleu_score_3gram.compute_blue_score(sent1, sent2))  # 3-grams
                # bleu_score.append(bleu_score_4gram.compute_blue_score(sent1, sent2))  # 4-grams
            os.makedirs('/log', exist_ok=True)
            with open('/log/bleu_score_1gram.txt', 'w') as f1: #, open('/log/bleu_score_2gram.txt', 'w') as f2:
                f1.write('\n'.join(map(str, bleu_score_1)) + '\n')
                #f2.write('\n'.join(map(str, bleu_score_2)) + '\n')
            
            bleu_score_1 = np.array(bleu_score_1)
            bleu_score_1 = np.mean(bleu_score_1, axis=1)
            #bleu_score_2 = np.array(bleu_score_2)
            #bleu_score_2 = np.mean(bleu_score_2, axis=1)
            score_1gram.append(bleu_score_1)
            #score_2gram.append(bleu_score_2)

    score1 = np.mean(np.array(score_1gram), axis=0)
    #score2 = np.mean(np.array(score_2gram), axis=0)
    

    return score1 #, score2





