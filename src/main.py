import argparse
import os
import json
import torch
from models.DeepSC import DeepSC
from eval.performance import performance
from eval.plots import plot_bleu_scores
from utils.log import read_tensors, view_transmitted_text

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
parser.add_argument('--input-text', type=str, default="This is a sample input text", 
                    help="Input text to process with the model")
parser.add_argument('--task', type=str, required=True, 
                    choices=['performance', 'view_transmitted_text', 'sample', 'read_log', 'plot_bleu'],
                    help="Task to perform: 'performance', 'view_transmitted_text', 'sample', 'read_log', or 'plot_bleu'")



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    args = parser.parse_args()
    SNR = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    args.vocab_file = 'data/' + args.vocab_file

    # -------------------- Load the vocabulary --------------------
    vocab = json.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab['token_to_idx']
    idx_to_token = dict(zip(token_to_idx.values(), token_to_idx.keys()))
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]

    # -------------------- Load the trained model --------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    deepsc = DeepSC(args.num_layers, num_vocab, num_vocab,
                    num_vocab, num_vocab, args.d_model, args.num_heads,
                    args.dff, 0.1).to(device)

    model_paths = []
    for fn in os.listdir(args.checkpoint_path):
        if not fn.endswith('.pth'):
            continue
        idx = int(os.path.splitext(fn)[0].split('_')[-1])  # Read the idx of image
        model_paths.append((os.path.join(args.checkpoint_path, fn), idx))

    model_paths.sort(key=lambda x: x[1])  # Sort the models by the idx

    model_path, _ = model_paths[-1]
    checkpoint = torch.load(model_path, weights_only=True)

    # Adjust the keys in the state_dict
    new_state_dict = {}
    for key, value in checkpoint.items():
        new_key = key.replace("channel_encoder.", "channel_encoder.encoder.")
        new_state_dict[new_key] = value

    deepsc.load_state_dict(new_state_dict)
    print('Model Loaded!')

    if args.task == 'performance':
        # -------------------- Evaluate performance over SNR range --------------------
        score1, score2 = performance(args, SNR, deepsc)
        print(f"1-gram BLEU Score: {score1}")
        #print(f"2-gram BLEU Score: {score2}")

    elif args.task == 'plot_bleu':
        # -------------------- Plot BLEU scores --------------------
        score1, score2 = performance(args, SNR, deepsc)  # Ensure scores are calculated
        plot_bleu_scores(SNR, score1, score2)


    elif args.task == 'view_transmitted_text':
        # -------------------- View Transformed Text --------------------
        view_transmitted_text()


    elif args.task == 'read_log':
        #---------------------- New task to show saved tensors from checkpoints
        read_tensors()

    
