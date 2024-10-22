import json

def load_vocab(vocab_file_path):
    vocab = json.load(open(vocab_file_path, 'rb'))
    token_to_idx = vocab['token_to_idx']
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]
    return token_to_idx, pad_idx, start_idx, end_idx