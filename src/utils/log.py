
import torch
import os

def read_tensors():
    try:
        encoder_output = torch.load('checkpoints/encoder_output.pt', weights_only=True)
        print(f"Encoder Output Shape: {encoder_output.shape}, Mean: {encoder_output.mean()}, Std: {encoder_output.std()}, Type: {encoder_output.dtype}, Head: {encoder_output[0]}")

        channel_encoded_output = torch.load('checkpoints/channel_encoded_output.pt', weights_only=True)
        print(f"Channel Encoded Output Shape: {channel_encoded_output.shape}, Mean: {channel_encoded_output.mean()}, Std: {channel_encoded_output.std()}, Type: {channel_encoded_output.dtype}, Head: {channel_encoded_output[0]}")

        channel_decoded_output = torch.load('checkpoints/channel_decoded_output.pt', weights_only=True)
        print(f"Channel Decoded Output Shape: {channel_decoded_output.shape}, Mean: {channel_decoded_output.mean()}, Std: {channel_decoded_output.std()}, Type: {channel_decoded_output.dtype}, Head: {channel_decoded_output[0]}")

        decoder_output = torch.load('checkpoints/decoder_output.pt', weights_only=True)
        print(f"Decoder Output Shape: {decoder_output.shape}, Mean: {decoder_output.mean()}, Std: {decoder_output.std()}, Type: {decoder_output.dtype}, Head: {decoder_output[0]}")

    except FileNotFoundError:
        print("One or more checkpoint files are missing.")




def view_transmitted_text():
    log_file_path = '/log/transmitted_text.txt'
    if os.path.exists(log_file_path):
        with open(log_file_path, 'r') as f:
            lines = f.readlines()
            if lines:
                transmitted_lines = [line for line in lines if line.startswith("Transmitted:")]
                received_lines = [line for line in lines if line.startswith("Received:")]
                if transmitted_lines and received_lines:
                    print("Head of original data:")
                    print(transmitted_lines[0])  # Print only the first transmitted instance
                    print("Head of transmitted data:")
                    print(received_lines[0])  # Print only the first received instance
                else:
                    print("No valid transmitted or received data found in the log file.")
            else:
                print("Log file is empty.")
    else:
        print("Transmitted text log file not found.")