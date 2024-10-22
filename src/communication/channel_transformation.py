import torch
import torch.nn as nn
import torch.nn.functional as F
import logging


# Encoder with logging of the input and output shapes
class ChannelEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(ChannelEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim) 
        )

    def forward(self, x):
        logging.info(f"Input to ChannelEncoder shape: {x.shape}")
        encoded_output = self.encoder(x)
        logging.info(f"ChannelEncoder output shape: {encoded_output.shape}")
        torch.save(encoded_output, 'checkpoints/channel_encoded_output.pt')
        return encoded_output



class ChannelDecoder(nn.Module):
    def __init__(self, in_features, size1, size2):
        super(ChannelDecoder, self).__init__()

        self.linear1 = nn.Linear(in_features, size1)
        self.linear2 = nn.Linear(size1, size2)
        self.linear3 = nn.Linear(size2, size1)
        # self.linear4 = nn.Linear(size1, d_model)

        self.layernorm = nn.LayerNorm(size1, eps=1e-6)

    def forward(self, x):
        logging.info(f"Input to ChannelDecoder shape: {x.shape}")
        x1 = self.linear1(x)
        x2 = F.relu(x1)
        x3 = self.linear2(x2)
        x4 = F.relu(x3)
        x5 = self.linear3(x4)

        output = self.layernorm(x1 + x5)

        logging.info(f"ChannelDecoder output shape: {output.shape}")
        torch.save(output, 'checkpoints/channel_decoded_output.pt')

        return output
    



# RIS Configuration


# CSI Handling


