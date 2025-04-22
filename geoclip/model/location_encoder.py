import os
import torch
import torch.nn as nn
from .rff import GaussianEncoding

class LocationEncoderCapsule(nn.Module):
    def __init__(self, sigma):
        super(LocationEncoderCapsule, self).__init__()
        rff_encoding = GaussianEncoding(sigma=sigma, input_size=2, encoded_size=256)
        self.km = sigma
        self.capsule = nn.Sequential(rff_encoding,
                                     nn.Linear(512, 1024),
                                     nn.ReLU(),
                                     nn.Linear(1024, 1024),
                                     nn.ReLU(),
                                     nn.Linear(1024, 1024),
                                     nn.ReLU())
        self.head = nn.Sequential(nn.Linear(1024, 512))

    def forward(self, x):
        x = self.capsule(x)
        x = self.head(x)
        return x

class LocationEncoder(nn.Module):
    def __init__(self, sigma=[2**0, 2**4, 2**8]):
        super(LocationEncoder, self).__init__()
        self.sigma = sigma
        self.n = len(self.sigma)

        for i, s in enumerate(self.sigma):
            self.add_module('LocEnc' + str(i), LocationEncoderCapsule(sigma=s))

    # def load_weights(self, pretrained_path: str):
    #     self.load_state_dict(torch.load(pretrained_path, weights_only=True))

    def forward(self, location):
        device = next(self.parameters()).device
        if location.device != device:
            location = location.to(device)
        x_max, y_max = 3840, 2160
        location_norm = location.clone()
        location_norm[:, 0] = location[:, 0] / x_max
        location_norm[:, 1] = location[:, 1] / y_max
        # location_norm[:, 2] = location[:, 2] # angle from 0 to 359

        location_features = torch.zeros(location.shape[0], 512).to(location.device)
        for i in range(self.n):
            location_features += self._modules['LocEnc' + str(i)](location_norm)
        
        return location_features
