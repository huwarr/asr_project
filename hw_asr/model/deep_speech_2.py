import torch
from torch import nn
from torch.nn import Sequential

from hw_asr.base import BaseModel


class DeepSpeech2(BaseModel):
    def __init__(self, n_feats, n_class, fc_hidden=512, tau=80, num_cells=2560, **batch):
        super().__init__(n_feats, n_class, **batch)
        # GRU
        # three layers of 2D convolution
        # 9-layer, 7 RNN + BatchNorm + frequency Convolution
        #      or
        # 5-layer, 3 RNN + BatchNorm

        # The best English model has 2 layers of 2D convolution, 
        # followed by 3 layers of unidirectional recurrent layers 
        # with 2560 GRU cells each, followed by a lookahead convolution 
        # layer with τ = 80, trained with BatchNorm and SortaGrad.

        # Parameters of concolutions: page 5 (Table 2)

        self.num_cells = num_cells

        self.cnns = Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            # They didn't mention activation in the paper.....
            nn.ReLU(),
            nn.BatchNorm2d(num_features=32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=32)
        )

        input_size = 1024

        self.rnns = [
            nn.GRU(input_size=input_size, hidden_size=fc_hidden, batch_first=True),
            nn.GRU(input_size=fc_hidden, hidden_size=fc_hidden, batch_first=True),
            nn.GRU(input_size=fc_hidden, hidden_size=fc_hidden, batch_first=True)
        ]
        self.batchNorms = [
            nn.BatchNorm1d(num_features=fc_hidden),
            nn.BatchNorm1d(num_features=fc_hidden),
            nn.BatchNorm1d(num_features=fc_hidden)
        ]

        # Lookahead is just a CNN, but we move input to the right by tau/2 positions
        self.lookaheadConv = nn.Conv1d(
            in_channels=fc_hidden, out_channels=fc_hidden, kernel_size=tau + 1, padding=tau//2
        )

        self.lookahead = Sequential(
            self.lookaheadConv,
            nn.ReLU(),
            nn.BatchNorm1d(num_features=fc_hidden)
        )
        
        # Logits
        self.head = nn.Linear(in_features=fc_hidden, out_features=n_class)


    def forward(self, spectrogram, **batch):
        # add checnnels
        x = torch.unsqueeze(spectrogram, dim=1)
        x = self.cnns(x)
        # reshape for RNNs (3d -> 2d)
        # (batch size X features X sequence length)
        x = x.view(x.shape[0], -1, x.shape[-1])
        # pad/crop to get self.num_cells cells in RNNs
        # = min(x.shape[-1], self.num_cells)
        length = spectrogram.shape[-1]
        new_x = torch.zeros(x.shape[0], x.shape[1], length)
        new_x[:, :, :x.shape[-1]] = x
        # GRU takes input of shape: (batch size X sequence length X features)
        x = new_x.transpose(1, 2)
        #x = x.transpose(1, 2)
        # finally, RNNs
        for i, rnn in enumerate(self.rnns):
            # only hidden states go further
            x, _ = rnn(x)
            # BatchNorm1d takes input of shape: (batch size X features X sequence length)
            x = self.batchNorms[i](x.transpose(1, 2)).transpose(1, 2)
        # lookahead Convolution and co.
        x = self.lookahead(x.transpose(1, 2))
        # logits
        logits = self.head(x.transpose(1, 2))
        return {"logits": logits}

    def transform_input_lengths(self, input_lengths):
        return input_lengths

