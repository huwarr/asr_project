import torch
from torch import nn
from torch.nn import Sequential

from hw_asr.base import BaseModel


class DeepSpeech2(BaseModel):
    def __init__(self, n_feats, n_class, fc_hidden=512, tau=80, **batch):
        super().__init__(n_feats, n_class, **batch)
        # From the paper:
        #     The best English model has 2 layers of 2D convolution, 
        #     followed by 3 layers of unidirectional recurrent layers 
        #     with 2560 GRU cells each, followed by a lookahead convolution 
        #     layer with τ = 80, trained with BatchNorm and SortaGrad.

        # Parameters of convolutions: page 5 (Table 2)

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

        self.rnns = nn.ModuleList([
            nn.GRU(input_size=input_size, hidden_size=fc_hidden, batch_first=True),
            nn.GRU(input_size=fc_hidden, hidden_size=fc_hidden, batch_first=True),
            nn.GRU(input_size=fc_hidden, hidden_size=fc_hidden, batch_first=True)
        ])
        self.batchNorms = nn.ModuleList([
            nn.BatchNorm1d(num_features=fc_hidden),
            nn.BatchNorm1d(num_features=fc_hidden),
            nn.BatchNorm1d(num_features=fc_hidden)
        ])

        # Lookahead is just a CNN, but we move input to the right by tau/2 positions
        self.lookaheadConv = nn.Conv1d(
            in_channels=fc_hidden, out_channels=fc_hidden, kernel_size=tau + 1, padding=tau//2, groups=fc_hidden
        )

        self.lookahead = Sequential(
            self.lookaheadConv,
            nn.ReLU(),
            nn.BatchNorm1d(num_features=fc_hidden)
        )
        
        # Logits
        self.head = nn.Linear(in_features=fc_hidden, out_features=n_class)


    def forward(self, spectrogram, spectrogram_length, **batch):
        # add channels
        x = torch.unsqueeze(spectrogram, dim=1)
        x = self.cnns(x)
        # reshape for RNNs (4d -> 3d)
        # (batch size X features X sequence length)
        x = x.view(x.shape[0], -1, x.shape[-1])
        # GRU takes input of shape: (batch size X sequence length X features)
        x = x.transpose(1, 2)
        # finally, RNNs
        for i, rnn in enumerate(self.rnns):
            # pack padded sequence for rnn <3
            x = nn.utils.rnn.pack_padded_sequence(x, self.transform_input_lengths(spectrogram_length), batch_first=True, enforce_sorted=False)
            # only hidden states go further
            x, _ = rnn(x)
            # unpack sequence for batch norm
            x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
            # BatchNorm1d takes input of shape: (batch size X features X sequence length)
            x = self.batchNorms[i](x.transpose(1, 2)).transpose(1, 2)
        # lookahead Convolution and co.
        x = self.lookahead(x.transpose(1, 2))
        # logits
        logits = self.head(x.transpose(1, 2))
        return {"logits": logits}

    def transform_input_lengths(self, input_lengths):
        # Only CNNs transform length
        # first Convolution:
        lengths = (
            input_lengths + 2 * self.cnns[0].padding[1] - self.cnns[0].dilation[1] * (self.cnns[0].kernel_size[1] - 1) - 1
        ) // self.cnns[0].stride[1] + 1
        # second convolution:
        lengths = (
            lengths + 2 * self.cnns[3].padding[1] - self.cnns[3].dilation[1] * (self.cnns[3].kernel_size[1] - 1) - 1
        ) // self.cnns[3].stride[1] + 1
        # new lengths
        return lengths

