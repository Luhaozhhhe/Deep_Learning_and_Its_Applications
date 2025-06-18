import torch
import torch.nn as nn

class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, num_classes):

        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=False)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, h0=None, c0=None, lengths=None):

        batch_size = x.size(1)
        if h0 is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=x.dtype, device=x.device)
        if c0 is None:
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, dtype=x.dtype, device=x.device)
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=False, enforce_sorted=True)

        out, (h, c) = self.lstm(x, (h0, c0))

        idx = [-1] * batch_size
        if lengths is not None:
            out, idx = nn.utils.rnn.pad_packed_sequence(out, batch_first=False)
            idx = [i - 1 for i in idx]

        last_sequence_list = []
        for i in range(batch_size):
            last_sequence_list.append(out[idx[i], i, :])
        out = torch.stack(last_sequence_list)

        out = self.fc(out)
        return out, (h, c)
