import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Input gate
        self.W_ii = nn.Parameter(torch.randn(hidden_size, input_size))
        self.W_hi = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_i = nn.Parameter(torch.zeros(hidden_size))
        
        # Forget gate
        self.W_if = nn.Parameter(torch.randn(hidden_size, input_size))
        self.W_hf = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_f = nn.Parameter(torch.zeros(hidden_size))
        
        # Output gate
        self.W_io = nn.Parameter(torch.randn(hidden_size, input_size))
        self.W_ho = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_o = nn.Parameter(torch.zeros(hidden_size))
        
        # Cell state
        self.W_ig = nn.Parameter(torch.randn(hidden_size, input_size))
        self.W_hg = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_g = nn.Parameter(torch.zeros(hidden_size))
        
        # Initialize weights
        self.init_weights()
        
    def init_weights(self):
        for param in self.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param)
            else:
                nn.init.zeros_(param)
                
    def forward(self, x, h_prev, c_prev):

        # Input gate
        i_t = torch.sigmoid(x @ self.W_ii.t() + h_prev @ self.W_hi.t() + self.b_i)
        
        # Forget gate
        f_t = torch.sigmoid(x @ self.W_if.t() + h_prev @ self.W_hf.t() + self.b_f)
        
        # Output gate
        o_t = torch.sigmoid(x @ self.W_io.t() + h_prev @ self.W_ho.t() + self.b_o)
        
        # Cell candidate
        g_t = torch.tanh(x @ self.W_ig.t() + h_prev @ self.W_hg.t() + self.b_g)
        
        # Cell state update
        c_t = f_t * c_prev + i_t * g_t
        
        # Hidden state update
        h_t = o_t * torch.tanh(c_t)
        
        return h_t, c_t

class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(CustomLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Create LSTM layers
        self.lstm_cells = nn.ModuleList([
            LSTMCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x, h0=None, lengths=None):
        """
        x: input tensor of shape (seq_len, batch_size, input_size)
        h0: initial hidden state (optional)
        lengths: sequence lengths (optional)
        """
        batch_size = x.size(1)
        seq_len = x.size(0)
        
        # Initialize hidden and cell states
        if h0 is None:
            h_states = [torch.zeros(batch_size, self.hidden_size, device=x.device) 
                       for _ in range(self.num_layers)]
            c_states = [torch.zeros(batch_size, self.hidden_size, device=x.device) 
                       for _ in range(self.num_layers)]
        else:
            h_states, c_states = h0
            
        # Process each time step
        outputs = []
        for t in range(seq_len):
            x_t = x[t]
            
            # Process each layer
            for layer in range(self.num_layers):
                if layer == 0:
                    h_states[layer], c_states[layer] = self.lstm_cells[layer](
                        x_t, h_states[layer], c_states[layer]
                    )
                else:
                    h_states[layer], c_states[layer] = self.lstm_cells[layer](
                        h_states[layer-1], h_states[layer], c_states[layer]
                    )
            
            outputs.append(h_states[-1])
            
        # Stack outputs
        outputs = torch.stack(outputs)
        
        # Use the last output for classification
        final_output = self.fc(outputs[-1])
        final_output = self.softmax(final_output)
        
        return final_output, (h_states, c_states)

    def init_hidden(self, batch_size, device):
        h_states = [torch.zeros(batch_size, self.hidden_size, device=device) 
                   for _ in range(self.num_layers)]
        c_states = [torch.zeros(batch_size, self.hidden_size, device=device) 
                   for _ in range(self.num_layers)]
        return h_states, c_states