from utils import *


class LSTM(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            m_input_size: int,
            num_layers: int = 1,
            bias: bool = True,
            dropout: float = 0.5,
            bidirectional: bool = False,
            device=torch.device("cpu")
    ):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.device = device

        self.forward_lstm = nn.ModuleList()
        self.forward_dropout = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.forward_lstm.append(LSTMBlock(input_size, hidden_size, m_input_size, bias, device))
            else:
                self.forward_lstm.append(LSTMBlock(hidden_size, hidden_size, m_input_size, bias, device))
            if i != (num_layers - 1) and num_layers > 1 and dropout != 0.0:
                self.forward_dropout.append(nn.Dropout(dropout))
            else:
                self.forward_dropout.append(nn.Identity())

        if bidirectional:
            self.backward_lstm = nn.ModuleList()
            self.backward_dropout = nn.ModuleList()
            for i in range(num_layers):
                if i == 0:
                    self.backward_lstm.append(LSTMBlock(input_size, hidden_size, m_input_size, bias, device))
                else:
                    self.backward_lstm.append(LSTMBlock(hidden_size, hidden_size, m_input_size, bias, device))
                if i != (num_layers - 1) and num_layers > 1 and dropout != 0.0:
                    self.backward_dropout.append(nn.Dropout(dropout))
                else:
                    self.backward_dropout.append(nn.Identity())

    def __call__(self, x, m):
        output, h, c = self.forward(x, m)
        return output, (h, c)

    def init_state(
            self,
            batch_size: int = 8
    ):
        h_f = torch.zeros([self.num_layers, batch_size, self.hidden_size], device=self.device)
        c_f = torch.zeros([self.num_layers, batch_size, self.hidden_size], device=self.device)
        h_b = torch.zeros([self.num_layers, batch_size, self.hidden_size], device=self.device)
        c_b = torch.zeros([self.num_layers, batch_size, self.hidden_size], device=self.device)
        return h_f, c_f, h_b, c_b

    def forward(self, x, m, state=None):
        batch_size, seq_len, _ = x.size()

        if state is None:
            h_f, c_f, h_b, c_b = self.init_state(batch_size)
        else:
            h_f, c_f, h_b, c_b = state

        outputs_f = torch.zeros([batch_size, seq_len, self.hidden_size], device=self.device)
        outputs_b = torch.zeros([batch_size, seq_len, self.hidden_size], device=self.device)
        # Forward pass
        for t in range(seq_len):
            x_t = x[:, t, :]
            m_t = m[:, t, :]
            new_h_f = []
            new_c_f = []
            for layer in range(self.num_layers):
                x_t, (h_temp, c_temp) = self.forward_lstm[layer](x_t, (h_f[layer], c_f[layer]), m_t)
                h_temp = self.forward_dropout[layer](h_temp)
                new_h_f.append(h_temp)
                new_c_f.append(c_temp)
            h_f = torch.stack(new_h_f)
            c_f = torch.stack(new_c_f)
            outputs_f[:, t, :] = h_f[-1]
        if self.bidirectional:
            # Backward pass
            for t in reversed(range(seq_len)):
                x_t = x[:, t, :]
                m_t = m[:, t, :]
                new_h_b = []
                new_c_b = []
                for layer in range(self.num_layers):
                    x_t, (h_temp, c_temp) = self.forward_lstm[layer](x_t, (h_b[layer], c_b[layer]), m_t)
                    h_temp = self.forward_dropout[layer](h_temp)
                    new_h_b.append(h_temp)
                    new_c_b.append(c_temp)
                h_b = torch.stack(new_h_b)
                c_b = torch.stack(new_c_b)
                outputs_b[:, t, :] = h_b[-1]
            outputs_b.flip(dims=[1])
            # Concatenate forward and backward outputs
            outputs = torch.cat([outputs_f, outputs_b], dim=-1)
            h_final = torch.cat([h_f, h_b], dim=0)
            c_final = torch.cat([c_f, c_b], dim=0)
        else:
            outputs = outputs_f
            h_final, c_final = h_f, c_f

        return outputs, h_final, c_final


class LSTMBlock(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            hidden_dim: int,
            m_dim: int,
            bias: bool = True,
            device=torch.device("cpu")
    ):
        super(LSTMBlock, self).__init__()
        # x/h_prev/m Transformations
        self.x_f = nn.Linear(in_features=embedding_dim, out_features=hidden_dim, bias=True, device=device)
        self.h_f = nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True, device=device)
        self.m_f = nn.Linear(in_features=m_dim, out_features=hidden_dim, bias=True, device=device)
        self.x_i = nn.Linear(in_features=embedding_dim, out_features=hidden_dim, bias=True, device=device)
        self.h_i = nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True, device=device)
        self.m_i = nn.Linear(in_features=m_dim, out_features=hidden_dim, bias=True, device=device)
        self.x_o = nn.Linear(in_features=embedding_dim, out_features=hidden_dim, bias=True, device=device)
        self.h_o = nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True, device=device)
        self.m_o = nn.Linear(in_features=m_dim, out_features=hidden_dim, bias=True, device=device)
        self.x_c = nn.Linear(in_features=embedding_dim, out_features=hidden_dim, bias=True, device=device)
        self.h_c = nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=True, device=device)
        # Biases
        if bias:
            self.b_f = nn.Parameter(torch.zeros(hidden_dim)).to(device)
            self.b_i = nn.Parameter(torch.zeros(hidden_dim)).to(device)
            self.b_o = nn.Parameter(torch.zeros(hidden_dim)).to(device)
            self.b_c_tilde = nn.Parameter(torch.zeros(hidden_dim)).to(device)
        else:
            self.b_f = torch.zeros(hidden_dim).to(device)
            self.b_i = torch.zeros(hidden_dim).to(device)
            self.b_o = torch.zeros(hidden_dim).to(device)
            self.b_c_tilde = torch.zeros(hidden_dim).to(device)
        # Activation Functions
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def __call__(self, x, state_prev, m):
        o, state = self.forward(x, state_prev, m)
        return o, state

    def forward(self, x, state_prev, m):
        h_prev, c_prev = state_prev
        f = self.sigmoid(self.x_f(x) + self.h_f(h_prev) + self.m_f(m) + self.b_f)
        i = self.sigmoid(self.x_i(x) + self.h_i(h_prev) + self.m_i(m) + self.b_i)
        o = self.sigmoid(self.x_o(x) + self.h_o(h_prev) + self.m_o(m) + self.b_o)
        c_tilde = self.tanh(self.x_c(x) + self.h_c(h_prev) + self.b_c_tilde)
        c = (f * c_prev) + (i * c_tilde)
        h = o * c

        return o, (h, c)


class SpatioTemporal_LSTM(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            m_input_size: int,
            num_layers: int = 1,
            bidirectional: bool = False,
            device=torch.device("cpu")
    ):
        super(SpatioTemporal_LSTM, self).__init__()
        self.device = device

        self.lstm = LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            m_input_size=m_input_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            device=device
        )
        out_num_features = hidden_size
        if bidirectional:
            out_num_features = 2 * out_num_features
        self.out_normalization = nn.BatchNorm1d(num_features=out_num_features, device=device)
        self.h_normalization = nn.BatchNorm1d(num_features=hidden_size, device=device)
        self.c_normalization = nn.BatchNorm1d(num_features=hidden_size, device=device)

    def __call__(self, x: torch.Tensor, m):
        y = self.forward(x, m)
        return y

    def forward(self, x: torch.Tensor, m):
        output, (hidden, cell) = self.lstm(x, m)
        last_output = self.out_normalization(output[-1])
        last_hidden = self.h_normalization(hidden[-1])
        last_cell = self.c_normalization(cell[-1])
        return last_output, (last_hidden, last_cell)
