from torch import nn
import torch

class LSTM(nn.Module):
    def __init__(
        self, output_size, device, input_size=1, hidden_layer_size=100,
    ):
        super().__init__()
        self.device = device
        self.hidden_layer_size = hidden_layer_size

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)

        # Fully connected layer
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        # Initialise hidden state and cell state
        h0 = torch.zeros(1, input_seq.size(0), self.hidden_layer_size).to(
            self.device
        )
        c0 = torch.zeros(1, input_seq.size(0), self.hidden_layer_size).to(
            self.device
        )

        # LSTM layer
        lstm_out, _ = self.lstm(input_seq, (h0, c0))

        # Only take the output from the final timestep
        predictions = self.linear(lstm_out[:, -1, :])

        return predictions