# Import relevant libraries
import torch
import torch.nn as nn


class Conv2DLSTMModel(nn.Module):
    def __init__(self, n_variates, input_steps, output_steps, hidden_size):
        super(Conv2DLSTMModel, self).__init__()

        self.n_variates = n_variates
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.hidden_size = hidden_size

        self.conv2d = nn.Conv2d(
            in_channels=1, out_channels=64, kernel_size=(3, 3), padding=(1, 1)
        )
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(
            input_size=64 * n_variates,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.dense = nn.Linear(hidden_size, output_steps * n_variates)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.relu(x)

        batch_size = x.size(0)

        x = x.view(batch_size, self.input_steps, -1)
        h0 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)

        _, (hn, cn) = self.lstm(x, (h0, c0))
        hn = hn.squeeze(0)
        output = self.dense(hn)
        output = output.view(batch_size, self.output_steps, self.n_variates)

        return output
