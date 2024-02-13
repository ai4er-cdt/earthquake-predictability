# Import relevant libraries
import torch
import torch.nn as nn


class MultiStepLSTMSingleLayer(nn.Module):
    """
    A PyTorch neural network model using an LSTM for multi-step time series forecasting.


    Attributes:
        n_variates (int): Number of input variables (features).
        hidden_size (int): Number of features in the hidden state of the LSTM.
        n_layers (int): Number of recurrent layers in the LSTM.
        output_size (int): Number of features in the output/forecasted values.
        device (str): Device on which the model is being run (e.g., 'cuda' or 'cpu').

    Methods:
        forward(x):
            Performs a forward pass through the LSTM layer.

    Example:
        model = MultiStepLstmSingleLayer(N_VARIATES, HIDDEN_SIZE, N_LAYERS, OUTPUT_SIZE, device)
    """

    def __init__(self, n_variates, hidden_size, n_layers, output_size, device):
        """
        Initializes the MultiStepLSTM model.

        Parameters:
            - n_variates (int): Number of input variables (features).
            - hidden_size (int): Number of features in the hidden state of the LSTM.
            - n_layers (int): Number of recurrent layers in the LSTM.
            - output_size (int): Number of features in the output/forecasted values.
            - device (str): Device on which the model is being run (e.g., 'cuda' or 'cpu').
        """
        super().__init__()

        # Set model attributes
        self.n_variates = n_variates
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.output_size = output_size
        self.device = device

        # LSTM layer with specified input size, hidden size, and batch_first
        self.lstm = nn.LSTM(
            input_size=self.n_variates,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers,
            batch_first=True,
        )

        # Linear layer mapping the LSTM output to the forecasted values
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        """
        Performs a forward pass through the LSTM layer.

        Parameters:
            - x (torch.Tensor): Input data tensor with shape (batch_size, seq_length, n_variates).

        Returns:
            - torch.Tensor: Output tensor with shape (batch_size, output_size).
        """
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(
            self.device
        )
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(
            self.device
        )

        # LSTM layer
        lstm_out, _ = self.lstm(x, (h0, c0))

        # Extract the last time step output from the LSTM output
        lstm_out = lstm_out[:, -1, :]

        # Linear layer for the final output (forecasted values)
        output = self.linear(lstm_out)

        return output


class MultiStepLSTMMultiLayer(nn.Module):
    """
    A PyTorch neural network model using an LSTM for multi-step time series forecasting.
    Credit - Pritt's model!!!


    Attributes:
        n_variates (int): Number of input variables (features).
        hidden_size (int): Number of features in the hidden state of the LSTM.
        n_layers (int): Number of recurrent layers in the LSTM.
        output_size (int): Number of features in the output/forecasted values.
        device (str): Device on which the model is being run (e.g., 'cuda' or 'cpu').

    Methods:
        forward(x):
            Performs a forward pass through the LSTM layer.

    Example:
        model = MultiStepLstmMultiLayer(N_VARIATES, HIDDEN_SIZE, N_LAYERS, OUTPUT_SIZE, device)
    """

    def __init__(self, n_variates, hidden_size, n_layers, output_size, device):
        """
        Initializes the MultiStepLSTM model.

        Parameters:
            - n_variates (int): Number of input variables (features).
            - hidden_size (int): Number of features in the hidden state of the LSTM.
            - n_layers (int): Number of recurrent layers in the LSTM.
            - output_size (int): Number of features in the output/forecasted values.
            - device (str): Device on which the model is being run (e.g., 'cuda' or 'cpu').
        """
        super().__init__()
        # Set model attributes
        self.n_variates = n_variates
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.output_size = output_size
        self.device = device

        # LSTM layer with specified input size, hidden size, and batch_first
        self.lstm = nn.LSTM(
            input_size=self.n_variates,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers,
            batch_first=True,
        )

        self.fc1 = nn.Linear(n_layers * hidden_size, 128)
        self.fc2 = nn.Linear(128, output_size)
        self.relu = nn.ReLU6()

    def forward(self, x):
        """
        Performs a forward pass through the LSTM layer.

        Parameters:
            - x (torch.Tensor): Input data tensor with shape (batch_size, seq_length, n_variates).

        Returns:
            - torch.Tensor: Output tensor with shape (batch_size, output_size).
        """
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(
            x.device
        )
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size).to(
            x.device
        )

        _, (hn, cn) = self.lstm(x, (h0, c0))
        hn = hn.view(x.size(0), self.n_layers * self.hidden_size)
        out = self.relu(hn)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out
