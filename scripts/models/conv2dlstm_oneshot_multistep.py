# Import the necessary modules from PyTorch for building the model
import torch
import torch.nn as nn


# Define the Conv2DLSTMModel class, which inherits from nn.Module, the base class for all neural network modules in PyTorch
class Conv2DLSTMModel(nn.Module):
    """
    This model combines 2D convolutional layers with LSTM layers for time series forecasting.
    It is particularly useful for processing spatio-temporal data where both spatial and temporal features are crucial.
    The model first applies a convolutional layer to extract spatial features, then uses an LSTM layer to capture temporal dynamics.

    Attributes:
        n_variates (int): The number of variables (features) in each time step of the input data.
        input_steps (int): The number of time steps in the input data sequence.
        output_steps (int): The desired number of time steps to forecast.
        hidden_size (int): The number of features in the hidden state of the LSTM layer.

    Methods:
        forward(x): Defines the forward pass of the model.
    """

    def __init__(self, n_variates, input_steps, output_steps, hidden_size):
        super(Conv2DLSTMModel, self).__init__()

        # Initialize model attributes
        self.n_variates = n_variates  # Number of features per time step
        self.input_steps = input_steps  # Number of time steps in the input
        self.output_steps = (
            output_steps  # Number of time steps in the output prediction
        )
        self.hidden_size = hidden_size  # Size of the LSTM hidden layer

        # Define a 2D convolutional layer to process spatial features
        # The layer has 1 input channel, 64 output channels, a kernel size of 3x3, and padding of 1 to preserve spatial dimensions
        self.conv2d = nn.Conv2d(
            in_channels=1, out_channels=64, kernel_size=(3, 3), padding=(1, 1)
        )

        # Define a ReLU activation function to introduce non-linearity after the convolutional layer
        self.relu = nn.ReLU()

        # Define an LSTM layer to process temporal features
        # The input size is the product of the number of output channels from the conv2d layer and the number of features
        # The hidden_size parameter sets the size of the LSTM's hidden state
        self.lstm = nn.LSTM(
            input_size=64 * n_variates,
            hidden_size=hidden_size,
            batch_first=True,
        )

        # Define a linear (fully connected) layer to map the LSTM output to the desired output size
        # The output size is the product of the number of output time steps and the number of variables per time step
        self.dense = nn.Linear(hidden_size, output_steps * n_variates)

    def forward(self, x):
        # Define the forward pass through the model
        x = self.conv2d(x)  # Pass the input through the convolutional layer
        x = self.relu(x)  # Apply ReLU activation

        # Reshape the output to fit the LSTM layer's expected input format
        batch_size = x.size(0)
        x = x.view(
            batch_size, self.input_steps, -1
        )  # Flatten spatial dimensions and combine with feature dimension

        # Initialize the hidden and cell states of the LSTM with zeros
        h0 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)

        # Pass the reshaped input through the LSTM layer
        _, (hn, cn) = self.lstm(x, (h0, c0))

        hn = hn.squeeze(
            0
        )  # Remove the first dimension from the LSTM output to match the expected input of the dense layer

        # Pass the LSTM output through the dense layer to get the final prediction
        out = self.dense(hn)

        # Reshape the output to have the expected dimensions (batch size, output time steps, number of variables)
        out = out.view(batch_size, self.output_steps, self.n_variates)

        return out
