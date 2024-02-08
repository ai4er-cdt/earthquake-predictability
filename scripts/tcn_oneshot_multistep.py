# Import relevant libraries
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    """ Removes the trailing chomp_size from the input tensor. """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """ Defines a single temporal block consisting of a convolutional layer, chomping, and activation. """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        # Weight normalized convolutional layer
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        # Chomp1d removes excess padding to ensure output size matches input size
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        return out


class TemporalConvNet(nn.Module):
    """ Defines the Temporal Convolutional Network architecture. """
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        # Stack multiple temporal blocks
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            # Append temporal block to the network
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        # Sequentially connect the layers
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class MultiStepTCN(nn.Module):
    """
    A PyTorch neural network model using an ulti-step Temporal Convolutional Network model for
    forecasting.


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
        model = MultiStepTCN(N_VARIATES, N_CHANNELS, KERNEL_SIZE, OUTPUT_SIZE, device)
    """
    def __init__(self, n_variates, num_channels, kernel_size, output_size, device):
        """
        Initializes the MultiStepTCN model.

        Parameters:
            - n_variates (int): Number of input variables (features).
            - num_channels (list): List of integers specifying the number of channels in each TCN block.
            - kernel_size (int): Size of the convolutional kernel.
            - output_size (int): Number of features in the output/forecasted values.
            - device (str): Device on which the model is being run (e.g., 'cuda' or 'cpu').
        """
        super(MultiStepTCN, self).__init__()
        self.tcn = TemporalConvNet(n_variates, num_channels, kernel_size)
        self.fc1 = nn.Linear(num_channels[-1], 128)
        self.fc2 = nn.Linear(128, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Performs a forward pass through the MultiStepTCN model.

        Parameters:
            - x (torch.Tensor): Input data tensor with shape (batch_size, seq_length, n_variates).

        Returns:
            - torch.Tensor: Output tensor with shape (batch_size, output_size).
        """
        out = self.tcn(x.transpose(1, 2))  # TCN expects shape (batch_size, input_dim, seq_length)
        out = out[:, :, -1]  # taking the output of the last time step
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out
