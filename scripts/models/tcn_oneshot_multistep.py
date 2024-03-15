"""
Temporal Convolution Networks
-------
Paper      : An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling
Authors    : Shaojie Bai, J. Zico Kolter and Vladlen Koltun.
Paper Link : https://arxiv.org/abs/1803.01271
Github     : https://github.com/locuslab/TCN
"""

import torch.nn as nn
from torch.nn.utils.parametrizations import (
    weight_norm,  # depends on your version of pytorch... we should fix this
)

# from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    """
    A module that truncates the last few entries of a tensor along the last dimension.
    This is useful for removing the padding introduced by a convolution operation with padding,
    making the output size match the input size for certain types of temporal convolutional networks.
    """

    def __init__(self, chomp_size):
        """
        Initializes the Chomp1d module.

        Parameters:
            chomp_size (int): The size to be removed from the end of the tensor.
        """
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        Forward pass for Chomp1d.

        Parameters:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The truncated tensor.
        """
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    A single block for the Temporal Convolutional Network (TCN),
    consisting of two convolutional layers with weight normalization, ReLU activations,
    dropout, and a residual connection.
    """

    def __init__(
        self,
        n_inputs,
        n_outputs,
        kernel_size,
        stride,
        dilation,
        padding,
        dropout=0.2,
    ):
        """
        Initializes the TemporalBlock module.

        Parameters:
            n_inputs (int): Number of input channels.
            n_outputs (int): Number of output channels.
            kernel_size (int): Size of the kernel.
            stride (int): Stride of the convolution.
            dilation (int): Dilation rate of the convolution.
            padding (int): Padding added to both sides of the input.
            dropout (float): Dropout rate.
        """
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1)
            if n_inputs != n_outputs
            else None
        )
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """Initializes weights with a normal distribution for the convolutional layers and the downsample layer, if present."""
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        Forward pass for the TemporalBlock.

        Parameters:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after applying convolutions, activations, dropout, and a residual connection.
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """
    A Temporal Convolutional Network (TCN) composed of multiple TemporalBlock layers,
    designed for sequence modeling tasks.
    """

    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        """
        Initializes the TemporalConvNet module.

        Parameters:
            num_inputs (int): Number of input channels.
            num_channels (list of int): List specifying the number of channels for each layer of the TCN.
            kernel_size (int): Size of the kernel for convolutional layers.
            dropout (float): Dropout rate.
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass for the TemporalConvNet.

        Parameters:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor after passing through all TemporalBlock layers.
        """
        return self.network(x)


class MultiStepTCN(nn.Module):
    """
    A model for multi-step time series forecasting using a Temporal Convolutional Network (TCN)
    followed by linear layers to produce a forecast of multiple steps ahead.
    """

    def __init__(
        self,
        n_variates,
        seq_length,
        output_steps,
        hidden_size,
        kernel_size,
        dropout,
    ):
        """
        Initializes the MultiStepTCN module.

        Parameters:
            n_variates (int): Number of variables in the input time series.
            seq_length (int): Length of the input sequences.
            output_steps (int): Number of steps to forecast.
            hidden_size (list of int): Specifies the number of channels for each layer of the TCN.
            kernel_size (int): Size of the kernel for convolutional layers.
            dropout (float): Dropout rate.
        """
        super(MultiStepTCN, self).__init__()
        self.tcn = TemporalConvNet(
            num_inputs=n_variates,
            num_channels=hidden_size,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        self.fc1 = nn.Linear(hidden_size[-1] * seq_length, 128)
        self.fc2 = nn.Linear(128, output_steps)
        self.relu = nn.ReLU()

        self.hidden_size = hidden_size

    def forward(self, x):
        """
        Forward pass for the MultiStepTCN.

        Parameters:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The forecasted output tensor for multiple steps ahead.
        """
        out = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        out = out.reshape(x.size(0), self.hidden_size[-1] * x.size(1))
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out
