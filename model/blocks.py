import torch
import torch.nn as nn
from torch.nn import functional as F


class LinearNorm(nn.Module):
    """LinearNorm Projection"""

    def __init__(self, in_features, out_features, bias=False, spectral_norm=False):
        super(LinearNorm, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

        nn.init.xavier_uniform_(self.linear.weight)
        if bias:
            nn.init.constant_(self.linear.bias, 0.0)
        if spectral_norm:
            self.linear = nn.utils.spectral_norm(self.linear)

    def forward(self, x):
        x = self.linear(x)
        return x


class FCBlock(nn.Module):
    """Fully Connected Block"""

    def __init__(
        self,
        in_features,
        out_features,
        activation=None,
        bias=False,
        dropout=None,
        spectral_norm=False,
    ):
        super(FCBlock, self).__init__()
        self.fc_layer = nn.Sequential()
        self.fc_layer.add_module(
            "fc_layer",
            LinearNorm(
                in_features,
                out_features,
                bias,
                spectral_norm,
            ),
        )
        if activation is not None:
            self.fc_layer.add_module("activ", activation)
        self.dropout = dropout

    def forward(self, x):
        x = self.fc_layer(x)
        if self.dropout is not None:
            x = F.dropout(x, self.dropout, self.training)
        return x


class ReluBlock(nn.Module):

    def __init__(self, c_in, c_out, kernel_size, stride, padding):
        super(ReluBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                c_in,
                c_out,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=True,
            ),
            nn.InstanceNorm2d(c_out),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.conv(x)


class SkipGatedBlock(nn.Module):

    def __init__(self, c_in, c_out, kernel_size, stride, padding):
        super(SkipGatedBlock, self).__init__()
        self.conv = nn.Conv2d(
            c_in,
            c_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )
        self.gate = nn.Conv2d(
            c_in,
            c_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )
        self.skip_connection = c_in == c_out

    def forward(self, x):
        conv_output = self.conv(x)
        gated_output = torch.sigmoid(self.gate(x))
        output = conv_output * gated_output
        if self.skip_connection:
            output += x
        return output


class Conv2Encoder(nn.Module):
    def __init__(self, input_channel=1, hidden_dim=64, block="skip", n_layers=3):
        super(Conv2Encoder, self).__init__()
        if block == "skip":
            core = SkipGatedBlock
        elif block == "relu":
            core = ReluBlock
        else:
            raise ValueError(f"Invalid block type: {block}")

        layers = [
            core(
                c_in=input_channel, c_out=hidden_dim, kernel_size=3, stride=1, padding=1
            )
        ]

        for i in range(n_layers - 1):
            layers.append(
                core(
                    c_in=hidden_dim,
                    c_out=hidden_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class WatermarkEmbedder(nn.Module):

    def __init__(self, input_channel=1, hidden_dim=64, block="skip", n_layers=4):
        super(WatermarkEmbedder, self).__init__()
        if block == "skip":
            core = SkipGatedBlock
        elif block == "relu":
            core = ReluBlock
        else:
            raise ValueError(f"Invalid block type: {block}")

        layers = [
            core(
                c_in=input_channel, c_out=hidden_dim, kernel_size=3, stride=1, padding=1
            )
        ]

        for i in range(n_layers - 2):
            layers.append(
                core(
                    c_in=hidden_dim,
                    c_out=hidden_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )

        layers.append(
            core(c_in=hidden_dim, c_out=1, kernel_size=1, stride=1, padding=0)
        )

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class WatermarkExtracter(nn.Module):

    def __init__(self, input_channel=1, hidden_dim=64, block="skip", n_layers=6):
        super(WatermarkExtracter, self).__init__()
        if block == "skip":
            core = SkipGatedBlock
        elif block == "relu":
            core = ReluBlock
        else:
            raise ValueError(f"Invalid block type: {block}")
        layers = [
            core(
                c_in=input_channel, c_out=hidden_dim, kernel_size=3, stride=1, padding=1
            )
        ]

        for i in range(n_layers - 2):
            layers.append(
                core(
                    c_in=hidden_dim,
                    c_out=hidden_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )

        layers.append(
            core(c_in=hidden_dim, c_out=1, kernel_size=3, stride=1, padding=1)
        )

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)
