from torchsummary import summary
from torch import nn

hidden_conv_channels = 32
hidden_linear_channels = 64
# get number of last frames from kwargs
out_shape = 3

net = nn.Sequential(
    nn.Conv2d(3, hidden_conv_channels, kernel_size=3, stride=2),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(hidden_conv_channels, hidden_conv_channels, kernel_size=3, stride=2),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(hidden_conv_channels, hidden_conv_channels, kernel_size=3, stride=2),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(hidden_conv_channels, hidden_linear_channels),
    nn.ReLU(),
    nn.Linear(hidden_linear_channels, hidden_linear_channels),
    nn.ReLU(),
    nn.Linear(hidden_linear_channels, hidden_linear_channels),
    nn.ReLU()
)

summary(net, (3, 64, 64), device='cpu')