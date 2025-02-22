import torch
import torch.nn as nn

__all__ = [
    "ResNet18",
    "MiniResNet",
    "ResNet18H2",
    "MiniResNet2H",
    "ResNet3D18",
]


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, activation="ReLU", bias=False, ):
        super(ResidualBlock, self).__init__()

        self.conv1_bn = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias),
            nn.BatchNorm2d(out_channels),
        )
        self.conv2_bn = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.BatchNorm2d(out_channels),
        )
        self.act = getattr(nn, activation)()
        if in_channels == out_channels:  # keep the same size
            self.downsample = None
        else:  # if in_channels != out_channels, we need to downsample
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.conv1_bn(x)
        out = self.act(out)
        out = self.conv2_bn(out)
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        out = self.act(out)  # ReLU, etc.
        return out


class ResNet18(nn.Module):
    # reference: https://pytorch.org/vision/main/_modules/torchvision/models/resnet.html#resnet18
    def __init__(self, block=ResidualBlock, activation="ReLU", in_channels=3, out_features=1):
        super(ResNet18, self).__init__()

        self.activation = activation
        self.conv_bn_act_pool = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False, ),
            nn.BatchNorm2d(64),
            getattr(nn, activation)(),  # ReLU, etc.
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self.make_layer(block, 2, in_channels=64, out_channels=64, stride=1)
        self.layer2 = self.make_layer(block, 2, in_channels=64, out_channels=128, stride=2)
        self.layer3 = self.make_layer(block, 2, in_channels=128, out_channels=256, stride=2)
        self.layer4 = self.make_layer(block, 2, in_channels=256, out_channels=512, stride=2)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=out_features),
        )

    def make_layer(self, block, layer_nums, in_channels, out_channels, stride):
        # for the first block, if in_channels != out_channels, we need to downsample, the stride is > 1.
        # for the rest blocks, the in_channels = out_channels, we keep the same output size, the stride is 1.
        block_params_list = [(in_channels, out_channels, stride)] + [(out_channels, out_channels, 1)] * (layer_nums - 1)

        layers = [block(in_channels=in_channels, out_channels=out_channels, stride=stride,
                        activation=self.activation, bias=False)
                  for in_channels, out_channels, stride in block_params_list]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_bn_act_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.head(x)
        return x


class MiniResNet(nn.Module):
    def __init__(self, block=ResidualBlock, activation="ReLU", in_channels=3, out_features=1):
        super(MiniResNet, self).__init__()

        self.activation = activation
        self.conv_bn_act_pool = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False, ),
            nn.BatchNorm2d(16),
            getattr(nn, activation)(),  # ReLU, etc.
        )

        self.layer1 = self.make_layer(block, 2, in_channels=16, out_channels=16, stride=1)
        self.layer2 = self.make_layer(block, 2, in_channels=16, out_channels=32, stride=2)
        self.layer3 = self.make_layer(block, 2, in_channels=32, out_channels=64, stride=2)

        self.head = nn.Sequential(
            nn.AvgPool2d(8),
            nn.Flatten(),
            nn.Linear(in_features=64, out_features=out_features),
        )

    def make_layer(self, block, layer_nums, in_channels, out_channels, stride):
        # for the first block, if in_channels != out_channels, we need to downsample, the stride is > 1.
        # for the rest blocks, the in_channels = out_channels, we keep the same output size, the stride is 1.
        block_params_list = [(in_channels, out_channels, stride)] + [(out_channels, out_channels, 1)] * (layer_nums - 1)

        layers = [block(in_channels=in_channels, out_channels=out_channels, stride=stride,
                        activation=self.activation, bias=False)
                  for in_channels, out_channels, stride in block_params_list]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_bn_act_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.head(x)
        return x


class ResNet18H2(nn.Module):
    def __init__(self, block=ResidualBlock, activation="ReLU", in_channels=3, out_features=1):
        super(ResNet18H2, self).__init__()

        self.activation = activation
        self.conv_bn_act_pool = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False, ),
            nn.BatchNorm2d(64),
            getattr(nn, activation)(),  # ReLU, etc.
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self.make_layer(block, 2, in_channels=64, out_channels=64, stride=1)
        self.layer2 = self.make_layer(block, 2, in_channels=64, out_channels=128, stride=2)
        self.layer3 = self.make_layer(block, 2, in_channels=128, out_channels=256, stride=2)
        self.layer4 = self.make_layer(block, 2, in_channels=256, out_channels=512, stride=2)

        self.head0 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=out_features),
        )

        self.head1 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=out_features),
        )

    def make_layer(self, block, layer_nums, in_channels, out_channels, stride):
        # for the first block, if in_channels != out_channels, we need to downsample, the stride is > 1.
        # for the rest blocks, the in_channels = out_channels, we keep the same output size, the stride is 1.
        block_params_list = [(in_channels, out_channels, stride)] + [(out_channels, out_channels, 1)] * (layer_nums - 1)

        layers = [block(in_channels=in_channels, out_channels=out_channels, stride=stride,
                        activation=self.activation, bias=False)
                  for in_channels, out_channels, stride in block_params_list]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_bn_act_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x0 = self.head0(x)
        x1 = self.head1(x)
        return x0, x1


class MiniResNet2H(nn.Module):
    def __init__(self, block=ResidualBlock, activation="ReLU", in_channels=3, out_features=1):
        super(MiniResNet2H, self).__init__()

        self.activation = activation
        self.conv_bn_act_pool = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False, ),
            nn.BatchNorm2d(16),
            getattr(nn, activation)(),  # ReLU, etc.
        )

        self.layer1 = self.make_layer(block, 2, in_channels=16, out_channels=16, stride=1)
        self.layer2 = self.make_layer(block, 2, in_channels=16, out_channels=32, stride=2)
        self.layer3 = self.make_layer(block, 2, in_channels=32, out_channels=64, stride=2)

        self.head0 = nn.Sequential(
            nn.AvgPool2d(8),
            nn.Flatten(),
            nn.Linear(in_features=64, out_features=out_features),
        )

        self.head1 = nn.Sequential(
            nn.AvgPool2d(8),
            nn.Flatten(),
            nn.Linear(in_features=64, out_features=out_features),
        )

    def make_layer(self, block, layer_nums, in_channels, out_channels, stride):
        # for the first block, if in_channels != out_channels, we need to downsample, the stride is > 1.
        # for the rest blocks, the in_channels = out_channels, we keep the same output size, the stride is 1.
        block_params_list = [(in_channels, out_channels, stride)] + [(out_channels, out_channels, 1)] * (layer_nums - 1)

        layers = [block(in_channels=in_channels, out_channels=out_channels, stride=stride,
                        activation=self.activation, bias=False)
                  for in_channels, out_channels, stride in block_params_list]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_bn_act_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x0 = self.head0(x)
        x1 = self.head1(x)
        return x0, x1


class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride, activation="ReLU", bias=False, ):
        super(ResidualBlock3D, self).__init__()

        self.conv1_bn = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=bias),
            nn.BatchNorm3d(out_channels),
        )
        self.conv2_bn = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.BatchNorm3d(out_channels),
        )
        self.act = getattr(nn, activation)()
        if in_channels == out_channels:  # keep the same size
            self.downsample = None
        else:  # if in_channels != out_channels, we need to downsample
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=bias),
                nn.BatchNorm3d(out_channels),
            )

    def forward(self, x):
        out = self.conv1_bn(x)
        out = self.conv2_bn(out)
        out = self.act(out)
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        out = self.act(out)  # ReLU, etc.
        return out


class ResNet3D18(nn.Module):
    def __init__(self, block=ResidualBlock3D, activation="ReLU", use_batch_norm=True, in_channels=3, out_features=1):
        super(ResNet3D18, self).__init__()

        self.activation = activation
        self.conv_bn_act_pool = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False, ),
            nn.BatchNorm2d(64) if use_batch_norm else nn.InstanceNorm3d(64),
            getattr(nn, activation)(),  # ReLU, etc.
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self.make_layer(block, 2, in_channels=64, out_channels=64, stride=1)
        self.layer2 = self.make_layer(block, 2, in_channels=64, out_channels=128, stride=2)
        self.layer3 = self.make_layer(block, 2, in_channels=128, out_channels=256, stride=2)
        self.layer4 = self.make_layer(block, 2, in_channels=256, out_channels=512, stride=2)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=out_features),
        )

    def make_layer(self, block, layer_nums, in_channels, out_channels, stride):
        # for the first block, if in_channels != out_channels, we need to downsample, the stride is > 1.
        # for the rest blocks, the in_channels = out_channels, we keep the same output size, the stride is 1.
        block_params_list = [(in_channels, out_channels, stride)] + [(out_channels, out_channels, 1)] * (layer_nums - 1)

        layers = [block(in_channels=in_channels, out_channels=out_channels, stride=stride,
                        activation=self.activation, bias=False)
                  for in_channels, out_channels, stride in block_params_list]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_bn_act_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.head(x)
        return x


class ResNet3D18H2(nn.Module):
    def __init__(self, block=ResidualBlock3D, activation="ReLU", use_batch_norm=True, in_channels=3, out_features=1):
        super(ResNet3D18H2, self).__init__()

        self.activation = activation
        self.conv_bn_act_pool = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False, ),
            nn.BatchNorm2d(64) if use_batch_norm else nn.InstanceNorm3d(64),
            getattr(nn, activation)(),  # ReLU, etc.
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self.make_layer(block, 2, in_channels=64, out_channels=64, stride=1)
        self.layer2 = self.make_layer(block, 2, in_channels=64, out_channels=128, stride=2)
        self.layer3 = self.make_layer(block, 2, in_channels=128, out_channels=256, stride=2)
        self.layer4 = self.make_layer(block, 2, in_channels=256, out_channels=512, stride=2)

        self.head0 = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(512, 128),  # First hidden layer
            getattr(nn, activation)(),
            nn.Linear(128, 32),  # Second hidden layer
            getattr(nn, activation)(),
            nn.Linear(32, 16),  # Third hidden layer
            getattr(nn, activation)(),
            nn.Linear(16, out_features),  # Output layer
        )

        self.head1 = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(512, 128),  # First hidden layer
            getattr(nn, activation)(),
            nn.Linear(128, 32),  # Second hidden layer
            getattr(nn, activation)(),
            nn.Linear(32, 16),  # Third hidden layer
            getattr(nn, activation)(),
            nn.Linear(16, out_features),  # Output layer
        )

    def make_layer(self, block, layer_nums, in_channels, out_channels, stride):
        # for the first block, if in_channels != out_channels, we need to downsample, the stride is > 1.
        # for the rest blocks, the in_channels = out_channels, we keep the same output size, the stride is 1.
        block_params_list = [(in_channels, out_channels, stride)] + [(out_channels, out_channels, 1)] * (layer_nums - 1)

        layers = [block(in_channels=in_channels, out_channels=out_channels, stride=stride,
                        activation=self.activation, bias=False)
                  for in_channels, out_channels, stride in block_params_list]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_bn_act_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.head(x)
        return x


if __name__ == "__main__":
    # Create a simple ResNet18 model
    seed = 42
    torch.manual_seed(seed)

    model = ResNet18()
    res_input = torch.randn(1, 3, 224, 224)
    output = model(res_input)
    print(output)
