import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

class MnistNet(nn.Module):
    def __init__(self, n_channels=1):
        super(MnistNet, self).__init__()
        self.features = []
        self.initial = None
        self.classifier = []
        self.layers = collections.OrderedDict()

        self.conv1 = nn.Conv2d(in_channels=n_channels, out_channels=8, kernel_size=5)
        self.features.append(self.conv1) #0
        self.layers["conv1"] = self.conv1

        self.ReLU1 = nn.ReLU(False)
        self.features.append(self.ReLU1) #1
        self.layers["ReLU1"] = self.ReLU1

        self.pool1 = nn.MaxPool2d(2, 2)
        self.features.append(self.pool1) #2
        self.layers["pool1"] = self.pool1

        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5)
        self.features.append(self.conv2) #3
        self.layers["conv2"] = self.conv2

        self.ReLU2 = nn.ReLU(False)
        self.features.append(self.ReLU2) #4
        self.layers["ReLU2"] = self.ReLU2

        self.pool2 = nn.MaxPool2d(2, 2)
        self.features.append(self.pool2) #5
        self.layers["pool2"] = self.pool2

        self.feature_dims = 16 * 4 * 4
        self.fc1 = nn.Linear(self.feature_dims, 120) #start=6
        self.classifier.append(self.fc1)
        self.layers["fc1"] = self.fc1

        self.fc1act = nn.ReLU(False)
        self.classifier.append(self.fc1act)
        self.layers["fc1act"] = self.fc1act

        self.fc2 = nn.Linear(120, 84)
        self.classifier.append(self.fc2)
        self.layers["fc2"] = self.fc2

        self.fc2act = nn.ReLU(False)
        self.classifier.append(self.fc2act)
        self.layers["fc2act"] = self.fc2act

        self.fc3 = nn.Linear(84, 10)
        self.classifier.append(self.fc3)
        self.layers["fc3"] = self.fc3

        self.initial_params = [
            param.clone().detach().data for param in self.parameters()
        ]

    def forward(self, x, start=0, end=10):
        if start <= 5:  # start in self.features
            for idx, layer in enumerate(self.features[start:]):
                x = layer(x)
                if idx == end:
                    return x
            x = x.view(-1, self.feature_dims)
            for idx, layer in enumerate(self.classifier):
                x = layer(x)
                if idx + 6 == end:
                    return x
        else:
            if start == 6:
                x = x.view(-1, self.feature_dims)
            for idx, layer in enumerate(self.classifier):
                if idx >= start - 6:
                    x = layer(x)
                if idx + 6 == end:
                    return x

    def get_params(self, end=10):
        params = []
        for layer in list(self.layers.values())[: end + 1]:
            params += list(layer.parameters())
        return params

    def restore_initial_params(self):
        for param, initial in zip(self.parameters(), self.initial_params):
            param.data = initial.requires_grad_(True)


class CifarNet(nn.Module):
    def __init__(self, n_channels=3):
        super(CifarNet, self).__init__()
        self.features = []
        self.initial = None
        self.classifier = []
        self.layers = collections.OrderedDict()

        self.conv11 = nn.Conv2d(
            in_channels=n_channels, out_channels=64, kernel_size=3, padding=1
        )
        self.features.append(self.conv11)
        self.layers["conv11"] = self.conv11

        self.ReLU11 = nn.ReLU(True)
        self.features.append(self.ReLU11)
        self.layers["ReLU11"] = self.ReLU11

        self.conv12 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1
        )
        self.features.append(self.conv12)
        self.layers["conv12"] = self.conv12

        self.ReLU12 = nn.ReLU(True)
        self.features.append(self.ReLU12)
        self.layers["ReLU12"] = self.ReLU12

        self.pool1 = nn.MaxPool2d(2, 2)
        self.features.append(self.pool1)
        self.layers["pool1"] = self.pool1

        self.conv21 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1
        )
        self.features.append(self.conv21)
        self.layers["conv21"] = self.conv21

        self.ReLU21 = nn.ReLU(True)
        self.features.append(self.ReLU21)
        self.layers["ReLU21"] = self.ReLU21

        self.conv22 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, padding=1
        )
        self.features.append(self.conv22)
        self.layers["conv22"] = self.conv22

        self.ReLU22 = nn.ReLU(True)
        self.features.append(self.ReLU22)
        self.layers["ReLU22"] = self.ReLU22

        self.pool2 = nn.MaxPool2d(2, 2)
        self.features.append(self.pool2)
        self.layers["pool2"] = self.pool2

        self.conv31 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, padding=1
        )
        self.features.append(self.conv31)
        self.layers["conv31"] = self.conv31

        self.ReLU31 = nn.ReLU(True)
        self.features.append(self.ReLU31)
        self.layers["ReLU31"] = self.ReLU31

        self.conv32 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, padding=1
        )
        self.features.append(self.conv32)
        self.layers["conv32"] = self.conv32

        self.ReLU32 = nn.ReLU(True)
        self.features.append(self.ReLU32)
        self.layers["ReLU32"] = self.ReLU32

        self.pool3 = nn.MaxPool2d(2, 2)
        self.features.append(self.pool3)
        self.layers["pool3"] = self.pool3

        self.feature_dims = 4 * 4 * 128
        self.fc1 = nn.Linear(self.feature_dims, 512)
        self.classifier.append(self.fc1)
        self.layers["fc1"] = self.fc1

        self.fc1act = nn.Sigmoid()
        self.classifier.append(self.fc1act)
        self.layers["fc1act"] = self.fc1act

        self.fc2 = nn.Linear(512, 10)
        self.classifier.append(self.fc2)
        self.layers["fc2"] = self.fc2

        self.initial_params = [param.data for param in self.parameters()]

    def forward(self, x, start=0, end=17):
        if start <= len(self.features) - 1:  # start in self.features
            for idx, layer in enumerate(self.features[start:]):
                x = layer(x)
                if idx == end:
                    return x
            x = x.view(-1, self.feature_dims)
            for idx, layer in enumerate(self.classifier):
                x = layer(x)
                if idx + 15 == end:
                    return x
        else:
            if start == 15:
                x = x.view(-1, self.feature_dims)
            for idx, layer in enumerate(self.classifier):
                if idx >= start - 15:
                    x = layer(x)
                if idx + 15 == end:
                    return x

    def get_params(self, end=17):
        params = []
        for layer in list(self.layers.values())[: end + 1]:
            params += list(layer.parameters())
        return params

    def restore_initial_params(self):
        for param, initial in zip(self.parameters(), self.initial_params):
            param.data = initial

class MLP(nn.Module):
    def __init__(self, input_dim=28*28, hidden_dim=1024, n_layers=4):
        super().__init__()
        self.layers = nn.ModuleList([nn.Flatten(), nn.Linear(input_dim, hidden_dim), nn.ReLU()] + 
            [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
             nn.Linear(hidden_dim, hidden_dim), nn.ReLU()] + 
            [nn.Linear(hidden_dim, 10)]
        )
    
    def forward(self, x, start=0, end=7):
        for i, layer in enumerate(self.layers):
            if start <= i <= end:
                x = layer(x)
        return x

class CNN(nn.Module):
    def __init__(self, n_channels=1):
        super(CNN, self).__init__()
        self.features = []
        self.initial = None
        self.classifier = []
        self.layers = collections.OrderedDict()

        self.conv1 = nn.Conv2d(
            in_channels=n_channels,
            out_channels=8,
            kernel_size=5
        )
        self.features.append(self.conv1)
        self.layers['conv1'] = self.conv1

        self.ReLU1 = nn.ReLU(False)
        self.features.append(self.ReLU1)
        self.layers['ReLU1'] = self.ReLU1

        self.pool1 = nn.MaxPool2d(2, 2)
        self.features.append(self.pool1)
        self.layers['pool1'] = self.pool1

        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=5
        )
        self.features.append(self.conv2)
        self.layers['conv2'] = self.conv2

        self.ReLU2 = nn.ReLU(False)
        self.features.append(self.ReLU2)
        self.layers['ReLU2'] = self.ReLU2

        self.pool2 = nn.MaxPool2d(2, 2)
        self.features.append(self.pool2)
        self.layers['pool2'] = self.pool2

        self.feature_dims = 16 * 4 * 4
        self.fc1 = nn.Linear(self.feature_dims, 120)
        self.classifier.append(self.fc1)
        self.layers['fc1'] = self.fc1
     
        self.fc1act = nn.ReLU(False)
        self.classifier.append(self.fc1act)
        self.layers['fc1act'] = self.fc1act
     
        self.fc2 = nn.Linear(120, 84)
        self.classifier.append(self.fc2)
        self.layers['fc2'] = self.fc2
     
        self.fc2act = nn.ReLU(False)
        self.classifier.append(self.fc2act)
        self.layers['fc2act'] = self.fc2act
     
        self.fc3 = nn.Linear(84, 10)
        self.classifier.append(self.fc3)
        self.layers['fc3'] = self.fc3
        
        self.initial_params = [param.clone().detach().data for param in self.parameters()]

    def forward(self, x, start=0, end=10):
        if start <= 5: # start in self.features
            for idx, layer in enumerate(self.features[start:]):
                x = layer(x)
                if idx == end:
                    return x
            x = x.view(-1, self.feature_dims)
            for idx, layer in enumerate(self.classifier):
                x = layer(x)
                if idx + 6 == end:
                    return x
        else:
            if start == 6:
                x = x.view(-1, self.feature_dims)
            for idx, layer in enumerate(self.classifier):
                if idx >= start - 6:
                    x = layer(x)
                if idx + 6 == end:
                    return x
                
    def get_params(self, end=10):
        params = []
        for layer in list(self.layers.values())[:end+1]:
            params += list(layer.parameters())
        return params

    def restore_initial_params(self):
        for param, initial in zip(self.parameters(), self.initial_params):
            param.data = initial.requires_grad_(True)

class MLPMixerClient(nn.Module):
    def __init__(self, in_channels=3, img_size=32, patch_size=4, hidden_size=128, hidden_s=64, hidden_c=512, cut_layer=1, drop_p=0., off_act=False, is_cls_token=False):
        super(MLPMixerClient, self).__init__()
        num_patches = (img_size // patch_size) ** 2
        self.is_cls_token = is_cls_token

        self.patch_emb = nn.Sequential(
            nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b d h w -> b (h w) d')
        )

        if self.is_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
            num_patches += 1

        self.cut_layer = cut_layer
        self.mixer_layers = nn.ModuleList(
            [MixerLayer(num_patches, hidden_size, hidden_s, hidden_c, drop_p, off_act) for _ in range(cut_layer)]
        )

    def forward(self, x):
        out = self.patch_emb(x)
        if self.is_cls_token:
            out = torch.cat([self.cls_token.repeat(out.size(0), 1, 1), out], dim=1)
        for layer in self.mixer_layers:
            out = layer(out)
        return out

class MLPMixerServer(nn.Module):
    def __init__(self, num_patches, hidden_size=128, hidden_s=64, hidden_c=512, cut_layer=1, num_layers=8, num_classes=10, drop_p=0., off_act=False, is_cls_token=False):
        super(MLPMixerServer, self).__init__()
        self.is_cls_token = is_cls_token

        self.mixer_layers = nn.ModuleList(
            [MixerLayer(num_patches, hidden_size, hidden_s, hidden_c, drop_p, off_act) for _ in range(cut_layer, num_layers)]
        )
        self.ln = nn.LayerNorm(hidden_size)
        self.clf = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        for layer in self.mixer_layers:
            x = layer(x)
        x = self.ln(x)
        x = x[:, 0] if self.is_cls_token else x.mean(dim=1)
        x = self.clf(x)
        return x

class MixerLayer(nn.Module):
    def __init__(self, num_patches, hidden_size, hidden_s, hidden_c, drop_p, off_act):
        super(MixerLayer, self).__init__()
        self.mlp1 = MLP1(num_patches, hidden_s, hidden_size, drop_p, off_act)
        self.mlp2 = MLP2(hidden_size, hidden_c, drop_p, off_act)

    def forward(self, x):
        out = self.mlp1(x)
        out = self.mlp2(out)
        return out

class MLP1(nn.Module):
    def __init__(self, num_patches, hidden_s, hidden_size, drop_p, off_act):
        super(MLP1, self).__init__()
        self.ln = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Conv1d(num_patches, hidden_s, kernel_size=1)
        self.do1 = nn.Dropout(p=drop_p)
        self.fc2 = nn.Conv1d(hidden_s, num_patches, kernel_size=1)
        self.do2 = nn.Dropout(p=drop_p)
        self.act = F.gelu if not off_act else lambda x: x

    def forward(self, x):
        out = self.do1(self.act(self.fc1(self.ln(x))))
        out = self.do2(self.fc2(out))
        return out + x

class MLP2(nn.Module):
    def __init__(self, hidden_size, hidden_c, drop_p, off_act):
        super(MLP2, self).__init__()
        self.ln = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_c)
        self.do1 = nn.Dropout(p=drop_p)
        self.fc2 = nn.Linear(hidden_c, hidden_size)
        self.do2 = nn.Dropout(p=drop_p)
        self.act = F.gelu if not off_act else lambda x: x

    def forward(self, x):
        out = self.do1(self.act(self.fc1(self.ln(x))))
        out = self.do2(self.fc2(out))
        return out + x
