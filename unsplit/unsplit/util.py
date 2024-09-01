import torch
from torchmetrics.image.fid import FrechetInceptionDistance
import torchvision
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt
import random

import warnings
    

def compute_fid(inversion_targets, results):
    fid = FrechetInceptionDistance(normalize=True)
    fid.update(torch.cat([x[:100] for x in inversion_targets], dim=0).repeat(1, 3, 1, 1).cpu(), real=True)
    fid.update(torch.cat([x[:100] for x in results], dim=0).repeat(1, 3, 1, 1).cpu(), real=False)
    score = fid.compute()
    return score, fid

# computes total variation for an image
def TV(x):
    batch_size = x.size()[0]
    h_x = x.size()[2]
    w_x = x.size()[3]
    count_h = _tensor_size(x[:, :, 1:, :])
    count_w = _tensor_size(x[:, :, :, 1:])
    h_tv = torch.pow(x[:, :, 1:, :] - x[:, :, :h_x - 1, :], 2).sum()
    w_tv = torch.pow(x[:, :, :, 1:] - x[:, :, :, :w_x - 1], 2).sum()
    return (h_tv / count_h + w_tv / count_w) / batch_size


def l2loss(x):
    return (x ** 2).mean()


def _tensor_size(t):
    return t.size()[1] * t.size()[2] * t.size()[3]


# get the first `count` examples of the `target` class from dataset
def get_examples_by_class(dataset, target, count=1):
    result = []
    for image, label in dataset:
        if label == target:
            result.append(image)
        if len(result) == count:
            break
    return torch.stack(result)


def normalize(result):
    min_v = result.flatten(1).min(dim=1).values.unsqueeze(-1)
    normalized = ((result.flatten(1) - min_v) / 
                  (result.flatten(1).max(dim=1).values.unsqueeze(-1) - min_v))
    return normalized.reshape(result.shape)


def get_test_acc(m1, m2, testloader, split=0, device='cuda:0'):
    res = 0
    with torch.no_grad():
        for image, label in testloader:
            #image, label = image.to(device), label.to(device)
            pred = m2(m1(image, end=split), start=split+1)
            res += (pred.argmax(dim=1) == label).sum()
    return (res / len(testloader.dataset)).item()

# get random examples from a dataset
def get_random_example(set, count=1, batch_size=1):
    indices = []
    for i in range(count):
        if i not in indices:
            indices.append(random.randrange(len(set)))
    subset = torch.utils.data.Subset(set, indices)
    subsetloader = torch.utils.data.DataLoader(
        subset, batch_size=batch_size, num_workers=0, shuffle=False
    )
    return subsetloader


def get_test_score(m1, m2, dataset, split=0):
    score = 0
    imageloader = get_random_example(dataset, count=2000)
    for image, label in imageloader:
        pred = m2(m1(image, end=split), start=split + 1)
        if torch.argmax(pred) == label.detach():
            score += 1
    return 100 * score / len(imageloader)

def display_imagelist(images):
    if images[0].shape[0] > 1:
        print(f"Given set of {images[0].shape[0]} " +
                      "images for each class, displaying only " +
                      "the first ones for each class.")
    height, width = images[0].shape[-2:]
    fig, ax = plt.subplots(1, len(images))
    for index, image in enumerate(images):
        ax[index].axis('off')
        ax[index].imshow(image[0].cpu().detach().reshape(height, width))
    plt.show()