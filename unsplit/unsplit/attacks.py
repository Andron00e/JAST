import numpy as np
import torch
from torchvision import transforms, datasets

from .util import *

from tqdm import tqdm

def model_inversion_stealing(clone_model, split_layer, target, input_size, 
                            lambda_tv=0.1, lambda_l2=1, main_iters=1000, input_iters=100, model_iters=100, device='cuda:0'):
    x_pred = torch.empty(input_size, device=device).fill_(0.5).requires_grad_(True)
    target = target.to(device)
    clone_model = clone_model.to(device)
    input_opt = torch.optim.Adam([x_pred], lr=0.001, amsgrad=True)
    model_opt = torch.optim.Adam(clone_model.parameters(), lr=0.001, amsgrad=True)
    mse = torch.nn.MSELoss()
    loss_arr = []

    for main_iter in tqdm(range(main_iters)):
        for input_iter in range(input_iters):
            input_opt.zero_grad()
            pred = clone_model(x_pred, end=split_layer)
            loss = mse(pred, target)
            loss_arr.append(loss.item())
            loss += lambda_tv*TV(x_pred) + lambda_l2*l2loss(x_pred)
            loss.backward()
            input_opt.step()
        for model_iter in range(model_iters):
            model_opt.zero_grad()
            pred = clone_model(x_pred, end=split_layer)
            loss = mse(pred, target) 
            loss.backward()
            model_opt.step()
            loss_arr.append(loss.item())

    return x_pred.detach(), loss_arr