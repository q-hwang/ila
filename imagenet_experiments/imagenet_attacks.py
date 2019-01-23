import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import functools
import itertools
import operator

def ifgsm(model, X, y, niters=10, epsilon=0.01, learning_rate=0.005):
    out = model(X)
    error_original = (out.max(1)[1] != y).float().sum() / X.size(0)

    X_pert = X.clone()
    X_pert.requires_grad = True
    
    for i in range(niters):
        output_perturbed = model(X_pert)
        loss = nn.CrossEntropyLoss()(output_perturbed, y)
        loss.backward()
        pert = learning_rate * X_pert.grad.detach().sign()
        
        # add perturbation
        X_pert = X_pert.detach() + pert
        X_pert.requires_grad = True
        
        # make sure we don't modify the original image beyond epsilon
        X_pert = normalize_and_scale(X_pert.detach() - X.clone(), epsilon) + X.clone()
        X_pert.requires_grad = True

        # clamp image
        X_pert = X_pert.detach().clamp(X.min(), X.max())
        X_pert.requires_grad = True

    return X_pert


def momentum_ifgsm(model, X, y, niters=10, epsilon=0.01, visualize=False, learning_rate=0.005, decay=0.9):
    X_pert = X.clone()
    X_pert.requires_grad = True
    
    momentum = 0
    for _ in range(niters):
        output_perturbed = model(X_pert)
        loss = nn.CrossEntropyLoss()(output_perturbed, y)
        loss.backward()
        
        momentum = decay * momentum + X_pert.grad / torch.sum(torch.abs(X_pert.grad))
        pert = learning_rate * momentum.sign()

        # add perturbation
        X_pert = X_pert.detach() + pert
        X_pert.requires_grad = True
        
        # make sure we don't modify the original image beyond epsilon
        X_pert = normalize_and_scale(X_pert.detach() - X.clone(), epsilon) + X.clone()
        X_pert.requires_grad = True

        # adjust to be within [-1, 1]
        X_pert = X_pert.detach().clamp(X.min(), X.max())
        X_pert.requires_grad = True
        
    return X_pert


class Proj_Loss(torch.nn.Module):

    def __init__(self):
        super(Proj_Loss,self).__init__()

    def forward(self, old_attack_mid , new_mid, original_mid, coeff):
        x = (old_attack_mid - original_mid).view(1,-1)
        y = (new_mid - original_mid).view(1,-1)
        #print(x.norm(), y.norm())
        x_norm = x / x.norm()
        y_norm = y / y.norm()
        #angle_loss = torch.mm(x_norm, y_norm.transpose(0,1))
        #magnitude_gain = y.norm() / x.norm()
        proj_loss = torch.mm(y, x_norm.transpose(0,1)) / x.norm()
#         print(proj_loss)
        #print(x_norm.size(), x_norm)
        return proj_loss


class Mid_layer_target_Loss(torch.nn.Module):

    def __init__(self):
        super(Mid_layer_target_Loss,self).__init__()

    def forward(self, old_attack_mid , new_mid, original_mid, coeff):
        x = (old_attack_mid - original_mid).view(1,-1)
        y = (new_mid - original_mid).view(1,-1)
        #print(y.norm())
        x_norm = x / x.norm()
        if (y == 0).all():
            y_norm = y
        else:
            y_norm = y / y.norm()
        angle_loss = torch.mm(x_norm, y_norm.transpose(0,1))
        magnitude_gain = y.norm() / x.norm()
#         print(str(angle_loss.float()) + " " + str(magnitude_gain.float()) )
        return angle_loss + magnitude_gain * coeff
 

"""Return: perturbed x"""
mid_output = None

def ILA(with_projection, model, X, X_attack, y, feature_layer, niters=10, epsilon=0.01, coeff=1.0, learning_rate=1):
    
    X = X.detach()
    X_pert = torch.zeros(X.size()).cuda()
    X_pert.copy_(X).detach()
    X_pert.requires_grad = True
    
    
    def get_mid_output(m, i, o):
        global mid_output
        mid_output = o
    
    h = feature_layer.register_forward_hook(get_mid_output)
    
    out = model(X)
    mid_original = torch.zeros(mid_output.size()).cuda() 
    mid_original.copy_(mid_output)
   
    out = model(X_attack)
    mid_attack_original = torch.zeros(mid_output.size()).cuda() 
    mid_attack_original.copy_(mid_output)
   
    
    for i in range(niters):          
        output_perturbed = model(X_pert)
        # generate adversarial example by max middle layer pertubation in the direction of increasing loss
        if with_projection:
            loss = Proj_Loss()(mid_attack_original.detach(), mid_output, mid_original.detach(),coeff)
        else:
            loss = Mid_layer_target_Loss()(mid_attack_original.detach(), mid_output, mid_original.detach(),coeff)
         
        loss.backward()
        pert = learning_rate * X_pert.grad.detach().sign()

        # minimize loss
        X_pert = X_pert.detach() + pert
        X_pert.requires_grad = True

         
        # make sure we don't modify the original image beyond epsilon
        X_pert = normalize_and_scale(X_pert.detach() - X.clone(), epsilon) + X.clone()
        X_pert.requires_grad = True

        # clamp image
        X_pert = X_pert.detach().clamp(X.min(), X.max())
        X_pert.requires_grad = True
    
      
    h.remove()
    return X_pert


batch_size=32
mean_arr, stddev_arr = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# normalization (L-inf norm projection) code for output delta
def normalize_and_scale(delta_im, epsilon):

    delta_im = delta_im / delta_im.abs().max() # now -1..1    
    delta_im = delta_im.clone() + 1 # now 0..2
    delta_im = delta_im.clone() * 0.5 # now 0..1

    # normalize image color channels
    for c in range(3):
        delta_im[:,c,:,:] = (delta_im[:,c,:,:].clone() - mean_arr[c]) / stddev_arr[c]
        
    # threshold each channel of each image in deltaIm according to inf norm
    # do on a per image basis as the inf norm of each image could be different
#     for i in range(batch_size):
    for i in range(delta_im.size(0)):
        # do per channel l_inf normalization
        for ci in range(3):
            l_inf_channel = delta_im[i,ci,:,:].clone().detach().abs().max()
            mag_in_scaled_c = epsilon/stddev_arr[ci]
            delta_im[i,ci,:,:] = delta_im[i,ci,:,:].clone() * np.minimum(1.0, mag_in_scaled_c / l_inf_channel.cpu().numpy())

    return delta_im
