import torch
import torchvision
import torch.nn as nn

def ifgsm(model, X, y, niters=10, epsilon=0.01, visualize=False):
    X_pert = X.clone()
    X_pert.requires_grad = True
    
    for _ in range(niters):
        output_perturbed = model(X_pert)
        loss = nn.CrossEntropyLoss()(output_perturbed, y)
        loss.backward()
        pert = epsilon * X_pert.grad.detach().sign()

        # add perturbation
        X_pert = X_pert.detach() + pert
        X_pert.requires_grad = True
        
        # make sure we don't modify the original image beyond epsilon
        X_pert = (X_pert.detach() - X.clone()).clamp(-epsilon, epsilon) + X.clone()
        X_pert.requires_grad = True
        
        # adjust to be within [-1, 1]
        X_pert = X_pert.detach().clamp(-1, 1)
        X_pert.requires_grad = True
        
    return X_pert
