import torch
import torchvision
import numpy as np
import torch.nn as nn
from attacks import ifgsm

# square sum of dot product
class Mid_layer_Loss(torch.nn.Module):

    def __init__(self):
        super(Mid_layer_Loss,self).__init__()

    def forward(self,x,y):
        x = x.view(x.shape[0],-1)
        y = y.view(y.shape[0],-1)
        x_norm = x / x.norm(dim=1)[:, None]
        y_norm = y / y.norm(dim=1)[:, None]
        loss = torch.mm(x_norm, y_norm.transpose(0,1))
        return pow(loss, 2).sum()

def mid_attack_generate(model, X, y, k, niters=10, epsilon=0.01, visualize=False):
    out = model(X)

    X_pert = X.clone()
    X_pert.requires_grad = True

    for i in range(niters):
        output_perturbed = model(X_pert)

        # generate adversarial example by max middle layer pertubation in the direction of increasing loss
        loss = nn.CrossEntropyLoss()(output_perturbed, y)
        model.features[k].retain_grad()
        loss.backward(retain_graph=True)

        pert_mid = model.features[k].grad.detach().sign()
        loss = Mid_layer_Loss()(model.features[k], model.features[k].data + 2 * epsilon * pert_mid)
        X_pert.grad.zero_()
        X_pert.requires_grad = True
        loss.backward()
        pert =  X_pert.grad.detach()

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

# square sum of dot product
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
        print(proj_loss)
        #print(x_norm.size(), x_norm)
        return proj_loss

# square sum of dot product
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
def mid_attack_adjust_generate(with_projection, model, X, X_attack, y, feature_layer, niters=10, epsilon=0.01, coeff=1.0, learning_rate=1):

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
        X_pert = (X_pert.detach() - X.clone()).clamp(-epsilon, epsilon) + X.clone()
        X_pert.requires_grad = True

        # adjust to be within [-1, 1]
        X_pert = X_pert.detach().clamp(-1, 1)
        X_pert.requires_grad = True


    h.remove()
    return X_pert

def generate_adjusted_samples(model, testloader, layer, coeff=1.0):
    samples, total_error_original, total_error_ifgsm, total_error_attack = [],[], [], []
    for i, data in enumerate(testloader, 0):
        images, labels = data
        images, labels = images.cuda(), labels.cuda()

        err_o = (model(images).max(1)[1] != labels).float().sum() / images.size(0)
        adv = ifgsm(model, images, labels, niters=10, epsilon=0.03)
        err_ifgsm = (model(adv).max(1)[1] != labels).float().sum() / images.size(0)

        sample = mid_attack_adjust_generate(model, images, adv, labels, layer, niters=10, epsilon=0.03, coeff= coeff)
        samples.append([sample, images,labels, adv])
        err_p = (model(sample).max(1)[1] != labels).float().sum() / images.size(0)
        total_error_ifgsm.append(err_ifgsm.float())
        total_error_original.append(err_o.float())
        total_error_attack.append(err_p.float())

        if i > 1000:
            break
    return samples, np.array(total_error_original).sum() / len(total_error_original), np.array(total_error_ifgsm).sum() / len(total_error_ifgsm), np.array(total_error_attack).sum()/ len(total_error_attack)


def mid_sample_attack_generate(model, X, y, k, niters=10, epsilon=0.01, visualize=False):

    X_pert = X.clone()

    for i in range(niters):
        output_perturbed = model(X_pert)
        X_pert.requires_grad = True
        # generate adversarial example by max middle layer pertubation in the direction of increasing loss
        loss = nn.CrossEntropyLoss()(output_perturbed, y)
        model.features[k].retain_grad()
        loss.backward(retain_graph=True)
        mean_direction = model.features[k].grad.clone().sign()
        mean = mean_direction.view(-1)

        # TODO: does not work
        pert_mid = np.random.multivariate_normal(mean, np.eye(mean_direction.shape[1])).view(mean_direction.shape)

        loss = Mid_layer_Loss()(model.features[k], model.features[k].data + epsilon * pert_mid)
        X_pert.requires_grad = True
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

def generate_samples(model, method, testloader, k):
    samples, total_error_original, total_error_attack = [], [], []
    for i, data in enumerate(testloader, 0):
        images, labels = data
        images, labels = images.cuda(), labels.cuda()

        err_o = (model(images).max(1)[1] != labels).float().sum() / images.size(0)
        sample = method(model, images, labels, k, niters=10, epsilon=0.03)
        samples.append([sample, images,labels])
        err_p = (model(sample).max(1)[1] != labels).float().sum() / images.size(0)
        total_error_original.append(err_o.float())
        total_error_attack.append(err_p.float())

        if i > 1000:
            break
    return samples, np.array(total_error_original).sum() / len(total_error_original), np.array(total_error_attack).sum()/ len(total_error_attack)


def generate_middle_patch(model, X, y, k, niters=10, epsilon=0.01, visualize=False):
    out = model(X)

    mid_pert = torch.new_zeros(model.features[k].shape)
    X.requires_grad = True

    for i in range(niters):
        output_perturbed = model(X_pert)

        loss = nn.CrossEntropyLoss()(output_perturbed, y)
        model.features[k].retain_grad()
        loss.backward()


        # add perturbation
        mid_pert += epsilon* model.features[k].grad.detach().sign()

        # make sure we don't modify the original image beyond epsilon
        mid_pert = mid_pert.clamp(-epsilon, epsilon)

        # adjust to be within [-1, 1]
        mid_pert = mid_pert.clamp(-1, 1)

        X_pert.requires_grad = True
    return mid_pert

def generate_patches(model, testloader, k):
    patches =[]
    for i, data in enumerate(testloader, 0):
        images, labels = data
        images, labels = images.cuda(), labels.cuda()

        err_o = (model(images).max(1)[1] != labels).float().sum() / images.size(0)
        patch = generate_middle_patch(model, images, labels, k, niters=10, epsilon=0.03)
        patches.append([patch, images,labels])

        if i > 1000:
            break
    return patches
