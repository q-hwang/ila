import torch
from torch.autograd import Variable
import torchvision
import torch.nn as nn
import numpy as np

def ifgsm(model, X, y, niters=10, epsilon=0.01, visualize=False, learning_rate=0.005):
    X_pert = X.clone()
    X_pert.requires_grad = True
    
    for _ in range(niters):
        output_perturbed = model(X_pert)
        loss = nn.CrossEntropyLoss()(output_perturbed, y)
        loss.backward()
        pert = learning_rate * X_pert.grad.detach().sign()

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


"""4 more attacks: pgd, carnili"""
def pgd_new(model, X, y, epsilon = 0.03, niters=10, learning_rate=0.01): 
    out = model(X)
    ce = nn.CrossEntropyLoss()(out, y)
    err = (out.data.max(1)[1] != y.data).float().sum()  / X.size(0)

    X_pgd = Variable(X.data, requires_grad=True)
    for i in range(niters): 
        opt = torch.optim.Adam([X_pgd], lr=1e-3)
        opt.zero_grad()
        loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = learning_rate*X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        
        # adjust to be within [-epsilon, epsilon]
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        
    err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum() / X.size(0)
    return X_pgd


def pgd_ifgsm(model, X, y, niters=10, epsilon=0.01, visualize=False, learning_rate=0.005):
    out = model(X)
 
    X_pert = X.clone()
    X_pert.requires_grad = True
     
    for i in range(niters):
        output_perturbed = model(X_pert)
        loss = nn.CrossEntropyLoss()(output_perturbed, y)
        loss.backward()
        # PGD: project the gradiant to l-inf norm ball.
        pert = learning_rate / np.abs(X_pert.grad).cuda().max() * X_pert.grad
         
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



def fgsm(model, X, y, epsilon=0.01):
    out = model(X)
    error_original = (out.max(1)[1] != y).float().sum() / X.size(0)

    X_pert = X.clone()
    X_pert.requires_grad = True
    output_perturbed = model(X_pert)
    loss = nn.CrossEntropyLoss()(output_perturbed, y)
    loss.backward()
    
    pert = epsilon * X_pert.grad.detach().sign()
    X_pert = X_pert.detach() + pert
    X_pert = X_pert.detach().clamp(-1, 1)
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
        X_pert = (X_pert.detach() - X.clone()).clamp(-epsilon, epsilon) + X.clone()
        X_pert.requires_grad = True
        
        # adjust to be within [-1, 1]
        X_pert = X_pert.detach().clamp(-1, 1)
        X_pert.requires_grad = True
        
    return X_pert


# def lbfgs(model, x, y, lbfgs_max_iter, clip_min=-1, clip_max=1, binary_search_epsilon=1e-5):
#     # use fgsm first to find the easiest target class (why???)
#     adv_img = fgsm(model, x, y)
#     if model(adv_img).max(1)[1] == y:
#         print('L-BFGS failed to determine target class by FGSM. Fall back to random target.')
#         target_class = np.random.randint(0,10)
#     else:
#         target_class = model(adv_img).max(1)[1]
        
#     x_pert = x.clone()
#     x_pert.requires_grad
#     optimizer = torch.optim.LBFGS([x_pert])
#     for i in range(15):
#         print('STEP: ', i)
#         def closure():
#             x_pert = x_pert.detach().clamp(-1, 1)
#             x_pert.requires_grad = True
#             optimizer.zero_grad()
#             output_perturbed = model(x_pert)
#             dist_loss = nn.MSELoss()(x_pert-x)
#             ce_loss = nn.CrossEntropyLoss()(output_perturbed, y)
#             loss = dist_loss + c*ce_loss
#             loss.backward()
#             return loss
#         optimizer.step(closure)
#     return x_pert

#     x_pert = x.clone()
#     x_pert.requires_grad
#     optimizer = torch.optim.LBFGS([x_pert])
#     for i in range(15):
#         print('STEP: ', i)
#         def closure():
#             x_pert = x_pert.detach().clamp(-1, 1)
#             x_pert.requires_grad = True
#             optimizer.zero_grad()
#             output_perturbed = model(x_pert)
#             dist_loss = nn.MSELoss()(x_pert, x)
#             ce_loss = nn.CrossEntropyLoss()(output_perturbed, y)
#             loss = dist_loss + ce_loss
#             loss.backward()
#             return loss
#         optimizer.step(closure)
#     return x_pert
            
#    def optimize(model, x, target_class, lbfgs_max_iter, epsilon):
#         def loss(x, c):
#             output_perturbed = model(x_pert) 
#             dist_loss = nn.MSELoss()(output_perturbed-x_pert)
#             ce_loss = nn.CrossEntropyLoss()(output_perturbed, y)
#             loss = dist_loss + c*ce_loss
    
    
def deepfool_single(net, image, num_classes=10, overshoot=0.02, max_iter=20):
    """Choose the easiest target class"""
    import copy
    from torch.autograd.gradcheck import zero_gradients
    from torch.autograd import Variable
    
    f_image = net.forward(Variable(image[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()
    I = (np.array(f_image)).flatten().argsort()[::-1]

    I = I[0:num_classes]
    label = I[0]

    input_shape = image.cpu().numpy().shape
    pert_image = copy.deepcopy(image)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    x = Variable(pert_image[None, :], requires_grad=True)
    fs = net.forward(x)
    fs_list = [fs[0,I[k]] for k in range(num_classes)]
    k_i = label

    while k_i == label and loop_i < max_iter:

        pert = np.inf
        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.cpu().numpy().copy()

        for k in range(1, num_classes):
            zero_gradients(x)

            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = x.grad.data.cpu().numpy().copy()

            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()

            pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_i =  (pert+1e-4) * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)

        pert_image = image + (1+overshoot)*torch.from_numpy(r_tot).cuda()
        
        x = Variable(pert_image, requires_grad=True)
        fs = net.forward(x)
        k_i = np.argmax(fs.data.cpu().numpy().flatten())

        loop_i += 1

    r_tot = (1+overshoot)*r_tot
    return pert_image

def deepfool(model, images, labels, num_classes=10,niters=5):
    r = torch.zeros(images.size()).cuda()
    for i in range(images.size(0)):
        r[i] = deepfool_single(model, images[i], num_classes, max_iter=niters)
    return r

