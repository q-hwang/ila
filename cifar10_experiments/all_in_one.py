
import sys
from adversarial import mid_attack_adjust_generate  
from various_attacks import pgd_ifgsm, ifgsm, momentum_ifgsm, deepfool, pgd_new
from cifar10models import *
import numpy as np
import pandas as pd

def model_name(model):
    if model == ResNet18:
        return 'ResNet18'
    elif model == DenseNet121:
        return 'DenseNet121'
    elif model == GoogLeNet:
        return 'GoogLeNet'
    elif model == SENet18:
        return 'SENet18'
    assert False

def attack_name(attack):
    if attack == ifgsm:
        return 'ifgsm'
    elif attack == pgd_ifgsm:
        return 'pgd_ifgsm'
    elif attack == momentum_ifgsm:
        return 'momentum_ifgsm'
    elif attack == pgd_new:
        return 'pgd_new'
    elif attack == deepfool:
        return 'deepfool'
    assert False



def wrap_attack(attack, params):
    """
    A wrap for attack functions
    attack: an attack function
    params: kwargs for the attack func
    """
    def wrap_f(model, images, labels, niters):
        # attack should process by batch
        return attack(model, images, labels, niters=niters, **params)
    return wrap_f


def wrap_cw_l2(attack_class, params):
    """
    A wrap for attack class
    """
    def wrap_f(model, images, labels, niters):
        cw2 = attack_class(max_steps=niters, **params)
        cw2.num_classes = 10
        return torch.tensor(cw2.run(model, images, labels)).transpose(1,3).cuda()
    return wrap_f



ifgsm_params = {'learning_rate': 0.002, 'epsilon': 0.03}
momentum_ifgsm_params = {'learning_rate': 0.002, 'epsilon': 0.03, 'decay': 0.9}
cw2_params = {'targeted': False}
attacks = [(attack_name(ifgsm), wrap_attack(ifgsm, ifgsm_params)),
            (attack_name(momentum_ifgsm), wrap_attack(momentum_ifgsm, momentum_ifgsm_params))]


data_preprocess = (32, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

source_models = [(ResNet18, '/share/cuvl/weights/cifar10/resnet18_epoch_347_acc_94.77.pth'),
                 (DenseNet121,'/share/cuvl/weights/cifar10/densenet121_epoch_315_acc_95.61.pth'),
                 (GoogLeNet, '/share/cuvl/weights/cifar10/googlenet_epoch_227_acc_94.86.pth'),
                 (SENet18, '/share/cuvl/weights/cifar10/senet18_epoch_279_acc_94.59.pth')]

transfer_models = [(ResNet18, '/share/cuvl/weights/cifar10/resnet18_epoch_347_acc_94.77.pth'),
                   (DenseNet121,'/share/cuvl/weights/cifar10/densenet121_epoch_315_acc_95.61.pth'),
                   (GoogLeNet, '/share/cuvl/weights/cifar10/googlenet_epoch_227_acc_94.86.pth'),
                   (SENet18, '/share/cuvl/weights/cifar10/senet18_epoch_279_acc_94.59.pth')]

with_projection = True
fla_params = {attack_name(momentum_ifgsm): {'niters': 10, 'learning_rate': 0.006, 'epsilon':0.03, 'coeff': 5.0},
              attack_name(ifgsm): {'niters': 10, 'learning_rate': 0.006
                                   , 'epsilon':0.03, 'coeff': 0.8}}




source_layers = {model_name(ResNet18): list(enumerate(ResNet18()._modules.keys())),
                 model_name(DenseNet121): list(enumerate(DenseNet121()._modules.keys())),
                 model_name(GoogLeNet): list(enumerate(GoogLeNet()._modules.keys())),
                 model_name(SENet18):list(enumerate(SENet18()._modules.keys()))
                 }


def log(out_df, source_model, source_model_file, target_model, target_model_file, batch_index, layer_index, layer_name, fool_method, with_fla,  fool_rate, acc_after_attack, original_acc):
    return out_df.append({
        'source_model':model_name(source_model),
        'source_model_file': source_model_file,
        'target_model':model_name(target_model),
        'target_model_file': target_model_file,
        'batch_index':batch_index,
        'layer_index':layer_index,
        'layer_name':layer_name,
        'fool_method':fool_method,
        'with_fla':with_fla,
        'fool_rate':fool_rate,
        'acc_after_attack':acc_after_attack,
        'original_acc':original_acc},ignore_index=True)


def get_data(batch_size, mean, stddev):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, stddev)])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
    return trainloader, testloader

def get_fool_adv_orig(model, adversarial_xs, originals, labels):
    total = adversarial_xs.size(0)
    correct_orig = 0
    correct_adv = 0
    fooled = 0

    advs, ims, lbls = adversarial_xs.cuda(), originals.cuda(), labels.cuda()
#     print('adv', advs.shape)
    outputs_adv = model(advs)
    outputs_orig = model(ims)
    _, predicted_adv = torch.max(outputs_adv.data, 1)
    _, predicted_orig = torch.max(outputs_orig.data, 1)

    correct_adv += (predicted_adv == lbls).sum()
    correct_orig += (predicted_orig == lbls).sum()
    fooled += (predicted_adv != predicted_orig).sum()
    return [100.0 * float(fooled.item())/total, 100.0 * float(correct_adv.item())/total, 100.0 * float(correct_orig.item())/total]


def test_adv_examples_across_models(transfer_models, adversarial_xs, originals, labels):
    accum = []
    for (network, weights_path) in transfer_models:
        net = network().cuda()
        net.load_state_dict(torch.load(weights_path, map_location=lambda storage, loc: storage))
        net.eval()
        res = get_fool_adv_orig(net, adversarial_xs, originals, labels)
        res.append(weights_path)
        accum.append(res)
    return accum


def complete_loop(sample_num, out_df):
    trainloader, testloader = get_data(*data_preprocess)
    for model_class, source_weight_path in source_models:
        model = model_class().cuda()
        model.load_state_dict(torch.load(source_weight_path))
        model.eval()
        dic = model._modules
        for attack_name, attack in attacks:
            print('using source model {0} attack {1}'.format(model_name(model_class), attack_name))
            for batch_i, data in enumerate(testloader, 0):
                if batch_i%10 == 0:
                    print("batch" , batch_i)
                if batch_i == sample_num:
                    break
                images, labels = data
                images, labels = images.cuda(), labels.cuda()

                # baseline
                adversarial_xs = attack(model, images, labels, niters=20)

                transfer_list = test_adv_examples_across_models(transfer_models, adversarial_xs, images, labels)
                for i, (target_fool_rate, target_acc_attack, target_acc_original, target_weight_path) in enumerate(transfer_list):
                    out_df = log(out_df,model_class, source_weight_path,transfer_models[i][0],
                                 target_weight_path, batch_i, np.nan, "", attack_name, False,
                                 target_fool_rate, target_acc_attack, target_acc_original)

                 # ILA
                fla_input_xs = attack(model, images, labels, niters=10)
                for layer_ind, layer_name in source_layers[model_name(model_class)]:
#                     print(layer_name)
                    fla_adversarial_xs = mid_attack_adjust_generate(with_projection, model, images, fla_input_xs, labels, dic.get(layer_name), **(fla_params[attack_name]))
                    fla_transfer_list = test_adv_examples_across_models(transfer_models, fla_adversarial_xs, images, labels)
                    for i, (fooling_ratio, accuracy_perturbed, accuracy_original, attacked_model_path) in enumerate(fla_transfer_list):
                        out_df = log(out_df,model_class,attacked_model_path, transfer_models[i][0], source_weight_path, batch_i, layer_ind, layer_name, attack_name, True, fooling_ratio, accuracy_perturbed, accuracy_original)



            out_df.to_csv("result.csv", sep=',', encoding='utf-8')


if __name__ == "__main__":
    out_df = pd.DataFrame(columns=['source_model', 'source_model_file', 'target_model','target_model_file', 'batch_index','layer_index', 'layer_name', 'fool_method', 'with_fla',  'fool_rate', 'acc_after_attack', 'original_acc'])

    complete_loop(None, out_df);



















