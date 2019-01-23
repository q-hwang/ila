import sys
import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

from imagenet_attacks import ifgsm, momentum_ifgsm, ILA, batch_size, mean_arr, stddev_arr
from tqdm import tqdm

def attack_name(attack):
    if attack == ifgsm:
        return 'ifgsm'
    elif attack == momentum_ifgsm:
        return 'momentum_ifgsm'
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
        


ifgsm_params = {'learning_rate': 0.008, 'epsilon': 0.03}
momentum_ifgsm_params = {'learning_rate': 0.018, 'epsilon': 0.03, 'decay': 0.9}
cw2_params = {'targeted': False}            
attacks = [(attack_name(ifgsm), wrap_attack(ifgsm, ifgsm_params)),
          (attack_name(momentum_ifgsm), wrap_attack(momentum_ifgsm, momentum_ifgsm_params))]
          

data_preprocess = (batch_size, mean_arr, stddev_arr)

source_models = [('SqueezeNet1.0',models.squeezenet1_0)
                 ('DenseNet121',models.densenet121), 
                 ('SqueezeNet1.0',models.squeezenet1_0),
                 ('alexnet',models.alexnet)]


transfer_models = [('ResNet18',models.resnet18), 
                 ('DenseNet121',models.densenet121), 
                 ('SqueezeNet1.0',models.squeezenet1_0),
                 ('alexnet',models.alexnet)]


with_projection = True
fla_params = {attack_name(momentum_ifgsm): {'niters': 10, 'learning_rate': 0.018, 'epsilon':0.03, 'coeff': 0.8}, 
              attack_name(ifgsm): {'niters': 10, 'learning_rate': 0.01 , 'epsilon':0.03, 'coeff': 0.8}
             } 
             

    
def get_source_layers(model_name, model):
    if model_name == 'ResNet18':
        # exclude relu, maxpool
        return list(enumerate(map(lambda name: (name, model._modules.get(name)), ['conv1', 'bn1', 'layer1', 'layer2','layer3','layer4','fc'])))
    
    elif model_name == 'DenseNet121':
        # exclude relu, maxpool
        layer_list = list(map(lambda name: (name, model._modules.get('features')._modules.get(name)), ['conv0', 'denseblock1', 'transition1', 'denseblock2', 'transition2', 'denseblock3', 'transition3', 'denseblock4', 'norm5']))
        layer_list.append(('classifier', model._modules.get('classifier')))
        return list(enumerate(layer_list))
                                             
    elif model_name == 'SqueezeNet1.0':
        # exclude relu, maxpool
        layer_list = list(map(lambda name: ('layer '+name, model._modules.get('features')._modules.get(name)), ['0','3','4','5','7','8','9','10','12']))
        layer_list.append(('classifier', model._modules.get('classifier')._modules.get('1')))
        return list(enumerate(layer_list))
    
    elif model_name == 'alexnet':
        # exclude avgpool
        layer_list = list(map(lambda name: ('layer '+name, model._modules.get('features')._modules.get(name)), ['0','3','6','8','10']))
        layer_list += list(map(lambda name: ('layer '+name, model._modules.get('classifier')._modules.get(name)), ['1','4','6']))
        return list(enumerate(layer_list))
    
    else:
        # model is not supported
        assert False
    
    

def log(out_df, source_model_name, target_model_name, batch_index, layer_index, layer_name, fool_method, with_fla,  fool_rate, acc_after_attack, original_acc):
    return out_df.append({
        'source_model':source_model_name, 
        'target_model':target_model_name,
        'batch_index':batch_index,
        'layer_index':layer_index, 
        'layer_name':layer_name, 
        'fool_method':fool_method, 
        'with_fla':with_fla,  
        'fool_rate':fool_rate, 
        'acc_after_attack':acc_after_attack, 
        'original_acc':original_acc},ignore_index=True)
                                              

def get_data(batch_size, mean_arr, stddev_arr):
    transform_test = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean_arr, stddev_arr),
                      ])
    testset = torchvision.datasets.ImageFolder(root='/share/cuvl/datasets/imagenet/val', 
                                               transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, 
                                             num_workers=8, pin_memory=True)
    return testloader

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
    for name, net_class in transfer_models:
        net = net_class(pretrained=True).cuda()
        net.eval()
        res = get_fool_adv_orig(net, adversarial_xs, originals, labels)
        res.append(name)
        accum.append(res)
    return accum

    

def complete_loop(sample_num, out_df,with_projection):
    testloader = get_data(*data_preprocess)
    for source_model_name, model_class in source_models:
        model = model_class(pretrained=True).cuda()
        model.eval()
        for attack_name, attack in attacks:
            print('using source model {0} attack {1}'.format(source_model_name, attack_name))
            
            for batch_i, data in tqdm(enumerate(testloader, 0)):
                if batch_i < 1500:
                    continue
                if batch_i%100 == 0: 
#                     print("batch" , batch_i)
                    save_to_csv(out_df)
                if batch_i == sample_num:
                    break
                images, labels = data
                images, labels = images.cuda(), labels.cuda() 
        
               
                #### baseline 
                
                ### generate
                adversarial_xs = attack(model, images, labels, niters=20)
                
                ### eval
                transfer_list = test_adv_examples_across_models(transfer_models, adversarial_xs, images, labels)
                for target_fool_rate, target_acc_attack, target_acc_original, transfer_model_name in transfer_list:
                    out_df = log(out_df,source_model_name, transfer_model_name, 
                                 batch_i, np.nan, "", attack_name, False, 
                                 target_fool_rate, target_acc_attack, target_acc_original)
                
                #### ILA
                
                ### generate
                ## step1: reference 
                fla_input_xs = attack(model, images, labels, niters=10) 
                
                ## step2: ILA target at different layers
                for layer_ind, (layer_name, layer) in get_source_layers(source_model_name, model):
#                     print(layer_name)
                    fla_adversarial_xs = ILA(with_projection, model, images, fla_input_xs, labels, layer, **(fla_params[attack_name]))
                    
                    ### eval
                    fla_transfer_list = test_adv_examples_across_models(transfer_models, fla_adversarial_xs, images, labels)
                    for target_fool_rate, target_acc_attack, target_acc_original, transfer_model_name in fla_transfer_list:
                        out_df = log(out_df,source_model_name, transfer_model_name, batch_i, layer_ind, layer_name, attack_name, True, target_fool_rate, target_acc_attack, target_acc_original)
            
            save_to_csv(out_df)    
            
          
def save_to_csv(out_df):
     #save csv
    out_df.to_csv("imagenet_result.csv", sep=',', encoding='utf-8')

    

if __name__ == "__main__":
    out_df = pd.DataFrame(columns=['source_model','target_model','batch_index','layer_index', 'layer_name', 'fool_method', 'with_fla',  'fool_rate', 'acc_after_attack', 'original_acc'])
    
    complete_loop(None , out_df, with_projection);
    


















