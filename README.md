# Enhancing Adversarial Example Transferability with an Intermediate Level Attack 
This is the repo for the major experiment code for 2019 ICML submission. 

# Structure

The experiment of ILAP against I-FGSM and I-FGSM with momentum on cifar10 is in ```cifar10_experiments``` folder. Similar experiment on Imagenet is in ```imagenet_experiments```. Code of each experiment is self-contained in its own folder, except for the data. 

# Running Instruction 

```
# cifar10
cd cifar10_experiments
python all_in_one.py
cd imagenet_experiments
# imagenet
python all_in_one.py
```

Use proc_data.ipynb to visualize result.
