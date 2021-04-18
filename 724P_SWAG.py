#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
/*================================================================
*   Copyright (C) 2020. All rights reserved.
*   Author:  Leon Wang
*   Date: Sat Apr 17 22:54:50 2021
*   Email：leonwang@bu.edu
*   Description： 
================================================================*/
"""


import copy
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import os

#os.chdir('/Users/caesa/Desktop')


transform = transforms.Compose([transforms.ToTensor(),
                                                       transforms.Normalize((0.5,),(0.5,))])

#torch.cuda.is_available()

trainset = torchvision.datasets.USPS('data',download = True, train = True, transform = transform)
testset = torchvision.datasets.USPS('data',download = True, train = False, transform = transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size= 64,  num_workers= 3, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size= 64,  num_workers= 3, shuffle=True)



class classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3072, 256, bias=False)
#        self.fc1 = nn.Linear(256, 256, bias=False)
#        self.fc1 = nn.Linear(784, 256, bias=False)
        self.fc2 = nn.Linear(256, 128, bias=False)
        self.fc3 = nn.Linear(128, 10, bias=False)
        # self.norm1 = nn.BatchNorm1d(256)
        # self.norm2 = nn.BatchNorm1d(128)
        # self.norm3 = nn.BatchNorm1d(100)
        self.norm1 = nn.Identity(256)
        self.norm2 = nn.Identity(128)
        self.norm3 = nn.Identity(10)        

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.norm1(self.fc1(x)))
        x = F.relu(self.norm2(self.fc2(x)))
        x = F.log_softmax(self.norm3(self.fc3(x)), dim=1)
        return x

model = classifier()
loss_fn = nn.NLLLoss()

# [retrain the model for averaging]
model_swa = classifier()
#model_swa.load_state_dict(torch.load('50epches_cifar100.pt'))
#model_swa.load_state_dict(torch.load('289epches_cifar100.pt'))
model_swa.load_state_dict(torch.load('300epches_USPS.pt'))

epoch_rest = 150
cyclength = 30
lr_list = np.array([np.linspace(0.1,0.01,cyclength)]* (epoch_rest//cyclength)).ravel()
for t in range(epoch_rest):
    model_swa.train()
    for images, labels in trainloader:
            log_ps = model_swa(images)
            loss = loss_fn(log_ps, labels)
            loss.backward()
            # [gradient update part]
            with torch.no_grad():
                nn.utils.clip_grad_norm_(model_swa.parameters(), 1)
                for k,param in model_swa.named_parameters():                 
                        param -= lr_list[t] * param.grad
                        param.grad = None
    # [at the end of the cyclic learning rate]
    with torch.no_grad():
        if (t+1)%cyclength ==0:
            n_model = (t+1)/cyclength
            if n_model ==1:
                weights_swa = copy.deepcopy(model_swa.state_dict())
                weights_swa2 = copy.deepcopy(model_swa.state_dict())
                #weights_swalr =  {}
                for k,param in model_swa.named_parameters():      
                    weights_swa2[k] =   weights_swa2[k].pow(2)
                    #weights_swalr[k] = (-weights_swa[k]+param).reshape(-1).outer((-weights_swa[k]+param).reshape(-1))
            else:
                for k,param in model_swa.named_parameters():
                    # [update swa] 
                    print(param)
                    weights_swa[k] =  (weights_swa[k] * n_model + param)/(n_model +1)
                    weights_swa2[k] =  (weights_swa2[k] * n_model + param.pow(2))/(n_model +1)
                    #temp_outer = (-weights_swa[k]+param).reshape(-1).outer((-weights_swa[k]+param).reshape(-1))
                    #weights_swalr[k] = (weights_swalr[k] * n_model + temp_outer)/(n_model +1)
                    # [initialize w_hat again]                     
                    # model_swa.load_state_dict(torch.load('50epches_cifar100.pt'))
                    # model_swa.load_state_dict(torch.load('289epches_cifar100.pt'))                    
                    model_swa.load_state_dict(torch.load('300epches_USPS.pt'))                    
                    


                    
#torch.save(weights_swa, '289epches_cifar100swa2_lradjust.pt')
#torch.save(weights_swa2, '289epches_cifar100swa2square_lr_adjust.pt')
#torch.save(weights_swa, '300epches_USPSswa2.pt')
#torch.save(weights_swa2, '300epches_USPSswa2square.pt')
                    
                    
                    
                    
                    
                    
# SWAG Part 

model = classifier()
model_swa = classifier()
# model_swa.load_state_dict(torch.load('289epches_cifar100swa2_lradjust.pt'))
# model.load_state_dict(torch.load('289epches_cifar100.pt'))
model_swa.load_state_dict(torch.load('300epches_USPSswa2.pt'))
model.load_state_dict(torch.load('300epches_USPS.pt'))
model_swa.load_state_dict( torch.load('50epches_cifar10swa2.pt'))
model.load_state_dict(torch.load('50epches_cifar10.pt'))

import seaborn as sns                    
sns.distplot(model.fc3.weight.detach().numpy().ravel())
sns.distplot(model_swa.fc3.weight.detach().numpy().ravel())



# [testing error of weight swa] 
model.eval()
model_swa.eval()    
with torch.no_grad():
    total = 0
    correct = 0
    correct_swa = 0    
    #images, labels = test_iter.next()
    for images, labels in testloader:
        log_ps = model(images)
        max_index = torch.argmax(log_ps, dim =1)        
        total += labels.numel()
        correct += sum(max_index == labels).item()

        log_ps_swa = model_swa(images)
        max_index_swa = torch.argmax(log_ps_swa, dim =1)                
        correct_swa += sum(max_index_swa == labels).item()
        
    print('Test Accuarcy:'+str(round(correct / total * 100, 2)) + str('%'))        
    print('Test Accuarcy(SWA):'+str(round(correct_swa / total * 100, 2)) + str('%'))        
    


 
 ### for Bayesian sampling


trainset = torchvision.datasets.CIFAR10('data',download = True, train = True, transform = transform)
testset = torchvision.datasets.CIFAR10('data',download = True, train = False, transform = transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size= 9,  num_workers= 3, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size= 9,  num_workers= 3, shuffle=True)


mean_vec = torch.load('50epches_cifar10swa2.pt')
Sigma_vec = torch.load('50epches_cifar10swa2square.pt')
# mean_vec = torch.load('289epches_cifar100swa2.pt')
# Sigma_vec = torch.load('289epches_cifar100swa2square.pt')
#mean_vec = torch.load('300epches_USPSswa2.pt')
#Sigma_vec = torch.load('300epches_USPSswa2square.pt')


def sample_model(images, model_sample):    
    weight_sample = copy.deepcopy(model_bayes.state_dict())
    for k in weight_sample:
        Sigma_fc1 = Sigma_vec[k] - mean_vec[k].pow(2) + 10**(-5)
        weight_sample[k] =   torch.normal(mean_vec[k], std = Sigma_fc1.sqrt())    

    model_bayes.load_state_dict(weight_sample)    
    return model_bayes(images).exp()



test_iter = iter(testloader)
images, label = test_iter.next()
model_bayes = classifier()
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

prob_list = []
n_sample = 100
for i in range(n_sample):
    prob_list.append(sample_model(images, model_bayes).detach().numpy())
prob_list = np.array(prob_list)


plt.rcParams["axes.grid"] = False
fig = plt.figure(figsize = (26,12))
outer = gridspec.GridSpec(3, 3, wspace=0.2, hspace=0.2)
count =0
for i in range(9):
    inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[i], wspace=0.2, hspace=0.1)
    for j in range(2):
        ax = plt.Subplot(fig, inner[j])
        if j ==0:     
            ax.imshow(np.transpose(images[count,:]/2+0.5,(1,2,0)))
        else:
            pred_ub = np.quantile(prob_list[:,count,:],0.975, axis=0)
            pred_lb = np.quantile(prob_list[:,count,:],0.025, axis=0)
            ax.fill_between(classes, pred_ub, pred_lb, color='b', alpha=0.8)   
            #plt.xticks(rotation=90)
            for l in ax.get_xticklabels(): l.update(dict(rotation = 45))
            count += 1
        fig.add_subplot(ax)
       


sns.distplot(model.fc1.weight.detach().numpy().ravel())
sns.distplot(model_swa.fc1.weight.detach().numpy().ravel())      
plt.title('Layer 3 weights')  
plt.legend(labels = ['Initial Fit', 'SWA Fit'])                    