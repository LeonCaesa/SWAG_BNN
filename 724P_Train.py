#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
/*================================================================
*   Copyright (C) 2020. All rights reserved.
*   Author:  Leon Wang
*   Date: Mon Apr 12 11:00:46 2021
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
import matplotlib.pyplot as plt
import os

os.chdir('/Users/caesa/Desktop')


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
learning_rate = 0.001

# adam initialization
moment1 = copy.deepcopy(model.state_dict())
for k,v in moment1.items():
    moment1[k] = torch.zeros(moment1[k].shape)    
moment2 = copy.deepcopy(moment1)
beta1 = 0.9
beta2 = 0.999
epislon = np.power(10.0,-8)



epochs = 300
loss_list = []

for t in range(epochs):
    model.train()
    #images, labels = train_iter.next()
    for images, labels in trainloader:
        log_ps = model(images)
        loss = loss_fn(log_ps, labels)
        loss.backward()
    
        # [gradient update part]
        with torch.no_grad():
            #nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            for k,param in model.named_parameters():                 
                    moment1[k] = beta1 * moment1[k]  + (1-beta1) * param.grad
                    moment2[k] = beta2 * moment2[k]  + (1-beta2) * param.grad.pow(2)
                    #param -= learning_rate/(1 + 0.5 *t) * moment1[k]/ (moment2[k].pow(1/2) + epislon)
                    param -= learning_rate * moment1[k]/ (moment2[k].pow(1/2) + epislon)
                    param.grad = None
                    
            # total = 0
            # correct = 0                
            # max_index = torch.argmax(log_ps, dim =1)
            # total += labels.numel()
            # correct = sum(max_index == labels).item()
            #print('Accuarcy:'+str(correct/total*100))       
                  
    print(loss.item())
    loss_list.append(loss.item())
    
    # [testing set evaluation]
    model.eval()    
    with torch.no_grad():
        total = 0
        correct = 0
        #images, labels = test_iter.next()
        for images, labels in testloader:
            log_ps = model(images)
            max_index = torch.argmax(log_ps, dim =1)
            total += labels.numel()
            correct += sum(max_index == labels).item()
        
        print('Test Accuarcy:'+str(correct / total * 100))
            
plt.plot(loss_list)                
plt.show()

#torch.save(model.state_dict(), '300epches_USPS.pt')


        
