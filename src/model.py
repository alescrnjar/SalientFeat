import torch
import collections
import os

import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_epoch(net,dataloader,input_flag, lr=0.01,optimizer=None,loss_fn = torch.nn.CrossEntropyLoss(),epoch_size=None, report_freq=200):
    optimizer = optimizer or torch.optim.Adam(net.parameters(),lr=lr)
    loss_fn = loss_fn.to(device)
    net.train()
    acc,count,i = 0,0,0
    for batch_idx,(labels,features) in enumerate(dataloader):
        features, labels = features.to(device), labels.to(device) 
        input_for_net=features.to(torch.float32)

        optimizer.zero_grad()
        out = net(input_for_net)

        loss = loss_fn(out,labels) 
        #if batch_idx==0 and epoch_idx==0: print("First loss:",loss.item(),batch_idx)
        loss.backward()
        optimizer.step()

        _,predicted = torch.max(out,1)
        acc+=(predicted==labels).sum()
        count+=len(labels)
        i+=1

        if epoch_size and count>epoch_size:
            break
        if batch_idx%report_freq==0 or batch_idx==0:
            torch.save(net.state_dict(), './model_'+input_flag+'.pth')
    print("Epoch Done: Loss:",loss.item(),"acc/count:",acc.item()/count) #,"predicted/labels:",predicted,labels)

class Classifier_of_vocabulary(torch.nn.Module):                                                                                                                                 
    def __init__(self, input_length, num_class, n1=20):
        super().__init__()                                                                                                                                                         
        input_dim = input_length                                                                                                               
        output_dim = num_class                                                                                                                                                     
        self.hidden_layer1 = torch.nn.Sequential(torch.nn.Linear(input_dim, n1)) 
        self.output_layer = torch.nn.Sequential(torch.nn.Linear(n1, output_dim)) 

    def forward(self, x):                                                                                                                                    
        output = self.hidden_layer1(x)                                                                                                                                             
        output = self.output_layer(output)                                                                                                                                         
        return output.to(device)    