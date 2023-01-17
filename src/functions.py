import torch
import torchtext
import collections
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

tokenizer = torchtext.data.utils.get_tokenizer(None) #if None, returns split() function    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def simple_ohe(x,length):
    # returns a one-hot encoding of x, of given length.
    y=length*[0.]
    y[x-1]=1
    return y 

def load_dataset(input_dir, input_name, ngrams=1,min_freq=1,col_to_explain='Label'):
    df=pd.read_csv(input_dir+input_name,sep=';')
    position=0
    unique={}
    simil_vocab=[]
    float_headers=[]
    ranges={}
    for i_col in range(len(df.columns)): #starting from 1 because the first column is the counting, automatically created
        col=df.columns[i_col]
        if col != col_to_explain:
            if (type(df[col][0])==str or type(df[col][0])==int): #categorical data
                unique[col]=[]
                for row in range(df.shape[0]):
                    val=df[col][row]
                    if val not in unique[col]:
                        position+=1
                        unique[col].append(val)
                        simil_vocab.append(col+'_'+str(val))
            else: # non categorical data
                float_headers.append(col)
                max_val=-99999.
                min_val=99999.
                for row in range(df.shape[0]):
                    val=df[col][row]
                    if val>max_val: max_val=val
                    if val<min_val: min_val=val
                ranges[col]=[min_val,max_val]
    
    all_data=[]
    unique_categ_vals='' #across all categorical features
    classes=''
    already=[]

    number_float_cols=0
    
    for row in range(df.shape[0]):

        non_index_float=[] 
        non_index_categ=len(simil_vocab)*[0]
        for i_col in range(len(df.columns)): #starting from 1 because the first column is the counting, automatically created
            col=df.columns[i_col]
            val=df[col][row]
            if col != col_to_explain:
                if (type(df[col][0])==str or type(df[col][0])==int): #categorical data
                    to_add=col+'_'+str(val) #in this way, every vocabulary entry will refer to a specific value of a specific feature: "feature_value"
                    if (to_add not in unique_categ_vals): unique_categ_vals+=to_add+' '

                    non_index_categ1=non_index_categ
                    for ii,elem in enumerate(non_index_categ):
                        non_index_categ1[ii]=non_index_categ[ii] + simple_ohe(simil_vocab.index(to_add)+1,len(simil_vocab))[ii]
                    non_index_categ = non_index_categ1
                    
                else: #non categorical data
                    number_float_cols+=1

                    non_index_float.append((val-ranges[col][0])/(ranges[col][1]-ranges[col][0]))
                    
            else:
                if (val in already):
                    index=already.index(val)+1
                else:
                    index=len(already)+1 
                    already.append(val)
                classes+=str(val)+' '
        all_data.append((index,torch.cat((torch.tensor(non_index_float),torch.tensor(non_index_categ)))))
    
    classes=np.unique(classes.split(' '))
    
    total_entries=len(all_data[0][1])
    number_float_cols=int(number_float_cols/df.shape[0])

    return all_data,classes,simil_vocab,total_entries,float_headers
    
def encode(x,voc=None,tokenizer=tokenizer):
    v = vocab if voc is None else voc
    unk=v.get_stoi().get('7qo34c0v3lmo') 
    return [v.get_stoi().get(s,unk) for s in tokenizer(x)]                                                                                                                                              

def merge_saliency_for_category(saliency,headers): 
    h_all=[]
    for i in range(len(headers)):
        h_all.append(headers[i].split('_')[0])
    h_uniq=[]
    for ha in h_all:
        if ha not in h_uniq: h_uniq.append(ha)
    merged=torch.zeros(len(h_uniq))
    old_feat=headers[0].split('_')[0]
    som=0.0
    counts=0
    j=0
    for i,sal in enumerate(saliency[0]):
        new_feat=headers[i].split('_')[0]
        if new_feat!=old_feat or i==(len(saliency[0])-1):
            merged[j]=som/counts
            j+=1
            som=0.0
            counts=0
            old_feat=new_feat
        som+=sal
        counts+=1
    return h_uniq,merged

def saliency(net,input_data,headers):
    X=input_data
    X.requires_grad_()
    out = net(X)
    index = torch.argmax(out)
    score_max = out[0][index]
    score_max.backward()
    saliency=X.grad.data.abs()
    h_uniq,merged=merge_saliency_for_category(saliency,headers)
    return h_uniq,merged

def plot_saliency(saliency,features,max_feat=None,title='test sentence',png_name='saliency.png'):
    if max_feat==None: max_feat=len(saliency)
    fig = plt.figure(1, figsize=(4,4))
    plt.rcParams.update({'font.size': 4})
    plt.grid(color='grey', linestyle='-', linewidth=0.2)
    plt.plot(np.linspace(0,max_feat,max_feat),saliency[:max_feat].cpu().numpy(),linewidth=2,color='C0')
    plt.scatter(np.linspace(0,max_feat,max_feat),saliency[:max_feat].cpu().numpy(),color='C0')
    plt.xlabel('Feature')
    plt.ylabel('Saliency Value')
    plt.title(title,wrap=True)
    #plt.xticks(np.arange(max_feat),features[:max_feat],rotation=90)
    plt.xticks(np.linspace(0,max_feat,max_feat),features[:max_feat],rotation=90)
    fig.savefig(png_name,dpi=150) 
    plt.clf()
    print("DONE:",png_name)

def plot_average_saliency(saliency,saliency_std,features,max_feat=None,png_name='saliency.png'):
    if max_feat==None: max_feat=len(saliency)
    fig = plt.figure(1, figsize=(4, 4))
    plt.rcParams.update({'font.size': 6.8})
    #plt.plot(np.linspace(0,max_feat,max_feat),torch.add(saliency[:max_feat],-saliency_std[:max_feat]).cpu().numpy(),linewidth=0.5,color='C0')
    #plt.plot(np.linspace(0,max_feat,max_feat),saliency[:max_feat].cpu().numpy(),linewidth=1,color='C0')
    #plt.plot(np.linspace(0,max_feat,max_feat),torch.add(saliency[:max_feat],saliency_std[:max_feat]).cpu().numpy(),linewidth=0.5,color='C0')
    plt.errorbar(np.linspace(0,max_feat,max_feat),saliency[:max_feat].cpu().numpy(),yerr=saliency_std,fmt='none',elinewidth=1,ecolor='C0')
    plt.scatter(np.linspace(0,max_feat,max_feat),saliency[:max_feat].cpu().numpy(),color='C0')
    plt.xlabel('Feature')
    plt.ylabel('Saliency Value')
    #plt.xticks(np.arange(max_feat),features[:max_feat],rotation=90)     
    plt.xticks(np.linspace(0,max_feat,max_feat),features[:max_feat],rotation=90)
    fig.savefig(png_name,dpi=300)
    plt.clf()
    print("DONE:",png_name)

def decode_categories(encoded_list,vocab):
    categ_list=[]
    for j0,one_or_zero in enumerate(encoded_list):
        if int(one_or_zero)==1: categ_list.append(vocab[j0])
    return categ_list
