import torch
import torchtext
import collections
import os

import numpy as np
import matplotlib.pyplot as plt

from torchinfo import summary
#from torchnlp_AC import *

#mode='train'
#mode='test'

# /home/acrnjar/Dropbox/Thorsten_Project/pre_github

#from Model import *

import argparse
import random
import shap
import pandas as pd

parser = argparse.ArgumentParser()


print("# TODO as input I could take categorical data to which apply ohe")

parser.add_argument('--mode', default='test', type=str)

# Input settings                                                                                                                                                 
#parser.add_argument('--input_directory', default='./example_input/', type=str)
#parser.add_argument('--input_name', default='protgpt2_sequences.dat', type=str)
#parser.add_argument('--min_length', default=100, type=int)
#parser.add_argument('--max_input_data', default=50000, type=int)
#parser.add_argument('--max_eval_data', default=1000, type=int)
#parser.add_argument('--remove_n', default=True, type=bool)

#parser.add_argument('--max_dataset', default=1000, type=int) #100 #1000 #10000 

# Training settings                                                                                                                  
parser.add_argument('--n_epochs', default=100, type=int)
parser.add_argument('--batch_size', default=200, type=int) #10 #16
#parser.add_argument('--embed_dim', default=63, type=int) #64 #SE USI OHE: DEVE ESSERE UGUALE A VOCAB_SIZE!
parser.add_argument('--hidden_dim', default=32, type=int) #32
parser.add_argument('--learning_rate', default=0.01, type=float)

parser.add_argument('--max_feat', default=15, type=float)

# Output settings                                                                                                                                  
parser.add_argument('--log_freq', default=10, type=int)
#parser.add_argument('--output_directory', default='./example_output/', type=str)

args = parser.parse_args()

mode=args.mode
print("Mode:",mode)

print("WARNING: train and test model are separated. If you update the model usage, make a new training to have a new .pth file to upload")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device='cuda'                                                                                                                                                                    
#device = "cpu"                                                                                                                                                                    
print("Device:",device)
#tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
#print("IL TOKENIZER NON DEVE PIU ESSERE BASIC ENGLISH MA USARE SPLIT COME IN AMINOX PERO CON WORDS INVECE CHE CON CHARACTERS")
tokenizer = torchtext.data.utils.get_tokenizer(None) #if None, returns split() function    

def simple_ohe(x,length):
    y=length*[0.]
    y[x-1]=1
    return y #.float().to(device)
def load_mixed_dataset(ngrams=1,min_freq=1,col_to_explain='Label'):
    #print("AAA")
    df=pd.read_csv('mixed_data.csv',sep=';')
    #df=pd.read_csv('/home/acrnjar/Downloads/mushrooms.csv',sep=',')
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
            #print(col,col_to_explain)
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
    unk=v.get_stoi().get('7qo34c0v3lmo') #0 #alternativa: '0000' cioe un token che sicuramente non esiste
    return [v.get_stoi().get(s,unk) for s in tokenizer(x)]

def train_epoch(net,dataloader,lr=0.01,optimizer=None,loss_fn = torch.nn.CrossEntropyLoss(),epoch_size=None, report_freq=200):
    optimizer = optimizer or torch.optim.Adam(net.parameters(),lr=lr)
    loss_fn = loss_fn.to(device)
    net.train()
    acc,count,i = 0,0,0
    for batch_idx,(labels,features) in enumerate(dataloader):

        features, labels = features.to(device), labels.to(device) #AC orig

        

        input_for_net=features.to(torch.float32)

        optimizer.zero_grad()
        out = net(input_for_net)

        loss = loss_fn(out,labels) #cross_entropy(out,labels)
        if batch_idx==0 and epoch_idx==0: print("First loss:",loss.item(),batch_idx)
        #print("loss:",loss)
        loss.backward()
        optimizer.step()

        _,predicted = torch.max(out,1)
        acc+=(predicted==labels).sum()
        count+=len(labels)
        i+=1

        if epoch_size and count>epoch_size:
            break
        if batch_idx%report_freq==0 or batch_idx==0:
            torch.save(net.state_dict(), './model.pth')
    #print("Epoch Done: Loss:",loss.item(),"acc:",acc.item(),"count:",count,"acc/count:",acc.item()/count,"predicted/labels:",predicted,labels)
    print("Epoch Done: Loss:",loss.item(),"acc/count:",acc.item()/count,"predicted/labels:",predicted,labels)

class Classifier_of_vocabulary(torch.nn.Module):                                                                                                                                 
    def __init__(self, input_length, hidden_dim, num_class, n1=20):
        super().__init__()                                                                                                                                                         
        input_dim = input_length                                                                                                               
        output_dim = num_class                                                                                                                                                     
        self.hidden_layer1 = torch.nn.Sequential(torch.nn.Linear(input_dim, n1)) #, torch.nn.ReLU()) #, nn.Dropout(0.3))                                                              

        self.output_layer = torch.nn.Sequential(torch.nn.Linear(n1, output_dim)) #, torch.nn.LogSoftmax()) #nn.Sigmoid())

        
    def forward(self, x): #, labels):                                                                                                                                              
        output = self.hidden_layer1(x)                                                                                                                                             
        output = self.output_layer(output)                                                                                                                                         
        return output.to(device)                                                                                                                                                  

max_string_len=22
def padify(b,voc=None,tokenizer=tokenizer):
    v = [encode(x[1],voc=voc,tokenizer=tokenizer) for x in b]
    l = max_string_len #22 #max(map(len,v))
    return ( # tuple of two tensors - labels and features
        torch.LongTensor([t[0]-1 for t in b]),
        torch.stack([torch.nn.functional.pad(torch.tensor(t),(0,l-len(t)),mode='constant',value=vocab.get_stoi().get('the')) for t in v])
    )

# Packed Sequences
def pad_length(b):
    # build vectorized sequence
    v = [encode(x[1]) for x in b]
    # compute max length of a sequence in this minibatch and length sequence itself
    len_seq = list(map(len,v))
    l = max(len_seq)
    return ( # tuple of three tensors - labels, padded features, length sequence
        torch.LongTensor([t[0]-1 for t in b]),
        torch.stack([torch.nn.functional.pad(torch.tensor(t),(0,l-len(t)),mode='constant',value=0) for t in v]),
        torch.tensor(len_seq)
    )

def merge_saliency_for_category(saliency,headers): #,method='average'):

    print("LEN H:",len(headers))
    print("LEN S:",len(saliency[0]))
    
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
    plt.plot(np.linspace(0,max_feat,max_feat),torch.add(saliency[:max_feat],-saliency_std[:max_feat]).cpu().numpy(),linewidth=0.5,color='C0')
    plt.plot(np.linspace(0,max_feat,max_feat),saliency[:max_feat].cpu().numpy(),linewidth=1,color='C0')
    plt.plot(np.linspace(0,max_feat,max_feat),torch.add(saliency[:max_feat],saliency_std[:max_feat]).cpu().numpy(),linewidth=0.5,color='C0')
    plt.xlabel('Feature')
    plt.ylabel('Saliency Value')
    #plt.xticks(np.arange(max_feat),features[:max_feat],rotation=90)     
    plt.xticks(np.linspace(0,max_feat,max_feat),features[:max_feat],rotation=90)
    fig.savefig(png_name,dpi=300)
    plt.clf()
    print("DONE:",png_name)

def explain_with_shap(net,data):
    X=data
    X.requires_grad_()
    # https://towardsdatascience.com/pytorch-shap-explainable-convolutional-neural-networks-ece5f04c374f                                                                            
    expl = shap.DeepExplainer(net,X)
    shap_values = expl.shap_values(X)
    shap_numpy = [np.swapaxes(s, 1, -1) for s in shap_values]
    #test_numpy=np.swapaxes(X.cpu().numpy(), 1, -1)                                      
    test_numpy=np.swapaxes(X.cpu().detach().numpy(), 1, -1)
    
    return shap_values,test_numpy

def shap_summary_plot(shap_values,test_numpy,features,classes,png_name='shap.png',nlayers=20):
    fig = plt.figure(1, figsize=(4, 4))
    ##shap.force_plot(plot_cmap=color_list)                                                                                                                                        
    shap.summary_plot(shap_values,test_numpy,show=False,feature_names=features,class_names=classes,max_display=nlayers) #,color_bar=color_list) #,class_names=[classes_map[ii] for ii in cat_model.classes_],plot_type="bar")                                                                                                                                         
    fig.savefig(png_name,dpi=150)
    plt.clf()
    print("DONE:",png_name)


def decode_text(encoded_word_list):
    text_list=[]
    for j0,word_encoded in enumerate(encoded_word_list):
        text_list.append(vocab.get_itos()[int(word_encoded)])
    return text_list

def decode_categories(encoded_list,simil_vocab):
    categ_list=[]
    print("encoded_list:",len(encoded_list),encoded_list)
    print("simil_vocab:",len(simil_vocab),simil_vocab)
    for j0,one_or_zero in enumerate(encoded_list):
        if int(one_or_zero)==1: categ_list.append(vocab[j0])
    return categ_list

################################################

full_dataset,classes,vocab,total_entries,float_headers=load_mixed_dataset(col_to_explain='class')

vocab_size = len(vocab)
number_float_cols=len(float_headers)
print("Total entries:",total_entries)
print("Number of float columns:",number_float_cols)
print("Vocabulary length:",len(vocab))

for i,data in enumerate(full_dataset):
    print(i,data,len(data[1]))
    if (i==5): break #exit()

train_loader = torch.utils.data.DataLoader(full_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(full_dataset, batch_size=1, shuffle=True)
test_loader1 = torch.utils.data.DataLoader(full_dataset, batch_size=args.batch_size, shuffle=True)

print("train and test loader lengths:",len(train_loader),len(test_loader))

net= Classifier_of_vocabulary(total_entries,args.hidden_dim,len(classes)).to(device) #AC
if mode=='train':
    for epoch_idx in range(args.n_epochs):
        train_epoch(net,train_loader,lr=args.learning_rate,report_freq=args.log_freq)
        print("Done with epoch {}/{}".format(epoch_idx,args.n_epochs))
elif mode=='test':    
    net.load_state_dict(torch.load('./model.pth'))
    rate=0.

    sals=[]
    for i,(label,features) in enumerate(test_loader):
        #print("--- --- ---",i)
        features, label = features.to(device), label.to(device)
        text=features[:,number_float_cols:]
        #print("features:",features,len(features[0]))
        input_for_net=features.to(torch.float32)
        
        pred = net(input_for_net) #, off )
        y=torch.argmax(pred, dim=0)
        rate+=y==label
        data0=input_for_net

        huniq,sal=saliency(net,data0,float_headers+vocab)
        sals.append(sal.cpu().numpy())
        #print("sal:",sal,sal.shape) #.shape) #,sal)
        
        #print("text[0]:",text[0],len(text[0]))
        sentence_as_list=decode_categories(text[0].detach().cpu().numpy(),vocab)
        #print("sentence_as_list:",sentence_as_list)
        title=''
        for word in sentence_as_list:
            title+=word+' '
        print("huniq,sal:",huniq,sal)
        

    sals=torch.tensor(np.array(sals))
    sal_average=torch.mean(sals,dim=0)
    sal_std=torch.std(sals,dim=0,unbiased=False)
    plot_average_saliency(sal_average,sal_std,features=huniq,max_feat=len(huniq),png_name='saliency_average.png')

#print("TO DO: perche l acc si pianta a un valore anche se la loss continua a fare saliscendi?")
#print("Is the sorting forced?")
#print("maybe use a wiki webcrawler? Or UNIPROT crawler")
#print("Adjust embed_dim?")
#print("come e possibile che ci siano degli unk? Vuol dire che stai mal costruendo il dizionario.")
#print("ROC AUC?")

#print("\n\n HAI RIPRISTINATO LA SOFTMAX????? \n\n RICREARE BENE LA DISTINZIONE TRA TRAIN E TEST\n\n EVENTUALMENTLY DOVRO ANCHE IMPLEMENTARE LA POSSIBILITA DI IMBALANCED DATA IN MIXED_DATASET.py, SIA PER CATEGORICAL CHE PER FLOAT\n\nCOME SI FA A BILANCIARE UNA VARIABILE CATEGORIALE? SE PER ESEMPIO M/F COME INFO MANCA: NON PUOI FARE UNA MEDIA!\n\n")

