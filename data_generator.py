import numpy as np
import pandas as pd
import random
from itertools import product
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--output_directory', default='./example_input/', type=str)
parser.add_argument('--n_categ', default=4, type=int) # number of categories wanted 
parser.add_argument('--n_unique', default=4, type=int) # allows for making unique letter+number combinations
parser.add_argument('--multiplicity', default=4, type=int) # allows for making multiple categorical value for a single letter+number combination
parser.add_argument('--n_classes', default=2, type=int) # number of classes (must match the labeling method in rule_for_label)
parser.add_argument('--rule', default='A', type=str) # Index for labeling rule
args = parser.parse_args()

letters=['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

def unique_values(index, n_unique):
    u_val=[]
    for i in range(n_unique):
        u_val.append(letters[i]+str(index)+'_')
    return u_val      

def make_categorical_list(input_list,multiplicity=1):
    clist=[]
    for x in range(multiplicity):
        for lett in input_list:
            #clist.append(str(x)+lett)
            clist.append(lett+str(x))
    return(clist)

def rule_for_label(counts,comb,rule):
    if rule=='A':
        if 'A' in comb[0]:
            label=1
            counts[0]+=1
        else:
            label=2
            counts[1]+=1
    elif rule=='B':
        if 'B'in comb[1]:
            label=1
            counts[0]+=1
        else:
            label=2
            counts[1]+=1
    elif rule=='C':
        if 'B' in comb[1] or 'C' in comb[2]:
            label=1
            counts[0]+=1
        else:
            label=2
            counts[1]+=1
    elif rule=='D':
        if 'A' in comb[0] or 'B'in comb[1] or 'C' in comb[1]:
            label=1
            counts[0]+=1
        else:
            label=2
            counts[1]+=1
    else:
        print("Error: invalid rule for labeling.")
        exit()
    return counts,label            

def make_dataset(n_categ, multiplicity, n_unique, rule='A'):

    possible_values=[]
    col_names=[]
    for i_cat in range(n_categ):
        possible_values.append(make_categorical_list(unique_values(i_cat, n_unique),multiplicity))
        col_names.append('Cat.'+str(i_cat))

    counts=list(np.zeros(args.n_classes,dtype=int))
    
    data={}
    data['class']=[]
    for cn in col_names:
        data[cn]=[]

    combinations=list(product(*possible_values))
    for comb in combinations:
        counts,label=rule_for_label(counts,comb,rule)
        wclass='Class'+str(label)
        data['class'].append(wclass)
        for i_v,val in enumerate(comb):
            data[col_names[i_v]].append(val)

    print("Counts:",counts)
    return data

data=make_dataset(n_categ=args.n_categ, multiplicity=args.multiplicity, n_unique=args.n_unique, rule=args.rule)
df=pd.DataFrame(data=data) 
csvname=args.output_directory+'data_'+args.rule+'.csv'
df.to_csv(csvname,sep=';',index=False)
