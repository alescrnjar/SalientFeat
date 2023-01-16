import numpy as np
import pandas as pd
import random
from itertools import product

n_categ=4
multiplicity=4
n_unique=4

nclasses=2

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

def rule_for_label(counts,comb):
    
    #if 'A' in comb[0]:
    #if 'B' in comb[1]:
    if 'B' in comb[1] or 'C' in comb[2]:
        label=1
        counts[0]+=1
    else:
        label=2
        counts[1]+=1

    return counts,label            

def make_dataset(n_categ, multiplicity, n_unique):

    possible_values=[]
    col_names=[]
    for i_cat in range(n_categ):
        possible_values.append(make_categorical_list(unique_values(i_cat, n_unique),multiplicity))
        col_names.append('Categ'+str(i_cat))

    counts=list(np.zeros(nclasses,dtype=int))
    
    data={}
    data['class']=[]
    for cn in col_names:
        data[cn]=[]

    combinations=list(product(*possible_values))
    for comb in combinations:
        counts,label=rule_for_label(counts,comb)
        wclass='Class'+str(label)
        data['class'].append(wclass)
        for i_v,val in enumerate(comb):
            data[col_names[i_v]].append(val)

    print("Counts:",counts)
    return data

data=make_dataset(n_categ=n_categ, multiplicity=multiplicity, n_unique=n_unique)
df=pd.DataFrame(data=data) 
csvname='./example_input/data.csv'
df.to_csv(csvname,sep=';',index=False)
