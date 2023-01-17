import sys
sys.path.append('./src/')
from model import *
from functions import *
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--mode', default='test', type=str)

# Input settings                                                                                                                                                 
parser.add_argument('--input_directory', default='./example_input/', type=str)
parser.add_argument('--input_flag', default='A', type=str)

# Training settings                                                                                                                  
parser.add_argument('--n_epochs', default=10, type=int)
parser.add_argument('--batch_size', default=200, type=int) #10 #16
parser.add_argument('--hidden_dim', default=20, type=int) #32
parser.add_argument('--learning_rate', default=0.01, type=float)

# Output settings                                                                                                                                  
parser.add_argument('--log_freq', default=10, type=int)
parser.add_argument('--output_directory', default='./example_output/', type=str)
parser.add_argument('--max_feat', default=15, type=float)

args = parser.parse_args()

print("Input:",args.input_flag)

if args.mode!='train' and args.mode!='test':
    print("Error: mode must be either train or test.")
    exit()
print("Mode:",args.mode)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:",device)

input_name='data_'+args.input_flag+'.csv'

################################################

# Load data.
full_dataset,classes,vocab,total_entries,float_headers=load_dataset(args.input_directory, input_name, col_to_explain='class')

vocab_size = len(vocab)
number_float_cols = len(float_headers)
print("Total entries:",total_entries)
print("Number of float columns:",number_float_cols)
print("Vocabulary length:",len(vocab))

train_loader = torch.utils.data.DataLoader(full_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(full_dataset, batch_size=1, shuffle=True)
#test_loader1 = torch.utils.data.DataLoader(full_dataset, batch_size=args.batch_size, shuffle=True)

print("train and test loader lengths:",len(train_loader),len(test_loader))

# Define classifier.
net = Classifier_of_vocabulary(total_entries,num_class=len(classes),n1=args.hidden_dim).to(device) 

if args.mode=='train': # Training mode.
    for epoch_idx in range(args.n_epochs):
        train_epoch(net,train_loader,input_flag=args.input_flag,lr=args.learning_rate,report_freq=args.log_freq)
        print("Done with epoch {}/{}".format(epoch_idx,args.n_epochs))

elif args.mode=='test': # Test mode.   
    net.load_state_dict(torch.load('./model_'+args.input_flag+'.pth')) #load previously saved model.
    #rate = 0.

    sals=[] # List of saliency map tensors.
    for i,(label,features) in enumerate(test_loader):
        features, label = features.to(device), label.to(device)
        text=features[:,number_float_cols:]
        input_for_net = features.to(torch.float32)
        
        pred = net(input_for_net) #, off )
        y=torch.argmax(pred, dim=0)
        #rate += y==label

        # Evaluate saliency map.
        feature_names,sal = saliency(net,input_for_net,float_headers+vocab)
        sals.append(sal.cpu().numpy())

        sentence_as_list=decode_categories(text[0].detach().cpu().numpy(),vocab)
        title=''
        for word in sentence_as_list:
            title+=word+' '
        #print("feature_names,sal:",feature_names,sal)

        #if i==10: break
    
    # Evaluate and plot average saliency map.
    sals=torch.tensor(np.array(sals))
    sal_average=torch.mean(sals,dim=0)
    sal_std=torch.std(sals,dim=0,unbiased=False)
    plot_average_saliency(sal_average,sal_std,features=feature_names,max_feat=len(feature_names),png_name=args.output_directory+'saliency_average_'+args.input_flag+'.png')

