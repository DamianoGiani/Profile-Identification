from models import *
from sklearn import metrics
from datasets import *
import argparse
from collections import Counter
import numpy as np
#from torch.utils.tensorboard import SummaryWriters
from torch import optim
from tqdm import tqdm
from sklearn.metrics import balanced_accuracy_score
import time
from datetime import datetime
import random

random.seed(5)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, help="Inserisci il tipo di esperimento fra Single,Multi e Cross",required=True)
    parser.add_argument("--fold", type=int, help="Inserisci fold",required=False)
    parser.add_argument("--social", type=str, help="Inserisci Social",required=False)
    return parser


def main(args):
    start = time.time()
    experimentType=args.experiment
    if(args.social==None):
        social=''
    else:
        social=args.social
    fold=args.fold
    DEVICE='cuda'
    
    if(experimentType=="Single"):    
        train_dataset = OneSocialDatasetTrain('../index.csv',experimentType,fold, social)
        test_dataset=OneSocialDatasetTest('../index.csv',experimentType,fold, social)        
    elif(experimentType=="Multi"):
        train_dataset = AllSocialDatasetTrain('../index.csv',experimentType,fold)
        test_dataset=AllSocialDatasetTest('../index.csv',experimentType,fold)
    elif(experimentType=="Cross"):
        train_dataset = CrossSocialTrainDataset('../index.csv',experimentType, social)
        test_dataset=CrossSocialTestDataset('../index.csv',experimentType, social)
    

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True,
                                                num_workers=24)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False,
                                               num_workers=24)
       
    model=ResProj()    
    model.to(DEVICE)

    optimizer = optim.Adam(model.parameters(), 1e-3)    
    criterion = nn.CrossEntropyLoss()

    print(f"Training patches for epoch: {len(train_dataset)}")
    print(f"Validation patches for epoch: {len(test_dataset)}")

    #writer = SummaryWriter()
    accuracy=0
    t=0
    for epoch in range(1000):
        train_bar = tqdm(train_loader)
        train_bar.set_description(f"Training epoch {epoch + 1}")

        model.train()

        total_loss = 0.
        n_batch = 0

        for data in train_bar:
            inputs, labels = data
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            train_bar.set_postfix(loss=float(loss))
            loss.backward()
            optimizer.step()

            total_loss += float(loss)
            n_batch += 1

        #writer.add_scalar('train_loss', total_loss / n_batch, epoch)


        if(epoch%5==0):
            model.eval()
            test_bar = tqdm(test_loader)
            test_bar.set_description(f"Validating epoch {epoch + 1}")


            with torch.no_grad():
                y_true = []
                y_pred = []
                for data in test_bar:
                    inputs, labels,imagename = data
                    imagelist = list(set(imagename))
                    
                    diction={}
                    dictionlabel={}
                    for i in imagelist:

                        diction[i] = []
                        dictionlabel[i]=[]

                    inputs = inputs.to(DEVICE)
                    outputs = model(inputs).cpu()
                    
                    for j in range(len(outputs)):
                        
                        diction[imagename[j]].append(outputs[j].detach().numpy().argmax(axis=0))
                        dictionlabel[imagename[j]].append(labels[j].item())
                    
                    for n in diction.keys():
                        count_dict = Counter(diction[n])
                        most_label_pred = count_dict.most_common(1)[0][0]

                        count_dict_label = Counter(dictionlabel[n])
                        label = count_dict_label.most_common(1)[0][0]

                        y_true.append(label)
                        y_pred.append(most_label_pred)

            y_true = np.hstack(y_true)
            y_pred = np.hstack(y_pred)


            #writer.add_scalar('test_accuracy', balanced_accuracy_score(y_true, y_pred), epoch)
            if(balanced_accuracy_score(y_true, y_pred)>accuracy):
                accuracy=balanced_accuracy_score(y_true, y_pred)
                torch.save(model.state_dict(),experimentType+'Social-'+social+str(datetime.fromtimestamp(start).strftime("%Y-%m-%d %H:%M:%S"))+'.pt')
                t=0
            else:
                t+=1
                if(t>10):
                    print(optimizer.param_groups[0]['lr'])
                    optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']/10
                    print(optimizer.param_groups[0]['lr'])
                    t=0

    print("Training complete!")
    #writer.close()

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)