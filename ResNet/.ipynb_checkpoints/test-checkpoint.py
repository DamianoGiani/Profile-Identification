from models import *
from sklearn import metrics
from datasets import *
import argparse
from collections import Counter
import numpy as np


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, help="Inserisci il tipo di esperimento fra Single,Multi e Cross",required=True)
    parser.add_argument("--fold", type=int, help="Inserisci fold",required=False)
    parser.add_argument("--social", type=str, help="Inserisci Social",required=False)
    return parser

def main(args):
    
    experimentType=args.experiment
    
    DEVICE='cuda'
    
    if(experimentType=="Single"):    
        train_dataset = OneSocialDatasetTrain('index.csv',experimentType,args.fold, args.social)
        test_dataset=OneSocialDatasetTest('index.csv',experimentType,args.fold, args.social)        
    elif(experimentType=="Multi"):
        train_dataset = AllSocialDatasetTrain('index.csv',experimentType,args.fold)
        test_dataset=AllSocialDatasetTest('index.csv',experimentType,args.fold)
    elif(experimentType=="Cross"):
        train_dataset = CrossSocialTrainDataset('index.csv',experimentType, args.social)
        test_dataset=CrossSocialTestDataset('index.csv',experimentType, args.social)

        

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True,
                                                num_workers=24)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False,
                                               num_workers=24)
    model=ResProj()
    model.to(DEVICE)
    
    if(experimentType=="Multi"):
        model.load_state_dict(torch.load('weights/AllSocial1679735013.7286596.pt'), strict=True)
        
    
    elif(experimentType=="Single"):
        
        if(args.social=="Twitter"):    
            model.load_state_dict(torch.load('weights/OnlyTwitter1679733577.4387584.pt'), strict=True)
        elif(args.social=="Facebook"):
            model.load_state_dict(torch.load('weights/OnlyFacebook1679679765.1768184.pt'), strict=True)
        elif(args.social=="Instagram"):
            model.load_state_dict(torch.load('weights/OnlyInstagram1679680182.1561003.pt'), strict=True)
        else:
            print("This is not one of our studied social")
    elif(experimentType=="Cross"):
        
        if(args.social=="Twitter"):
            model.load_state_dict(torch.load('weights/CrossSocial-Twitter2023-06-27 15:03:18.pt'), strict=True)            
        elif(args.social=="Facebook"):
            model.load_state_dict(torch.load('weights/CrossSocial-Facebook1679900706.2676375.pt'), strict=True)
        elif(args.social=="Instagram"):
            model.load_state_dict(torch.load('weights/CrossSocial-Instagram2023-06-27 11:57:42.pt'), strict=True)
           
        else:
            print("This is not one of our studied social")
        
    
    model.eval()
    with torch.no_grad():
        y_true = []
        y_pred = []
        for data in test_loader:
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


    print("Accuracy Score:" + str(metrics.accuracy_score(y_true, y_pred, normalize=True)))
    confusion=metrics.confusion_matrix(y_true, y_pred, normalize='true')

    print("confusion matrix diagonal:")
    print(np.diag(confusion))

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)