import numpy as np
from collections import defaultdict
from collections import Counter
import torch
from numpy.linalg import norm


def distancecalc(target,median):
    res=[]
    for i in range(len(median)):
        if median[i]!='null':
            cos_sim = np.dot(target, median[i])/(norm(target)*norm(median[i]))        
            res.append(cos_sim)
            #print(max(res))
        else:
            res.append(np.NINF)
    return(res.index(max(res)),max(res))


def validationRes(model,test_loader,median):
    DEVICE='cuda'
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
                index,res=distancecalc(outputs[j].detach().numpy(),median)
                diction[imagename[j]].append(index)
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
        
    return(y_true,y_pred)  

def validationResTrain(model,train_loader,test_loader):
    DEVICE='cuda'
    model.eval()
    result=[ [] for _ in range(21)]
    with torch.no_grad():
        for data in train_loader:

            inputs, labels = data          
            
            labels=labels.to(DEVICE)
            inputs=inputs.to(DEVICE)
            
            outputs = model(inputs)
            
            for j in range(len(labels)):              
                result[labels[j].item()].append(outputs[j].cpu().detach().numpy())  
        
        median=[]
        for i in range(len(result)):
            if result[i]!=[]:
                median.append(np.median(result[i],axis=0))  
            else:
                median.append('null')

            
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
                    index,res=distancecalc(outputs[j].detach().numpy(),median)
                    diction[imagename[j]].append(index)
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
        
        return(y_true,y_pred)  