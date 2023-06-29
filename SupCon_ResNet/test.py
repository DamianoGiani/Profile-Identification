import utils
from models import *
from sklearn import metrics
from datasets import *
import argparse
from collections import Counter
import numpy as np
import os
from sklearn.metrics import balanced_accuracy_score,roc_auc_score
import matplotlib.pyplot as plt


def get_parser():
    parser = argparse.ArgumentParser()    
    parser.add_argument("--fold", type=int, help="Inserisci fold",required=False)
    parser.add_argument("--social", type=str, help="Inserisci Social",required=False)
    return parser

def main(args):
    
    DEVICE='cuda'
    df = pd.DataFrame()
    weight=os.listdir('../../Test'+args.social)
    for w in weight:
        if w=='.ipynb_checkpoints':
            print(w)
        else:
            name = w.split('No')[1].split('1')
            name=name[0]
            print(name)
           
            train_dataset = UnknownProfileTrainDataset('index.csv',args.fold, name,args.social)
            test_dataset=UnknownProfileTestDataset('index.csv',args.fold, name,args.social)
            out_profile_dataset=OnlyUnknownProfileTestDataset('index.csv',args.fold, name,args.social)
            TrainForTest_dataset=TrainForTest('index.csv',args.fold, name,args.social)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True,
                                                        num_workers=24)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False,
                                                       num_workers=24)
            out_profile_loader=torch.utils.data.DataLoader(out_profile_dataset, batch_size=128, shuffle=False,
                                                       num_workers=24)
            TrainForTest_loader=torch.utils.data.DataLoader(TrainForTest_dataset, batch_size=64, shuffle=False,
                                                        num_workers=24)


            DEVICE = 'cuda'
            model=ResProj()
            model.to(DEVICE)
            model.load_state_dict(torch.load('../../Test'+args.social+'/'+w))
            optimizer = torch.optim.Adam(model.parameters(), 1e-4)



            
            model.eval()
            result=[ [] for _ in range(21)]
            with torch.no_grad():
                for data in TrainForTest_loader:

                    inputs, labels,_ = data          

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


            y_true,y_pred=utils.validationRes(model,test_loader,median)
            accuracy=metrics.accuracy_score(y_true, y_pred, normalize=True)

            predictedResult=[ [] for _ in range(21)]
            predictedResTUTTIINSIEME=[]
            y_trueRight=[]
            yPred=[]
            with torch.no_grad():
                y_true = []
                y_pred = []
                for data in test_loader:
                    inputs, labels,imagename = data
                    imagelist = list(set(imagename))                
                    diction={}
                    dictionlabel={}
                    for i in imagelist:                    
                        if i not in diction.keys():
                            diction[i] = []
                            dictionlabel[i]=[]




                    inputs = inputs.to(DEVICE)
                    outputs = model(inputs).cpu()

                    for j in range(len(outputs)):
                        index,res=utils.distancecalc(outputs[j].detach().numpy(),median)
                        diction[imagename[j]].append(index)
                        dictionlabel[imagename[j]].append(labels[j].item())


                    for n in diction.keys():
                        count_dict = Counter(diction[n])
                        most_label_pred = count_dict.most_common(1)[0][0]

                        count_dict_label = Counter(dictionlabel[n])
                        label = count_dict_label.most_common(1)[0][0]
                        predictedResTUTTIINSIEME.append(count_dict.most_common(1)[0][1]/len(diction[n]))
                        predictedResult[most_label_pred].append(count_dict.most_common(1)[0][1]/len(diction[n]))  
                        y_trueRight.append(1)
                        if(count_dict.most_common(1)[0][1]/len(diction[n])<0.5):
                            yPred.append(0)
                        else:
                            yPred.append(1)



            outclass=[]
            yPredOUT=[]
            y_trueRightOut=[]
            for data in out_profile_loader:
                inputs, labels,imagename = data
                imagelist = list(set(imagename))                
                diction={}
                dictionlabel={}
                for i in imagelist:
                    if i not in diction.keys():
                        diction[i] = []
                        dictionlabel[i]=[]

                inputs = inputs.to(DEVICE)
                outputs = model(inputs).cpu()

                for j in range(len(outputs)):
                    index,res=utils.distancecalc(outputs[j].detach().numpy(),median)
                    diction[imagename[j]].append(index)
                    dictionlabel[imagename[j]].append(labels[j].item())


                for n in diction.keys():
                    count_dict = Counter(diction[n])
                    most_label_pred = count_dict.most_common(1)[0][0]

                    count_dict_label = Counter(dictionlabel[n])
                    label = count_dict_label.most_common(1)[0][0]

                    outclass.append(count_dict.most_common(1)[0][1]/len(diction[n]))
                    y_trueRightOut.append(0)
                    if(count_dict.most_common(1)[0][1]/len(diction[n])<0.5):
                        yPredOUT.append(0)
                    else:
                        yPredOUT.append(1)

            ALLyPred=[]
            ALLy_trueRight=[]
            ALLy_trueRight=y_trueRight+y_trueRightOut
            ALLyPred=yPred+yPredOUT

            noveltyAccuracy=balanced_accuracy_score(ALLy_trueRight, ALLyPred)    
            predictedResTUTTIINSIEME=predictedResTUTTIINSIEME+outclass
            rocAuc=roc_auc_score(ALLy_trueRight, predictedResTUTTIINSIEME)

            print(name,accuracy,rocAuc,noveltyAccuracy)
            data = {'unknownProfile': [name],
                    'Accuracy score known classes': [str(round(accuracy, 3))],
                   'Roc Auc Novelty detection':[str(round(rocAuc,3))],
                    'Accuracy Novelty detection':[str(round(noveltyAccuracy,3))]                                                   }

            # Create DataFrame
            df1 = pd.DataFrame(data)
            df=df.append(df1)
            ii=np.zeros(21,dtype=int)
            ii[common.PROFILES_IDS[name]]=1
            predictedResult[common.PROFILES_IDS[name]]=outclass

            fig, ax = plt.subplots()

            # create the boxplot
            ax.boxplot(predictedResult)

            # set the x-axis tick labels
            ax.set_xticklabels(ii)

            # set the y-axis label
            ax.set_ylabel('Data')

            # set the title of the plot
            ax.set_title('unknownProfile:'+name)

            # display the plot
            plt.savefig('boxPlotOfNoveltyDetection'+args.social+'/'+name+'.png')
    df.to_csv('Result'+args.social+'.csv',index=False)

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)