import pickle
import common
import argparse
import numpy as np
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

random.seed(42)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, help="Inserisci il tipo di esperimento fra Single,Multi e Cross",required=True)    
    parser.add_argument("--social", type=str, help="Inserisci Social",required=False)
    return parser

def main(args):
    experimentType=args.experiment
    
    if(experimentType=="Cross"):
        with open('datasets/trainSetsDCT.pickle', 'rb') as f:
            trainSets = pickle.load(f)
        with open('datasets/testSetsDCT.pickle', 'rb') as f:
            testSets = pickle.load(f)
        with open('datasets/trainLabelsDCT.pickle', 'rb') as f:
            trainLabels = pickle.load(f) 
        with open('datasets/testLabelsDCT.pickle', 'rb') as f:
            testLabels = pickle.load(f)  
        if(args.social=="Facebook"):
            social=0
        elif(args.social=="Instagram"):
            social=1
        elif(args.social=="Twitter"):
            social=2
        X_train=trainSets[social]
        X_test=testSets[social]
        Y_train = trainLabels[social]
        Y_test = testLabels[social]
    
    elif(experimentType=="Multi"):
        with open('datasets/testSetsDCT.pickle', 'rb') as f:
            train = pickle.load(f)
        with open('datasets/testLabelsDCT.pickle', 'rb') as f:
            trainLabels = pickle.load(f) 
        alltrain=np.concatenate((train[0],train[1],train[2]))
        allLabels=trainLabels[0]+trainLabels[1]+trainLabels[2]
        X_train, X_test, Y_train, Y_test = train_test_split(alltrain, allLabels, test_size = 0.10,random_state=42)
    elif(experimentType=="Single"):
        with open('datasets/testSetsDCT.pickle', 'rb') as f:
            train = pickle.load(f)
        with open('datasets/testLabelsDCT.pickle', 'rb') as f:
            trainLabels = pickle.load(f)  
        if(args.social=="Facebook"):
            social=0
        elif(args.social=="Instagram"):
            social=1
        elif(args.social=="Twitter"):
            social=2
        X_train, X_test, Y_train, Y_test = train_test_split(train[social], trainLabels[social], test_size = 0.10,random_state=42)
        
    

    
    clf = RandomForestClassifier(random_state = 42)
    clf.fit(X_train, Y_train)
    pred=clf.predict(X_test)
    val_acc_Social = accuracy_score(Y_test, pred)
    print("validation accuracy: "+str(val_acc_Social))
    confusionSocial=confusion_matrix(Y_test, pred,normalize='true',labels=common.PROFILES)
    print("Confusion matrix: ")
    print(np.diagonal(confusionSocial))

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)