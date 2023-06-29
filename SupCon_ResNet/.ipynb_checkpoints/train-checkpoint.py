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
import time
from datetime import datetime
from pytorch_metric_learning.losses import SupConLoss
from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser()    
    parser.add_argument("--fold", type=int, help="Inserisci fold",required=False)
    parser.add_argument("--social", type=str, help="Inserisci Social",required=False)
    return parser




def main(args):
    
    for prof in common.NOPROFILES:
        print(prof)
        start = time.time()
        train_dataset = UnknownProfileTrainDataset('index.csv',args.fold, prof,args.social)
        test_dataset=UnknownProfileTestDataset('index.csv',args.fold, prof,args.social)
        out_profile_dataset=OnlyUnknownProfileTestDataset('index.csv',args.fold, prof,args.social)
        #TrainForTest=TrainForTest('index.csv',0, 'FoxNews','Facebook')
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True,
                                                    num_workers=24)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False,
                                                   num_workers=24)
        out_profile_loader=torch.utils.data.DataLoader(out_profile_dataset, batch_size=128, shuffle=False,
                                                   num_workers=24)
        DEVICE = 'cuda'

        model=ResProj()
        model.to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), 1e-4)
        criterion = SupConLoss()

        print(f"Training patches for epoch: {len(train_dataset)}")
        print(f"Validation patches for epoch: {len(test_dataset)}")
        ccc=0
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
                test_bar = tqdm(test_loader)
                test_bar.set_description(f"Validating epoch {epoch + 1}")
                model.eval()
                y_true,y_pred=utils.validationResTrain(model,train_loader,test_loader)

                #writer.add_scalar('test_accuracy', balanced_accuracy_score(y_true, y_pred), epoch)
                if(balanced_accuracy_score(y_true, y_pred)>accuracy):
                    accuracy=balanced_accuracy_score(y_true, y_pred)
                    torch.save(model.state_dict(),'No'+prof+str(datetime.fromtimestamp(start).strftime("%Y-%m-%d %H:%M:%S"))+args.social+'.pt')
                    t=0
                    ccc=0
                else:
                    c+=1
                    t+=1
                    if(t>10):
                        print(optimizer.param_groups[0]['lr'])
                        optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']/10
                        print(optimizer.param_groups[0]['lr'])
                        t=0
                    if(ccc>30):
                        return


        print("Training complete!")
        #writer.close()
        return
    
    
if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)