from argparse import ArgumentParser
from sklearn import preprocessing
import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pandas as pd
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn import metrics
from sklearn.model_selection import KFold
import numpy as np
from pytorch_lightning.callbacks import LearningRateMonitor
import pickle
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor

seed = 42
torch.manual_seed(seed)
Dataset = ""

# batch the training dataset
# prepare dataset
class BlockChainDataset(Dataset):
    def __init__(self, data, label):
        self.len = data.shape[0]
        self.x_data = torch.from_numpy(data).type(torch.float32)
        self.y_data = torch.from_numpy(label).type(torch.float32)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

class Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.droprate = 0.1
        if Dataset != "HFBTP":
            self.layer = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(True),
            nn.Dropout(self.droprate),
            nn.Linear(64, 8),
            nn.Dropout(self.droprate),
            nn.ReLU(True),
            nn.Linear(8, 1)
        )
        else:
            self.layer = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(True),
            nn.Dropout(self.droprate),
            nn.Linear(64, 8),
            nn.Dropout(self.droprate),
            nn.ReLU(True),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        out = self.layer(x)
        return out

class LitClassifier(pl.LightningModule):
    def __init__(self, backbone, learning_rate=1e-3,GradientBoosting=None,RandomForest=None,Bagging=None,KNeighbors=None):
        super().__init__()
        self.learning_rate=learning_rate
        self.save_hyperparameters()
        self.backbone = backbone
        self.criterion = nn.MSELoss()

        self.GradientBoosting =GradientBoosting
        self.RandomForest=RandomForest
        self.Bagging=Bagging
        self.KNeighbors=KNeighbors
    def forward(self, x):
        embedding = self.backbone(x)
        return embedding

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = self.criterion(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = self.criterion(y_hat, y)
        self.log('valid_loss', loss, on_step=True)


    def evaluate(self, batch, stage=None):
        x, y = batch
        y_hat = self.backbone(x)
        y_hat=torch.cat((y_hat,torch.reshape(self.GradientBoosting.predict(x),(-1,1))),dim=1)
        y_hat = torch.cat((y_hat, torch.reshape(self.RandomForest.predict(x), (-1, 1))), dim=1)
        y_hat = torch.cat((y_hat, torch.reshape(self.Bagging.predict(x), (-1, 1))), dim=1)
        y_hat = torch.cat((y_hat, torch.reshape(self.KNeighbors.predict(x), (-1, 1))), dim=1)
        y_hat=torch.reshape(torch.mean(y_hat,dim=1),(-1,1))
        loss = self.criterion(y_hat, y)
        MAE = metrics.mean_absolute_error(y, y_hat)
        RMSE = metrics.mean_squared_error(y, y_hat) ** 0.5
        MAPE = metrics.mean_absolute_percentage_error(y, y_hat)
        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_MAE", MAE, prog_bar=True)
            self.log(f"{stage}_RMSE", RMSE, prog_bar=True)
            self.log(f"{stage}_MAPE", MAPE, prog_bar=True)
        return MAE,RMSE,MAPE

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        MAE,RMSE,MAPE=self.evaluate(batch, "test")
        print('MAE:{MAE} RMSE:{RMSE} MAPE:{MAPE}'.format(MAE=MAE, RMSE=RMSE, MAPE=MAPE))
        return  MAE,RMSE,MAPE

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--dataset', type=str, default="BPD1", help='Indicates which data set to use')
    parser.add_argument('--task', type=str, default="Throughput", help='Indicates which task to perform')
    args = parser.parse_args()
    Dataset = args.dataset
    # ------------
    # data
    # ------------
    raw_data = pd.read_csv('../data/BPD' + str(args.dataset) + '.csv').values
    if args.dataset != "HFBTP":
        X = raw_data[:, :2]
        if args.task == "Throughput":
            Y1 = raw_data[:, 3].reshape((-1, 1))
        elif args.task == "Latency":
            Y1 = raw_data[:, 2].reshape((-1, 1))
    else:
        X = raw_data[:, :3]
        if args.task == "Throughput":
            Y1 = raw_data[:, 3].reshape((-1, 1))
        elif args.task == "Latency":
            Y1 = raw_data[:, 4].reshape((-1, 1))
    results=[]
    KF = KFold(n_splits=5, random_state=seed, shuffle=True)
    i=1
    os.makedirs(os.path.join('Models', str(args.dataset),str(args.task)), exist_ok=True)
    for train_index, test_index in KF.split(X):
        Xtrain1, Xtest1 = X[train_index], X[test_index]
        Ytrain1, Ytest1 = Y1[train_index], Y1[test_index]
        # property scaling
        min_max_scaler1 = preprocessing.MinMaxScaler()
        # Scaling training set data
        Xtrain1_minmax = min_max_scaler1.fit_transform(Xtrain1)
        # Apply the same scaling to the test set data
        Xtest1_minmax = min_max_scaler1.transform(Xtest1)

        train_dataset = BlockChainDataset(Xtrain1_minmax, Ytrain1)
        val_dataset = BlockChainDataset(Xtest1_minmax, Ytest1)
        test_dataset = BlockChainDataset(Xtest1_minmax, Ytest1)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=Ytest1.shape[0], shuffle=False, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=Xtest1.shape[0], shuffle=False, pin_memory=True)

        # ------------
        # modeling
        # ------------
        GradientBoosting = GradientBoostingRegressor()
        GradientBoosting.fit(Xtrain1_minmax, Ytrain1)
        with open(os.path.join('Models', str(args.dataset), str(args.task), 'GradientBoosting' + str(i) + '.pkl'), 'wb') as file:
            pickle.dump(Bagging, file)
            
        RandomForest = RandomForestRegressor()
        RandomForest.fit(Xtrain1_minmax, Ytrain1)
        with open(os.path.join('Models', str(args.dataset), str(args.task), 'RandomForest' + str(i) + '.pkl'), 'wb') as file:
            pickle.dump(RandomForest, file)

        
        Bagging = BaggingRegressor()
        Bagging.fit(Xtrain1_minmax, Ytrain1)
        with open(os.path.join('Models', str(args.dataset), str(args.task), 'Bagging' + str(i) + '.pkl'), 'wb') as file:
            pickle.dump(Bagging, file)

        KNeighbors = KNeighborsRegressor()
        KNeighbors.fit(Xtrain1_minmax, Ytrain1)
        with open(os.path.join('Models', str(args.dataset), str(args.task), 'KNeighbors' + str(i) + '.pkl'), 'wb') as file:
            pickle.dump(KNeighbors, file)

        MLP = Backbone()
        model = LitClassifier(MLP, args.learning_rate,GradientBoosting,RandomForest,Bagging,KNeighbors)
        early_stop_callback = EarlyStopping(monitor="val_MAE", min_delta=0.000, patience=100, verbose=True, mode="min")
        
        # ------------
        # training
        # ------------
        trainer = pl.Trainer(max_epochs=2000, check_val_every_n_epoch=50,
                             callbacks=[early_stop_callback])
        trainer.fit(model, train_loader, val_loader)
        torch.save(MLP.state_dict(), os.path.join('Models',str(args.dataset), str(args.task),'MLP' + str(i) + '.ckpt'))
        
        # ------------
        # testing
        # ------------
        result = trainer.test(model, test_loader)
        result=list(result[0].values())[1:]
        i=i+1
        results.append(result)
        
    results=np.array(results)
    resultpd=pd.DataFrame(np.vstack((np.vstack((results,results.mean(0).reshape((1,-1)))),results.std(0))))
    if args.task=="Latency":
        resultpd.columns=['latency_MAE','latency_RMSE','latency_MAPE']
    if args.task=="Throughput":
        resultpd.columns=['throughput_MAE','throughput_RMSE','throughput_MAPE']

    #save result
    os.makedirs(os.path.join(str(args.dataset), str(args.task)), exist_ok=True)
    resultpd.to_csv(os.path.join(str(args.dataset), str(args.task), 'Result.csv'),index=False)


if __name__ == '__main__':
    cli_main()
