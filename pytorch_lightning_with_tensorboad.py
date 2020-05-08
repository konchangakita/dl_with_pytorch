import torch 
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import Trainer

from sklearn.datasets import load_iris


# Iris データセットの読み込み
x, t = load_iris(return_X_y=True)

x = torch.tensor(x, dtype=torch.float32)
t = torch.tensor(t, dtype=torch.int64)

#DataSetに格納
dataset = torch.utils.data.TensorDataset(x, t)

# datasetの分割
n_train = int(len(dataset) * 0.6)
n_val = int(len(dataset) * 0.2)
n_test = len(dataset) - n_train - n_val

# ランダムに分割する
torch.manual_seed(0)
train, val, test = torch.utils.data.random_split(dataset, [n_train, n_val, n_test])


# 学習データ用クラス
class TrainNet(pl.LightningModule):
    
    @pl.data_loader
    def train_dataloader(self):
        return torch.utils.data.DataLoader(train, self.batch_size, shuffle=True)
    
    def training_step(self, batch, batch_nb):
        x, t = batch
        y = self.forward(x)
        loss = self.lossfun(y, t)
        y_label = torch.argmax(y, dim=1)
        acc = torch.sum(t == y_label) * 1.0 / len(t)
        tensorboard_logs = {'train/train_loss': loss, 'train/train_acc': acc} # tensorboard
        results = {'loss': loss, 'log': tensorboard_logs}
        #results = {'loss': loss}
        return results

    
# 検証データ用クラス
class ValidationNet(pl.LightningModule):

    @pl.data_loader
    def val_dataloader(self):
        return torch.utils.data.DataLoader(val, self.batch_size)

    def validation_step(self, batch, batch_nb):
        x, t = batch
        y = self.forward(x)
        loss = self.lossfun(y, t)
        y_label = torch.argmax(y, dim=1)
        acc = torch.sum(t == y_label) * 1.0 / len(t)
        results = {'val_loss': loss, 'val_acc': acc}
        return results

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        tensorboard_logs = {'val/avg_loss': avg_loss, 'val/avg_acc': avg_acc}
        results = {'val_loss': avg_loss, 'val_acc': avg_acc, 'log': tensorboard_logs}        
        #results = {'val_loss': avg_loss, 'val_acc': avg_acc}
        return results

    
# テストデータ用クラス
class TestNet(pl.LightningModule):

    @pl.data_loader
    def test_dataloader(self):
        return torch.utils.data.DataLoader(test, self.batch_size)

    def test_step(self, batch, batch_nb):
        x, t = batch
        y = self.forward(x)
        loss = self.lossfun(y, t)
        y_label = torch.argmax(y, dim=1)
        acc = torch.sum(t == y_label) * 1.0 / len(t)
        results = {'test_loss': loss, 'test_acc': acc}
        return results

    def test_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        results = {'test_loss': avg_loss, 'test_acc': avg_acc}
        return results

    
# 学習データ、検証データ、テストデータクラスの継承クラス
class Net(TrainNet, ValidationNet, TestNet):
    def __init__(self, input_size=4, hidden_size=4, output_size=3, batch_size=10):
        super(Net, self).__init__()
        self.L1 = nn.Linear(input_size, hidden_size)
        self.L2 = nn.Linear(hidden_size, output_size)
        self.batch_size = batch_size
        
        self.bn = nn.BatchNorm1d(input_size)
        
    def forward(self, x):
        x = self.L1(x)
        x = F.relu(x)
        x = self.L2(x)
        return x
    
    def lossfun(self, y, t):
        return F.cross_entropy(y, t)
        #return nn.CrossEntropyLoss(y, t)
    
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)
    

# 学習に関する一連の流れを実行
net = Net()
trainer = Trainer(max_epochs=10) # 学習用のインスタンス化と学習の
trainer.fit(net)