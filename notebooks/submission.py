#%%

import os
import numpy as np
import pandas as pd
import torch

import torch._dynamo
torch._dynamo.disable()

import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import inspect

nn_except = ["Module", "Parameter", "Sequential"]
for m in inspect.getmembers(nn):
    if not m[0] in nn_except and m[0][0:2] != "__":
        delattr(nn, m[0])

# 再現性
seed = 1234
torch.manual_seed(seed)
np.random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# データ読み込み（修正不要）
x_train = np.load('../data/x_train.npy')
t_train = np.load('../data/y_train.npy')
x_test = np.load('../data/x_test.npy')

class train_dataset(torch.utils.data.Dataset):
    def __init__(self, x_train, t_train):
        self.x_train = x_train.reshape(-1, 784).astype('float32') / 255
        self.t_train = t_train

    def __len__(self):
        return self.x_train.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.x_train[idx], dtype=torch.float), torch.tensor(self.t_train[idx], dtype=torch.long)

class test_dataset(torch.utils.data.Dataset):
    def __init__(self, x_test):
        self.x_test = x_test.reshape(-1, 784).astype('float32') / 255

    def __len__(self):
        return self.x_test.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.x_test[idx], dtype=torch.float)

trainval_data = train_dataset(x_train, t_train)
test_data = test_dataset(x_test)

# --- activation functions --- 
def relu(x):
    return torch.where(x > 0, x, torch.zeros_like(x))

def softmax(x):
    x_max = x.max(dim=1, keepdim=True)[0]
    exp_x = torch.exp(x - x_max)
    return exp_x / exp_x.sum(dim=1, keepdim=True)

# Dense層の定義
class Dense(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Parameter(torch.randn(in_dim, out_dim) * 0.01)
        self.b = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        return x.matmul(self.W) + self.b

# MLPの定義
class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.l1 = Dense(in_dim, hid_dim)
        self.l2 = Dense(hid_dim, out_dim)

    def forward(self, x):
        h = relu(self.l1(x))
        y = softmax(self.l2(h))
        return y

# ハイパーパラメータ
in_dim = 784
hid_dim = 200
out_dim = 10
lr = 0.001
n_epochs = 10
batch_size = 32

# データローダー
val_size = 10000
train_size = len(trainval_data) - val_size
train_data, val_data = torch.utils.data.random_split(trainval_data, [train_size, val_size])
dataloader_train = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
dataloader_valid = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)
dataloader_test  = torch.utils.data.DataLoader(test_data,  batch_size=batch_size, shuffle=False)

# モデル・オプティマイザ
mlp = MLP(in_dim, hid_dim, out_dim).to(device)
# optimizer = optim.SGD(mlp.parameters(), lr=0.1)
optimizer = optim.Adam(mlp.parameters(), lr=lr)

# 訓練と検証ループ
for epoch in range(n_epochs):
    losses_train = []
    losses_valid = []
    train_num = train_true_num = 0
    valid_num = valid_true_num = 0

    # 訓練
    mlp.train()
    for x, t in dataloader_train:
        x, t = x.to(device), t.to(device)
        optimizer.zero_grad()
        y = mlp(x)
        # クロスエントロピー
        loss = -torch.log(y[torch.arange(x.size(0)), t]).mean()
        loss.backward()
        optimizer.step()
        losses_train.append(loss.item())

        pred = y.argmax(1).cpu()
        correct = (pred == t.cpu()).sum().item()
        train_true_num += correct
        train_num += x.size(0)

    # 検証
    mlp.eval()
    with torch.no_grad():
        for x, t in dataloader_valid:
            x, t = x.to(device), t.to(device)
            y = mlp(x)
            loss = -torch.log(y[torch.arange(x.size(0)), t]).mean()
            losses_valid.append(loss.item())

            pred = y.argmax(1).cpu()
            correct = (pred == t.cpu()).sum().item()
            valid_true_num += correct
            valid_num += x.size(0)

    print(
        f'EPOCH: {epoch}, '
        f'Train [Loss: {np.mean(losses_train):.3f}, Accuracy: {train_true_num/train_num:.3f}], '
        f'Valid [Loss: {np.mean(losses_valid):.3f}, Accuracy: {valid_true_num/valid_num:.3f}]'
    )

# テストデータで予測
mlp.eval()
t_pred = []
with torch.no_grad():
    for x in dataloader_test:
        x = x.to(device)
        y = mlp(x)
        pred = y.argmax(1).tolist()
        t_pred.extend(pred)

datetime = pd.to_datetime('today').strftime('%Y%m%d%H%M')
submission = pd.Series(t_pred, name='label')
submission.to_csv(
    f'sub/{datetime}.csv',
    header=True, index_label='id'
)
