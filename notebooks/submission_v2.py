#%%
import os
import numpy as np
import pandas as pd
import torch

import torch._dynamo
torch._dynamo.disable()

import torch.nn as nn
import torch.optim as optim
import inspect

nn_except = ["Module", "Parameter", "Sequential"]
for m in inspect.getmembers(nn):
    if not m[0] in nn_except and m[0][:2] != "__":
        delattr(nn, m[0])

import importlib
nn.parameter = importlib.import_module("torch.nn.parameter")

seed = 1234
torch.manual_seed(seed)
np.random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ガンマ補正
def gamma_correction(x, gamma):
    return np.power(x, 1.0 / gamma)


x_train = np.load('../data/x_train.npy')
t_train = np.load('../data/y_train.npy')
x_test = np.load('../data/x_test.npy')

# 1) [0,255]→[0,1]
x_train = x_train.astype('float32') / 255
x_test  = x_test.astype('float32')  / 255

# 2) ガンマ補正 (γ=1.7)
gamma = 1.7
x_train = gamma_correction(x_train, gamma)
x_test  = gamma_correction(x_test,  gamma)

# 3) 平均・分散で標準化
mean = x_train.mean(axis=0)
std  = x_train.std(axis=0) + 1e-7
x_train = (x_train - mean) / std
x_test  = (x_test  - mean) / std


# ---------- Dataset / DataLoader ----------
class train_dataset(torch.utils.data.Dataset):
    def __init__(self, x, t):
        self.x = x.reshape(-1, 784).astype('float32')
        self.t = t

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx], dtype=torch.float32), \
               torch.tensor(self.t[idx], dtype=torch.long)

class test_dataset(torch.utils.data.Dataset):
    def __init__(self, x):
        self.x = x.reshape(-1, 784).astype('float32')

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx], dtype=torch.float32)

trainval = train_dataset(x_train, t_train)
testdata = test_dataset(x_test)

# split train / valid
val_size = 10000
train_size = len(trainval) - val_size
train_data, valid_data = torch.utils.data.random_split(trainval, [train_size, val_size])

batch_size = 32
loader_train = torch.utils.data.DataLoader(train_data,  batch_size=batch_size, shuffle=True)
loader_valid = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False)
loader_test  = torch.utils.data.DataLoader(testdata,  batch_size=batch_size, shuffle=False)


# ---------- 自作 BatchNorm1d ----------
class BatchNorm1d(nn.Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta  = nn.Parameter(torch.zeros(dim))
        self.register_buffer('running_mean', torch.zeros(dim))
        self.register_buffer('running_var',  torch.ones(dim))
        self.eps = eps
        self.momentum = momentum

    def forward(self, x):
        if self.training:
            mu  = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            # update running stats
            self.running_mean = (1-self.momentum)*self.running_mean + self.momentum*mu.detach()
            self.running_var  = (1-self.momentum)*self.running_var  + self.momentum*var.detach()
            x_hat = (x - mu) / torch.sqrt(var + self.eps)
        else:
            x_hat = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        return self.gamma * x_hat + self.beta


# ---------- MLP の定義 ----------
def relu(x):
    return torch.where(x > 0, x, torch.zeros_like(x))

def softmax(x):
    x_max = x.max(dim=1, keepdim=True)[0]
    exp_x = torch.exp(x - x_max)
    return exp_x / exp_x.sum(dim=1, keepdim=True)

class Dropout(nn.Module):
    def __init__(self, p=0.3):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.training:
            # keep_prob = 1‑p、mask で無効化
            mask = (torch.rand_like(x) > self.p).float()
            return mask * x / (1 - self.p)
        return x

class Dense(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # self.W = nn.Parameter(torch.randn(in_dim, out_dim) * 0.01)
        self.W = nn.Parameter(torch.randn(in_dim, out_dim) * (2 / in_dim) ** 0.5)  # ←He 初期化
        self.b = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        return x.matmul(self.W) + self.b

class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super().__init__()
        self.l1 = Dense(in_dim, hid_dim)
        self.bn = BatchNorm1d(hid_dim)
        self.l2 = Dense(hid_dim, out_dim)

    def forward(self, x):
        h = relu(self.bn(self.l1(x)))
        y = softmax(self.l2(h))
        return y

class DeepMLP(nn.Module):
    def __init__(self, in_dim, hid1, hid2, hid3, out_dim, p_drop=0.3):
        super().__init__()

        self.fc1 = Dense(in_dim, hid1)
        self.bn1 = BatchNorm1d(hid1)
        self.do1 = Dropout(p_drop)

        self.fc2 = Dense(hid1, hid2)
        self.bn2 = BatchNorm1d(hid2)
        self.do2 = Dropout(p_drop)

        self.fc3 = Dense(hid2, hid3)
        self.bn3 = BatchNorm1d(hid3)
        self.do3 = Dropout(p_drop)

        self.fc4 = Dense(hid3, out_dim)

    def forward(self, x):
        x = relu(self.bn1(self.fc1(x)))
        x = self.do1(x)
        x = relu(self.bn2(self.fc2(x)))
        x = self.do2(x)
        x = relu(self.bn3(self.fc3(x)))
        x = self.do3(x)
        y = softmax(self.fc4(x))
        return y

# ---------- ハイパーパラメータグリッドサーチ & アンサンブル用トレーニング ----------
in_dim   = 784
out_dim  = 10
n_epochs = 20
patience = 3

param_grid = {
    'hid_dim': [300],
    'lr': [5e-4]
}

best_models = []
best_acc = 0.0
import math

print("===== Grid Search Start =====")
for hid in param_grid['hid_dim']:
    for lr0 in param_grid['lr']:
        print(f"\n>> Config: hid_dim={hid}, lr={lr0}")
        # model = MLP(in_dim, hid, out_dim).to(device)
        model = DeepMLP(
            in_dim=in_dim,
            hid1=512,
            hid2=512,
            hid3=256,
            out_dim=out_dim,
            p_drop=0.3
        ).to(device)

        opt   = optim.Adam(model.parameters(), lr=lr0)

        best_val_acc = 0.0
        no_improve   = 0

        for epoch in range(n_epochs):
            # Cosine annealing LR
            new_lr = lr0 * 0.5 * (1 + math.cos(math.pi * epoch / n_epochs))
            for pg in opt.param_groups:
                pg['lr'] = new_lr

            # train
            model.train()
            for x, t in loader_train:
                x, t = x.to(device), t.to(device)
                opt.zero_grad()
                y = model(x)
                loss = -torch.log(y[torch.arange(x.size(0)), t]).mean()
                loss.backward()
                opt.step()

            # valid
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for x, t in loader_valid:
                    x, t = x.to(device), t.to(device)
                    y = model(x)
                    preds = y.argmax(1)
                    correct += (preds == t).sum().item()
                    total   += t.size(0)
            val_acc = correct / total

            # ログ出力
            print(f"   Epoch {epoch+1:02d}/{n_epochs:02d}  lr={new_lr:.5f}  val_acc={val_acc:.4f}")

            # early stop 判定
            if val_acc > best_val_acc + 1e-4:
                best_val_acc = val_acc
                best_state   = {k:v.cpu() for k,v in model.state_dict().items()}
                no_improve   = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                print(f"    Early stopping at epoch {epoch+1}")
                break

        # この設定での最良結果を報告
        print(f"<< Best val_acc for hid_dim={hid}, lr={lr0}: {best_val_acc:.4f} >>")

        # 全体ベストと比較・更新
        if best_val_acc > best_acc + 1e-4:
            best_acc    = best_val_acc
            best_models = [best_state]
        elif abs(best_val_acc - best_acc) < 1e-4:
            best_models.append(best_state)

print(f"\n===== Grid Search Done: Best validation accuracy = {best_acc:.4f}, ensemble size = {len(best_models)} =====")


# ---------- テストデータでアンサンブル予測 ----------
ensemble_preds = []
for state in best_models:
    m = DeepMLP(in_dim, list(state['bn.gamma'].shape)[0], out_dim).to(device)
    m.load_state_dict(state)
    m.eval()
    ensemble_preds.append(m)

t_pred = []
with torch.no_grad():
    for x in loader_test:
        x = x.to(device)
        # 各モデルの確率を足し合わせ
        ps = sum(m(x) for m in ensemble_preds)
        avg_p = ps / len(ensemble_preds)
        t_pred.extend(avg_p.argmax(1).cpu().tolist())

datetime = pd.to_datetime('today').strftime('%Y%m%d%H%M')
submission = pd.Series(t_pred, name='label')
submission.to_csv(
    f'sub/{datetime}.csv',
    header=True, index_label='id'
)
