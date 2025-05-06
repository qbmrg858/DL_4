#%%
#----------------------------------------------------------------------------
import os, math, importlib, inspect
import numpy as np, pandas as pd, torch
import torch._dynamo;  torch._dynamo.disable()

import torch.nn as nn
import torch.optim as optim

# --- nn 高レベル API を除去 -------------------------------------------------
_keep = ["Module", "Parameter", "Sequential"]
for n,_ in inspect.getmembers(nn):
    if n not in _keep and not n.startswith("__"):
        delattr(nn, n)
nn.parameter = importlib.import_module("torch.nn.parameter")

# --- ユーティリティ ---------------------------------------------------------
seed = 1234
torch.manual_seed(seed); np.random.seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def gamma_correction(x,g): return np.power(x,1.0/g)
def relu(x): return torch.where(x>0,x,torch.zeros_like(x))
def softmax(x):
    m=x.max(1,keepdim=True)[0]; e=torch.exp(x-m); return e/e.sum(1,keepdim=True)

# --- データロード ------------------------------------------------------------
x_train = np.load("../data/x_train.npy").astype("float32")/255
y_train = np.load("../data/y_train.npy")
x_test  = np.load("../data/x_test.npy" ).astype("float32")/255

x_train = gamma_correction(x_train,1.7); x_test = gamma_correction(x_test,1.7)
mean,std = x_train.mean(0), x_train.std(0)+1e-7
x_train = (x_train-mean)/std;  x_test=(x_test-mean)/std

class TrainDS(torch.utils.data.Dataset):
    def __init__(s,x,t): s.x=x.reshape(-1,784); s.t=t
    def __len__(s): return len(s.x)
    def __getitem__(s,i): return torch.tensor(s.x[i]), torch.tensor(s.t[i])

class TestDS(torch.utils.data.Dataset):
    def __init__(s,x): s.x=x.reshape(-1,784)
    def __len__(s): return len(s.x)
    def __getitem__(s,i): return torch.tensor(s.x[i])

full_ds, test_ds = TrainDS(x_train,y_train), TestDS(x_test)
val_sz=len(full_ds)//6
train_ds,val_ds=torch.utils.data.random_split(full_ds,[len(full_ds)-val_sz,val_sz])

batch_size=32
ldr_tr = torch.utils.data.DataLoader(train_ds,batch_size,shuffle=True)
ldr_va = torch.utils.data.DataLoader(val_ds,  batch_size,shuffle=False)
ldr_te = torch.utils.data.DataLoader(test_ds, batch_size,shuffle=False)

# --- レイヤ ------------------------------------------------------------------
class BatchNorm1d(nn.Module):
    def __init__(s,d,eps=1e-5,mom=0.1):
        super().__init__(); s.g=nn.Parameter(torch.ones(d)); s.b=nn.Parameter(torch.zeros(d))
        s.register_buffer("rm",torch.zeros(d)); s.register_buffer("rv",torch.ones(d))
        s.eps, s.mom = eps,mom
    def forward(s,x):
        if s.training:
            mu,vr=x.mean(0),x.var(0,unbiased=False)
            s.rm = (1-s.mom)*s.rm + s.mom*mu.detach()
            s.rv = (1-s.mom)*s.rv + s.mom*vr.detach()
            x=(x-mu)/torch.sqrt(vr+s.eps)
        else:
            x=(x-s.rm)/torch.sqrt(s.rv+s.eps)
        return s.g*x+s.b

class Dropout(nn.Module):
    def __init__(s,p): super().__init__(); s.p=p
    def forward(s,x):
        if s.training:
            m=(torch.rand_like(x)>s.p).float()
            return m*x/(1-s.p)
        return x

class Dense(nn.Module):
    def __init__(s,i,o):
        super().__init__(); s.W=nn.Parameter(torch.randn(i,o)*(2/i)**0.5); s.b=nn.Parameter(torch.zeros(o))
    def forward(s,x): return x@s.W + s.b

class DeepMLP(nn.Module):
    def __init__(self, in_d, h1, h2, h3, out_d, p=0.3):
        super().__init__()
        self.f1, self.f2, self.f3, self.f4 = (
            Dense(in_d, h1), Dense(h1, h2), Dense(h2, h3), Dense(h3, out_d)
        )
        self.b1, self.b2, self.b3 = BatchNorm1d(h1), BatchNorm1d(h2), BatchNorm1d(h3)
        self.d1 = Dropout(p)                          # ★ 個別生成
        self.d2 = Dropout(p)
        self.d3 = Dropout(p)

    def forward(self, x):
        x = relu(self.b1(self.f1(x))); x = self.d1(x)  # ★ self.d1
        x = relu(self.b2(self.f2(x))); x = self.d2(x)  # ★ self.d2
        x = relu(self.b3(self.f3(x))); x = self.d3(x)  # ★ self.d3
        return softmax(self.f4(x))


# --- 損失 (Label Smoothing + soft target 対応) -------------------------------
def loss_soft_ce(pred, target):          # pred: (B,C) prob, target: (B,C) soft
    return -(target*torch.log(pred+1e-12)).sum(1).mean()

def one_hot(labels, num_cls=10):
    labels = labels.long()                           # ★ 追加
    o = torch.zeros(labels.size(0), num_cls, device=labels.device)
    o.scatter_(1, labels.unsqueeze(1), 1.0)
    return o


eps_ls=0.1   # label‑smoothing ε
alpha_mu=0.2 # MixUp α
num_cls=10

# --- ハイパラ ---------------------------------------------------------------
lr0=5e-4; n_epochs=20; patience=3; swa_start=int(n_epochs*0.75)

best_acc, best_state = 0.0, None
model = DeepMLP(784,512,512,256,10,0.3).to(device)
opt   = optim.Adam(model.parameters(), lr=lr0)

# SWA 用
swa_state, swa_n = None, 0

for ep in range(n_epochs):
    new_lr=lr0*0.5*(1+math.cos(math.pi*ep/n_epochs))
    for g in opt.param_groups: g['lr']=new_lr

    # ---- TRAIN (MixUp) ----
    model.train()
    for xb, tb in ldr_tr:
        xb,tb=xb.to(device),tb.to(device)
        # MixUp
        lam=np.random.beta(alpha_mu,alpha_mu)
        idx=torch.randperm(xb.size(0),device=device)
        x_mix = lam*xb + (1-lam)*xb[idx]
        t_one = one_hot(tb,num_cls); t_one_mix = lam*t_one + (1-lam)*one_hot(tb[idx],num_cls)
        # Label Smoothing
        t_soft = (1-eps_ls)*t_one_mix + eps_ls/num_cls
        # FW / BW
        opt.zero_grad()
        y_pred=model(x_mix)
        loss=loss_soft_ce(y_pred,t_soft)
        loss.backward(); opt.step()

    # ---- SWA accumulate after swa_start ----
    if ep >= swa_start:
        if swa_state is None:
            swa_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            for k in swa_state:
                swa_state[k]=swa_state[k]*swa_n/(swa_n+1)+model.state_dict()[k].detach().cpu()/(swa_n+1)
        swa_n+=1

    # ---- VALID ----
    model.eval(); correct=tot=0
    with torch.no_grad():
        for xb,tb in ldr_va:
            y=model(xb.to(device))
            pred=y.argmax(1).cpu()
            correct+=(pred==tb).sum().item(); tot+=tb.size(0)
    val_acc=correct/tot
    print(f"Epoch {ep+1:02d}/{n_epochs}  lr={new_lr:.5f}  val_acc={val_acc:.4f}")

    if val_acc>best_acc+1e-4:
        best_acc,val_state=val_acc,{k:v.cpu() for k,v in model.state_dict().items()}
        wait=0
    else:
        wait+=1
    if wait>=patience:
        print("Early stopping."); break

# ---- SWA param に置換 ----
if swa_state is not None:
    print(f"SWA applied over {swa_n} epoch(s)")
    model.load_state_dict(swa_state)
else:
    model.load_state_dict(val_state)

# ---- TEST 予測 ----
model.eval(); preds=[]
with torch.no_grad():
    for xb in ldr_te:
        y=model(xb.to(device)); preds.extend(y.argmax(1).cpu().tolist())

os.makedirs("sub",exist_ok=True)
fname=f"sub/{pd.Timestamp('now').strftime('%Y%m%d%H%M')}.csv"
pd.Series(preds,name="label").to_csv(fname,header=True,index_label="id")
print("saved:",fname)
