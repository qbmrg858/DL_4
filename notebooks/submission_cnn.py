#%%
import os, math, yaml, inspect, importlib, numpy as np, pandas as pd, torch
import torch._dynamo; torch._dynamo.disable()

import torch.nn as nn
import torch.optim as optim

# ---------- 1. CONFIG -------------------------------------------------------
with open("config_cnn.yaml", "r") as f:
    C = yaml.safe_load(f)

seed = C["seed"]
torch.manual_seed(seed); np.random.seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- 2. limit nn API -------------------------------------------------
_keep = ["Module", "Parameter", "Sequential"]
for n, _ in inspect.getmembers(nn):
    if n not in _keep and not n.startswith("__"):
        delattr(nn, n)
nn.parameter = importlib.import_module("torch.nn.parameter")

# ---------- 3. util ---------------------------------------------------------
def gamma_cor(x, g): return np.power(x, 1.0 / g)
def relu(x): return torch.where(x > 0, x, torch.zeros_like(x))
def softmax(x):
    m = x.max(1, keepdim=True)[0]
    e = torch.exp(x - m)
    return e / e.sum(1, keepdim=True)

# ---------- 4. data ---------------------------------------------------------
x_tr = np.load("../data/x_train.npy").astype("float32") / 255
y_tr = np.load("../data/y_train.npy")
x_te = np.load("../data/x_test.npy").astype("float32") / 255

x_tr = gamma_cor(x_tr, C["data"]["gamma"])
x_te = gamma_cor(x_te, C["data"]["gamma"])
mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-7
x_tr = (x_tr - mu) / sd
x_te = (x_te - mu) / sd

class DS(torch.utils.data.Dataset):
    def __init__(s, x, t=None):
        s.x = x.reshape(-1, 1, 28, 28).astype("float32")
        s.t = t
    def __len__(s): return len(s.x)
    def __getitem__(s, i):
        if s.t is None:
            return torch.tensor(s.x[i])
        return torch.tensor(s.x[i]), torch.tensor(s.t[i], dtype=torch.long)

full_ds, test_ds = DS(x_tr, y_tr), DS(x_te)
val_sz = len(full_ds) // C["data"]["val_split"]
tr_ds, va_ds = torch.utils.data.random_split(full_ds, [len(full_ds)-val_sz, val_sz])

B = C["train"]["batch_size"]
ldr_tr = torch.utils.data.DataLoader(tr_ds, B, shuffle=True)
ldr_va = torch.utils.data.DataLoader(va_ds, B, shuffle=False)
ldr_te = torch.utils.data.DataLoader(test_ds, B, shuffle=False)

# ---------- 5. primitive layers --------------------------------------------
class BN(nn.Module):
    def __init__(s, d, eps=1e-5, mom=0.1):
        super().__init__()
        s.g = nn.Parameter(torch.ones(d)); s.b = nn.Parameter(torch.zeros(d))
        s.register_buffer("rm", torch.zeros(d)); s.register_buffer("rv", torch.ones(d))
        s.eps, s.mom = eps, mom
    def _stat(s, x):
        if x.dim()==4: return x.mean((0,2,3)), x.var((0,2,3), unbiased=False)
        return x.mean(0), x.var(0, unbiased=False)
    def forward(s, x):
        if s.training:
            mu, vr = s._stat(x)
            s.rm = (1-s.mom)*s.rm + s.mom*mu.detach()
            s.rv = (1-s.mom)*s.rv + s.mom*vr.detach()
        else:
            mu, vr = s.rm, s.rv
        if x.dim()==4: mu, vr = mu[None,:,None,None], vr[None,:,None,None]
        return s.g.view(1,-1,1,1)*((x-mu)/torch.sqrt(vr+s.eps)) + s.b.view(1,-1,1,1)

class DO(nn.Module):
    def __init__(s, p): super().__init__(); s.p=p
    def forward(s, x):
        if s.training:
            mask = (torch.rand_like(x) > s.p).float()
            return mask * x / (1 - s.p)
        return x

class Dense(nn.Module):
    def __init__(s, i, o):
        super().__init__()
        s.W = nn.Parameter(torch.randn(i, o) * (2 / i) ** 0.5)
        s.b = nn.Parameter(torch.zeros(o))
    def forward(s, x): return x @ s.W + s.b

# ---------- 6. Conv2D & MaxPool2D (functional 不使用) -----------------------
class Conv2D(nn.Module):
    def __init__(s, in_c, out_c, k=3, stride=1, pad=1):
        super().__init__()
        s.k, s.stride, s.pad = k, stride, pad
        s.W = nn.Parameter(torch.randn(out_c, in_c, k, k) * (2/(in_c*k*k))**0.5)
        s.b = nn.Parameter(torch.zeros(out_c))
    def forward(s, x):                       # x: (B,C,H,W)
        B,C,H,W = x.shape
        if s.pad:
            x_pad = torch.zeros(B, C, H+2*s.pad, W+2*s.pad, device=x.device, dtype=x.dtype)
            x_pad[:,:, s.pad:s.pad+H, s.pad:s.pad+W] = x
        else:
            x_pad = x
        k, st = s.k, s.stride
        out_H = (x_pad.size(2) - k)//st + 1
        out_W = (x_pad.size(3) - k)//st + 1
        # unfold via Tensor.unfold
        patches = x_pad.unfold(2, k, st).unfold(3, k, st)  # (B,C,out_H,out_W,k,k)
        patches = patches.contiguous().view(B, C, out_H, out_W, -1)  # flatten k*k
        W_flat = s.W.view(s.W.size(0), -1)                              # (out_c, C*k*k)
        # (B,out_H,out_W,out_c)
        out = (patches.permute(0,2,3,1,4).reshape(-1, C*k*k) @ W_flat.t()) + s.b
        out = out.view(B, out_H, out_W, -1).permute(0,3,1,2).contiguous()
        return out

class MaxPool2D(nn.Module):
    def __init__(s, k=2, stride=2):
        super().__init__(); s.k, s.stride = k, stride
    def forward(s, x):                        # x: (B,C,H,W)
        k, st = s.k, s.stride
        B,C,H,W = x.shape
        out_H = (H - k)//st + 1
        out_W = (W - k)//st + 1
        patches = x.unfold(2, k, st).unfold(3, k, st)  # (B,C,out_H,out_W,k,k)
        patches = patches.contiguous().view(B, C, out_H, out_W, -1)
        out, _ = patches.max(-1)
        return out

# ---------- 7. SimpleCNN ----------------------------------------------------
class SimpleCNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        p = cfg["model"]["dropout"]
        self.c1 = Conv2D(1, 32)
        self.b1 = BN(32)
        self.c2 = Conv2D(32, 32)
        self.b2 = BN(32)
        self.p1 = MaxPool2D()

        self.c3 = Conv2D(32, 64)
        self.b3 = BN(64)
        self.c4 = Conv2D(64, 64)
        self.b4 = BN(64)
        self.p2 = MaxPool2D()

        self.do = DO(p)
        self.fc = Dense(64, C["model"]["out_dim"])

    def forward(self, x):
        x = relu(self.b1(self.c1(x)))
        x = relu(self.b2(self.c2(x)))
        x = self.p1(x)

        x = relu(self.b3(self.c3(x)))
        x = relu(self.b4(self.c4(x)))
        x = self.p2(x)

        x = x.mean((-2, -1))      # GAP
        x = self.do(x)
        return softmax(self.fc(x))

# ---------- 8. loss utils ---------------------------------------------------
num_cls = C["model"]["out_dim"]
def one_hot(t):
    t = t.long()
    o = torch.zeros(t.size(0), num_cls, device=t.device)
    o.scatter_(1, t.unsqueeze(1), 1.0)
    return o
def loss_soft_ce(p, q): return -(q * torch.log(p + 1e-12)).sum(1).mean()

# ---------- 9. train --------------------------------------------------------
M = SimpleCNN(C).to(device)
opt = optim.Adam(M.parameters(), lr=C["train"]["lr"])

n_ep = C["train"]["n_epochs"]; patience = C["train"]["patience"]
swa_after = int(n_ep * (1 - C["train"]["swa_pct"]))

best_acc, wait, swa_state, swa_n = 0.0, 0, None, 0

for ep in range(n_ep):
    if C["train"]["lr_schedule"] == "cosine":
        lr = C["train"]["lr"] * 0.5 * (1 + math.cos(math.pi * ep / n_ep))
        for g in opt.param_groups: g["lr"] = lr

    # ---- TRAIN (MixUp+LS) ----
    M.train()
    alpha = C["augment"]["mixup_alpha"]; eps_ls = C["augment"]["label_smooth_eps"]
    for xb, tb in ldr_tr:
        xb, tb = xb.to(device), tb.to(device)
        lam = np.random.beta(alpha, alpha)
        idx = torch.randperm(xb.size(0), device=device)
        x_mix = lam * xb + (1 - lam) * xb[idx]
        t_mix = lam * one_hot(tb) + (1 - lam) * one_hot(tb[idx])
        t_soft = (1 - eps_ls) * t_mix + eps_ls / num_cls
        opt.zero_grad(); loss = loss_soft_ce(M(x_mix), t_soft); loss.backward(); opt.step()

    # ---- SWA collect ----
    if ep >= swa_after:
        if swa_state is None:
            swa_state = {k: v.detach().cpu().clone() for k, v in M.state_dict().items()}
        else:
            for k in swa_state:
                swa_state[k] = (swa_state[k] * swa_n + M.state_dict()[k].detach().cpu()) / (swa_n + 1)
        swa_n += 1

    # ---- VALID ----
    M.eval(); corr = tot = 0
    with torch.no_grad():
        for xb, tb in ldr_va:
            pred = M(xb.to(device)).argmax(1).cpu()
            corr += (pred == tb).sum().item(); tot += tb.size(0)
    acc = corr / tot
    print(f"Epoch {ep+1}/{n_ep} val_acc={acc:.4f}")

    if acc > best_acc + 1e-4:
        best_acc, best_state = acc, {k: v.cpu() for k, v in M.state_dict().items()}
        wait = 0
    else:
        wait += 1
    if wait >= patience:
        print("Early stopping."); break

# ---- SWA or best ----
M.load_state_dict(swa_state if swa_state is not None else best_state)

# ---------- 10. test --------------------------------------------------------
M.eval(); preds=[]
with torch.no_grad():
    for xb in ldr_te:
        preds.extend(M(xb.to(device)).argmax(1).cpu().tolist())

os.makedirs("sub", exist_ok=True)
ts = pd.Timestamp("now").strftime("%Y%m%d%H%M")
fn_csv = f"sub/{ts}_cnn_{best_acc:.4f}.csv"
fn_yaml = fn_csv.replace(".csv", ".yaml")
pd.Series(preds, name="label").to_csv(fn_csv, header=True, index_label="id")
with open(fn_yaml, "w") as f: yaml.safe_dump(C, f, sort_keys=False)
print("saved:", fn_csv)
