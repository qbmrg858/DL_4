#%%
import os, math, yaml, inspect, importlib, numpy as np, pandas as pd, torch
import torch._dynamo; torch._dynamo.disable()

import torch.nn as nn
import torch.optim as optim

# ---------- 1. CONFIG -------------------------------------------------------
with open("config.yaml", "r") as f:
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

# def relu(x): return torch.where(x > 0, x, torch.zeros_like(x))
def relu(x, alpha=0.01):
    return torch.where(x > 0, x, alpha * x)

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
        s.x = x.reshape(-1, C["model"]["in_dim"])
        s.t = t
    def __len__(s): return len(s.x)
    def __getitem__(s, i):
        if s.t is None:
            return torch.tensor(s.x[i])
        return torch.tensor(s.x[i]), torch.tensor(s.t[i], dtype=torch.long)

full_ds, test_ds = DS(x_tr, y_tr), DS(x_te)
v_split = C["data"]["val_split"]
val_sz = len(full_ds) // v_split
tr_ds, va_ds = torch.utils.data.random_split(
    full_ds, [len(full_ds) - val_sz, val_sz])

B = C["train"]["batch_size"]
ldr_tr = torch.utils.data.DataLoader(tr_ds, B, shuffle=True)
ldr_va = torch.utils.data.DataLoader(va_ds, B, shuffle=False)
ldr_te = torch.utils.data.DataLoader(test_ds, B, shuffle=False)

# ---------- 5. layers -------------------------------------------------------
class BN(nn.Module):
    def __init__(s, d, eps=1e-5, mom=0.1):
        super().__init__()
        s.g = nn.Parameter(torch.ones(d)); s.b = nn.Parameter(torch.zeros(d))
        s.register_buffer("rm", torch.zeros(d)); s.register_buffer("rv", torch.ones(d))
        s.eps, s.mom = eps, mom
    def forward(s, x):
        if s.training:
            mu, vr = x.mean(0), x.var(0, unbiased=False)
            s.rm = (1 - s.mom) * s.rm + s.mom * mu.detach()
            s.rv = (1 - s.mom) * s.rv + s.mom * vr.detach()
            x = (x - mu) / torch.sqrt(vr + s.eps)
        else:
            x = (x - s.rm) / torch.sqrt(s.rv + s.eps)
        return s.g * x + s.b

class DO(nn.Module):
    def __init__(s, p): super().__init__(); s.p = p
    def forward(s, x):
        if s.training:
            m = (torch.rand_like(x) > s.p).float()
            return m * x / (1 - s.p)
        return x

class Dense(nn.Module):
    def __init__(s, i, o):
        super().__init__()
        s.W = nn.Parameter(torch.randn(i, o) * (2 / i) ** 0.5)
        s.b = nn.Parameter(torch.zeros(o))
    def forward(s, x): return x @ s.W + s.b

# ---------- 5. layers (ModuleList → setattr 方式) ---------------------------
class DeepMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hid   = cfg["model"]["hidden"]        # 例: [512,512,256] 可変長
        d_in  = cfg["model"]["in_dim"]
        d_out = cfg["model"]["out_dim"]
        pdrop = cfg["model"]["dropout"]

        dims = [d_in] + hid + [d_out]
        self.n_hidden = len(hid)

        # 動的に作成して register
        for i in range(self.n_hidden):
            setattr(self, f"fc{i}", Dense(dims[i],   dims[i+1]))
            setattr(self, f"bn{i}", BN(   dims[i+1]))
            setattr(self, f"do{i}", DO(pdrop))

        self.fc_out = Dense(dims[-2], dims[-1])   # 出力層

    def forward(self, x):
        for i in range(self.n_hidden):
            fc = getattr(self, f"fc{i}")
            bn = getattr(self, f"bn{i}")
            do = getattr(self, f"do{i}")
            x  = relu(bn(fc(x)))
            x  = do(x)
        return softmax(self.fc_out(x))


# ---------- 6. loss utils ---------------------------------------------------
num_cls = C["model"]["out_dim"]           
def one_hot(t):
    t = t.long()
    o = torch.zeros(t.size(0), num_cls, device=t.device)
    o.scatter_(1, t.unsqueeze(1), 1.0)
    return o


def loss_soft_ce(p, q):
    return -(q * torch.log(p + 1e-12)).sum(1).mean()

# ---------- 7. train --------------------------------------------------------
M = DeepMLP(C).to(device)
opt = optim.Adam(M.parameters(), lr=C["train"]["lr"])
n_ep = C["train"]["n_epochs"]; patience = C["train"]["patience"]
swa_after = int(n_ep * (1 - C["train"]["swa_pct"]))

best_acc, wait = 0.0, 0
swa_state, swa_n = None, 0

for ep in range(n_ep):
    # LR scheduler (cosine)
    if C["train"]["lr_schedule"] == "cosine":
        lr = C["train"]["lr"] * 0.5 * (1 + math.cos(math.pi * ep / n_ep))
        for g in opt.param_groups: g["lr"] = lr

    # ---- train (MixUp + LS) ----
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

    # ---- SWA accumulate ----
    if ep >= swa_after:
        if swa_state is None:
            swa_state = {k: v.detach().cpu().clone() for k, v in M.state_dict().items()}
        else:
            for k in swa_state:
                swa_state[k] = (swa_state[k] * swa_n + M.state_dict()[k].detach().cpu()) / (swa_n + 1)
        swa_n += 1

    # ---- validate ----
    M.eval(); corr = tot = 0
    with torch.no_grad():
        for xb, tb in ldr_va:
            pred = M(xb.to(device)).argmax(1).cpu()
            corr += (pred == tb).sum().item(); tot += tb.size(0)
    acc = corr / tot
    print(f"Epoch {ep+1}/{n_ep}  val_acc={acc:.4f}")

    if acc > best_acc + 1e-4:
        best_acc, best_state = acc, {k: v.cpu() for k, v in M.state_dict().items()}
        wait = 0
    else:
        wait += 1
    if wait >= patience:
        print("Early stopping."); break

# ---- apply SWA or best ----
if swa_state is not None:
    print(f"SWA over {swa_n} epoch(s) applied")
    M.load_state_dict(swa_state)
else:
    M.load_state_dict(best_state)

# ---------- 8. test ---------------------------------------------------------
M.eval(); preds=[]
with torch.no_grad():
    for xb in ldr_te:
        preds.extend(M(xb.to(device)).argmax(1).cpu().tolist())

os.makedirs("sub", exist_ok=True)
fn = f"sub/{pd.Timestamp('now').strftime('%Y%m%d%H%M')}_{best_acc:.4f}.csv"
fn_yaml = fn.replace(".csv", ".yaml")
pd.Series(preds, name="label").to_csv(fn, header=True, index_label="id")
print("saved:", fn)

with open(fn_yaml, "w") as f:
    yaml.safe_dump(C, f, sort_keys=False)