# Canonical IEEE TIFS submission experiment.
# Corrected protocol: common federated initialization, sample-weighted
# FedAvg/FedProx, online-only feature normalization, source-account-consistent
# typology skew, time-consistent drift injection, client-macro metrics, and
# seed-level statistical inference.

# ===== cell01_config_local.py =====
# ---------------- 0. Config ----------------
import os
FAST_DEV      = os.environ.get("FT_FAST_DEV", "0") == "1"
DATASET       = os.environ.get("FT_DATASET", "ibm").strip().lower()
if DATASET not in {"ibm","samld"}:
    raise ValueError("FT_DATASET must be 'ibm' or 'samld'")
SEED          = 42
K_CLIENTS     = 5
D_MODEL       = 64
N_WINDOWS_CAP = 20   # reporting target; loaders retain every natural window
WINDOW_FREQ   = "24h" if DATASET=="ibm" else "17D"
LOCAL_EPOCHS  = 1
LR            = 2e-3
FOCAL_GAMMA   = 2.0
LAMBDA_ALERT  = 0.4        # weight of noisy alert labels vs confirmed
M_PROTO       = 4          # prototypes per client (v2: was 6; positives are scarce)
MIN_CONF_POS  = 8          # v2: min confirmed positives before building prototypes
NEG_SUBSAMPLE = 4000 if FAST_DEV else 20000
VALID_W       = 20         # metrics computed on windows < VALID_W
EVAL_START_W  = 1          # window 0 initializes client-local online statistics
RHO_DAMP      = 0.2        # cross-cluster damping
BETA_REGISTRY = 0.15       # registry risk-boost weight (inoculation mechanism)
NOVELTY_GATE  = 0.3        # v2: was 0.5
MU_PROX       = 1e-3       # v2: was 0.01 (broken baseline)
MU_PROTO      = 0.1        # FedProto confirmed-label representation penalty
ALERT_TPR     = 0.60       # immediate alert sensitivity on true positives
ALERT_FPR     = 0.01       # immediate alert false-positive rate
DELAY_MEDIAN_WINDOWS = 3   # normalized evaluation cycles, not calendar weeks
BUDGET_K      = 50         # precision@budget alerts per client per window
ROWS_DEV      = 350000 if FAST_DEV else None
N_SUBMISSION_SEEDS = 1 if FAST_DEV else int(os.environ.get("FT_N_SEEDS","10"))
SUBMISSION_SEEDS = tuple(range(42,42+N_SUBMISSION_SEEDS))

import math, json, random, warnings
import hashlib
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import average_precision_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
warnings.filterwarnings("ignore")
rng = np.random.default_rng(SEED); random.seed(SEED); np.random.seed(SEED)
WINDOW_SECONDS = int(pd.Timedelta(WINDOW_FREQ).total_seconds())

# ---- dataset discovery ----
# ADAPTED FOR LOCAL SERVER: searches /tmp/ml/data first (this run's scoped
# data directory), then /kaggle/input as a harmless no-op fallback so this
# script would also still work unchanged if ever run on Kaggle. Preserves
# the original fail-loudly philosophy: if a data directory exists but the
# expected CSV isn't in it, raise rather than silently using synthetic data.
import glob
DATA_ROOTS = [
    os.environ.get("FT_DATA_ROOT", "/tmp/ml/data"),
    "/kaggle/input",
]
_cands, _pats = [], []
for root in DATA_ROOTS:
    _cands += glob.glob(f"{root}/**/HI-Small_Trans.csv", recursive=True)
    _pats  += glob.glob(f"{root}/**/HI-Small_Patterns.txt", recursive=True)
IBM_PATH     = _cands[0] if _cands else ""
IBM_PATTERNS = _pats[0]  if _pats  else ""
_samld_root=os.environ.get("FT_SAMLD_ROOT","/tmp/ml/data_samld")
_samld_cands=glob.glob(f"{_samld_root}/**/SAML-D.csv",recursive=True)
SAMLD_PATH=_samld_cands[0] if _samld_cands else ""
print("IBM_PATH ->", IBM_PATH or "NOT FOUND")
print("PATTERNS ->", IBM_PATTERNS or "NOT FOUND")
print("SAMLD_PATH ->", SAMLD_PATH or "NOT FOUND")
_present_roots = [r for r in DATA_ROOTS if os.path.isdir(r)]
if DATASET=="ibm" and (not IBM_PATH or not IBM_PATTERNS):
    _contents = []
    for r in _present_roots:
        _contents += glob.glob(f"{r}/**/*", recursive=True)
    raise FileNotFoundError(
        "Both HI-Small_Trans.csv and HI-Small_Patterns.txt are required. "
        "At least one was not found in the configured data roots. Contents:\n"
        + "\n".join(_contents[:60]))
if DATASET=="samld" and not SAMLD_PATH:
    raise FileNotFoundError(f"SAML-D.csv not found under {_samld_root}")

# ===== cell02_loader.py =====
# ---------------- 1. Canonical dataset loaders ----------------
PATTERN_IDS = {"FAN-OUT":0,"FAN-IN":1,"GATHER-SCATTER":2,"SCATTER-GATHER":3,
               "CYCLE":4,"RANDOM":5,"BIPARTITE":6,"STACK":7}
UNSTRUCTURED = 8
TYP_NAMES = {v:k for k,v in PATTERN_IDS.items()}; TYP_NAMES[UNSTRUCTURED]="UNSTRUCTURED"
MODEL_COLUMNS = [
    "txn_id","src_account","dst_account","time_window","timestamp","amount",
    "channel","cross_border","label","typology_id",
]

def compact_model_frame(frame):
    """Retain model inputs only; essential for the 9.5M-row SAML-D run."""
    frame=frame.loc[:,MODEL_COLUMNS].copy()
    frame["txn_id"]=frame.txn_id.astype(np.int32)
    frame["src_account"]=frame.src_account.astype(np.int32)
    frame["dst_account"]=frame.dst_account.astype(np.int32)
    frame["time_window"]=frame.time_window.astype(np.int16)
    frame["cross_border"]=frame.cross_border.astype(np.int8)
    frame["label"]=frame.label.astype(np.int8)
    frame["typology_id"]=frame.typology_id.astype(np.int16)
    frame["channel"]=frame.channel.astype("category")
    return frame

def parse_patterns(path):
    import re
    mapping, current = {}, None
    with open(path) as f:
        for line in f:
            line=line.strip()
            m=re.match(r"BEGIN LAUNDERING ATTEMPT - ([A-Z\-]+)", line)
            if m: current=PATTERN_IDS.get(m.group(1).strip(),UNSTRUCTURED); continue
            if line.startswith("END LAUNDERING ATTEMPT"): current=None; continue
            if current is not None and "," in line:
                p=line.split(",")
                if len(p)>=7: mapping[(p[0],p[2],p[4],p[5])]=current
    return mapping

def load_ibm(nrows=None):
    df = pd.read_csv(IBM_PATH, nrows=nrows)
    df = df.rename(columns={"Timestamp":"ts_raw","Account":"src_raw","Account.1":"dst_raw",
        "Amount Paid":"amount","Payment Currency":"currency","Payment Format":"channel",
        "Is Laundering":"label","From Bank":"src_bank","To Bank":"dst_bank"})
    ts = pd.to_datetime(df["ts_raw"], format="%Y/%m/%d %H:%M")
    df["timestamp"] = ts.astype("int64")//10**9
    sf = df["src_bank"].astype(str)+"_"+df["src_raw"].astype(str)
    dfl= df["dst_bank"].astype(str)+"_"+df["dst_raw"].astype(str)
    codes,_ = pd.factorize(pd.concat([sf,dfl]))
    df["src_account"]=codes[:len(df)]; df["dst_account"]=codes[len(df):]
    df["cross_border"]=(df["src_bank"]!=df["dst_bank"]).astype("int8")
    df["txn_id"]=np.arange(len(df))
    df["typology_id"]=-1
    if os.path.exists(IBM_PATTERNS):
        mp = parse_patterns(IBM_PATTERNS)
        key = list(zip(df["ts_raw"].astype(str),df["src_raw"].astype(str),
                       df["dst_raw"].astype(str),df["amount"].astype(str)))
        df["typology_id"]=[mp.get(k,-1) for k in key]
    df.loc[(df.label==1)&(df.typology_id==-1),"typology_id"]=UNSTRUCTURED
    # Retain the complete natural time span.  Earlier versions clipped the
    # integer window index at 19, which silently collapsed the IBM tail into
    # one oversized terminal window.
    df["time_window"]=((ts-ts.min())/pd.Timedelta(WINDOW_FREQ)).astype(int)
    return compact_model_frame(df)

def load_samld(nrows=None):
    d=pd.read_csv(SAMLD_PATH,nrows=nrows)
    d=d.rename(columns={
        "Amount":"amount","Payment_currency":"currency",
        "Payment_type":"channel","Is_laundering":"label"})
    ts=pd.to_datetime(d["Date"]+" "+d["Time"])
    d["timestamp"]=ts.astype("int64")//10**9
    codes,_=pd.factorize(pd.concat([
        d["Sender_account"].astype(str),
        d["Receiver_account"].astype(str)]))
    d["src_account"]=codes[:len(d)]; d["dst_account"]=codes[len(d):]
    d["cross_border"]=(
        d["Sender_bank_location"]!=d["Receiver_bank_location"]).astype("int8")
    d["txn_id"]=np.arange(len(d))
    d["typology_id"]=-1
    pos=d.label==1
    typ_codes,typ_uniques=pd.factorize(d.loc[pos,"Laundering_type"])
    d.loc[pos,"typology_id"]=typ_codes
    global TYP_NAMES
    TYP_NAMES={i:str(name) for i,name in enumerate(typ_uniques)}
    d["time_window"]=((ts-ts.min())/pd.Timedelta(WINDOW_FREQ)).astype(int)
    return compact_model_frame(d)

if DATASET=="ibm" and IBM_PATH and os.path.exists(IBM_PATH):
    df = load_ibm(nrows=ROWS_DEV)
    DATA_NAME="IBM-AML HI-Small"
elif DATASET=="samld":
    df=load_samld(nrows=ROWS_DEV)
    DATA_NAME="SAML-D"
else:
    raise RuntimeError(f"unreachable dataset selection: {DATASET}")
N_WINDOWS = int(df.time_window.max())+1
if df.loc[df.label==1,"typology_id"].nunique()<2:
    raise RuntimeError(
        "Fewer than two positive typologies were parsed; drift and "
        "typology-skew experiments would be invalid."
    )
print(f"dataset={DATA_NAME} txns={len(df):,} "
      f"accounts={pd.concat([df.src_account,df.dst_account]).nunique():,} "
      f"windows={N_WINDOWS} prevalence={100*df.label.mean():.4f}% "
      f"positives={int(df.label.sum()):,} "
      f"typologies={df.loc[df.typology_id>=0,'typology_id'].nunique()}")

# ===== cell03_partition.py =====
# ---------------- 2. Partition (P1 community x P2 typology-skew) ----------------
def partition_community(df,k,seed=SEED):
    g=nx.Graph(); grp=df.groupby(["src_account","dst_account"]).size()
    g.add_weighted_edges_from((s,d,w) for (s,d),w in grp.items())
    comms=sorted(nx.community.louvain_communities(g,seed=seed),key=len,reverse=True)
    loads=np.zeros(k); a2c={}
    for c in comms:
        j=int(np.argmin(loads)); loads[j]+=len(c)
        for a in c: a2c[a]=j
    out=df.src_account.map(a2c).fillna(0).astype(int)
    return pd.Series(out.values,index=df.txn_id.values,name="client")

def apply_typology_skew(df,assign,alpha=0.3,seed=SEED):
    """Induce typology skew without splitting a source account across clients.

    Only source accounts with confirmed suspicious activity are reassigned,
    based on their dominant typology; all outgoing transactions of each such
    account move together. This retains a coherent source-account silo while
    preserving the community partition for all other accounts.
    """
    r=np.random.default_rng(seed)
    k=int(assign.max())+1
    base = pd.DataFrame({
        "src_account": df.src_account.values,
        "client": assign.loc[df.txn_id].values,
    }).groupby("src_account").client.first().to_dict()
    sus=df[df.label==1]
    typs=sorted(sus.typology_id.unique())
    prof=r.dirichlet([alpha]*len(typs),size=k)
    dominant=(sus.groupby("src_account").typology_id
              .agg(lambda x: x.value_counts().index[0]))
    for ti,t in enumerate(typs):
        accounts=dominant[dominant==t].index.to_numpy()
        if len(accounts)==0: continue
        w=prof[:,ti]/prof[:,ti].sum()
        chosen=r.choice(k,size=len(accounts),p=w)
        for account,client in zip(accounts,chosen):
            base[account]=int(client)
    out=df.src_account.map(base).fillna(0).astype(int)
    return pd.Series(out.values,index=df.txn_id.values,name="client")

# ===== cell04_drift.py =====
# ---------------- 3. Drift injection (D1 + D4 staggered for E2) + SAR delay ----------------
class DriftInjector:
    def __init__(self,df,assign,seed=SEED):
        self.df=df.copy(); self.assign=assign; self.rng=np.random.default_rng(seed); self.events=[]
    def d1(self,client,typ,onset):
        d=self.df
        wmax=int(d.time_window.max())
        if onset>wmax:
            self.events.append(dict(kind="D1",client=int(client),
                                    typology=int(typ),window=int(onset),moved=0))
            print(f"  WARNING: onset {onset} exceeds final window {wmax}; "
                  "event retained as a zero-move diagnostic")
            return
        m=((d.label==1)&(d.typology_id==typ)&(self.assign.loc[d.txn_id].values==client)
           &(d.time_window<onset))
        n=int(m.sum())
        if n:
            new_w=self.rng.integers(onset,wmax+1,n).astype(d.time_window.dtype)
            base_ts=int(d.timestamp.min())
            offsets=(d.loc[m,"timestamp"].to_numpy()-base_ts)%WINDOW_SECONDS
            d.loc[m,"time_window"]=new_w
            d.loc[m,"timestamp"]=base_ts+new_w.astype(np.int64)*WINDOW_SECONDS+offsets
        if n==0: print(f"  WARNING: D1 moved 0 txns (client={client}, typ={typ}) — check targeting")
        self.events.append(dict(kind="D1",client=int(client),typology=int(typ),window=int(onset),moved=n))
    def d4(self,clients,typ,base,lag):
        for i,c in enumerate(clients):
            self.d1(c,typ,base+i*lag); self.events[-1].update(kind="D4",stagger=i,lag=lag)

def build_data(condition, seed):
    """Build (df_e, assign_e, drift_events) for one experiment. condition in {'control','drift'}."""
    rng_e = np.random.default_rng(seed)
    assign_e = apply_typology_skew(df, partition_community(df,K_CLIENTS,seed=seed),
                                   alpha=0.3, seed=seed)
    d = df.copy()
    events = pd.DataFrame()
    if condition == "drift":
        sus_e = d[d.label==1]
        hold = pd.crosstab(assign_e.loc[sus_e.txn_id].values, sus_e.typology_id)
        tmass = hold.sum(0).sort_values(ascending=False).index.tolist()
        if DATASET=="ibm":
            # UNSTRUCTURED is a catch-all positive label, not a named
            # laundering pattern. Retain it for training/fidelity but do not
            # present its re-timing as a controlled typology onset.
            tmass=[typ for typ in tmass if int(typ)!=UNSTRUCTURED]
        if len(tmass)<2:
            raise RuntimeError("drift injection requires two named typologies")
        t_d1 = int(tmass[0]); t_d4 = int(tmass[1] if len(tmass)>1 else tmass[0])
        c_d1 = int(hold[t_d1].idxmax())
        c_d4 = [int(c) for c in hold[t_d4].sort_values(ascending=False).index[:3]]
        inj = DriftInjector(d, assign_e, seed=seed)
        inj.d1(c_d1, t_d1, N_WINDOWS//2)
        inj.d4(c_d4, t_d4, N_WINDOWS//3, max(2, N_WINDOWS//6))
        d = inj.df; events = pd.DataFrame(inj.events)
    # Delayed confirmation in normalized evaluation cycles. The immediate
    # alert stream uses asymmetric error rates rather than symmetric flips,
    # which would turn roughly 10% of all legitimate transactions positive.
    mu_ = math.log(DELAY_MEDIAN_WINDOWS); sig_ = (math.log(DELAY_MEDIAN_WINDOWS*4)-mu_)/1.645
    delay_ = np.rint(np.exp(rng_e.normal(mu_, sig_, len(d)))).astype(int)
    d["confirmed_window"] = d.time_window + np.maximum(1, delay_)
    draw = rng_e.random(len(d))
    d["alert_label"] = np.where(d.label.values==1,
                                draw < ALERT_TPR,
                                draw < ALERT_FPR).astype(np.int8)
    return d, assign_e, events

# ===== cell05_features.py =====
# ---------------- 4. Feature engineering ----------------
CHANNELS = df.channel.astype(str).unique().tolist()[:8]
CH_IDX={c:i for i,c in enumerate(CHANNELS)}
def edge_features(sub, client_stats=None):
    la=np.log1p(sub.amount.values)
    # The first natural window is a declared, non-scored warm-up window. Its
    # within-window scale only advances the model/memory and seeds the
    # persistent online statistics used for every evaluated window.
    if client_stats is None: m,s=la.mean(),la.std()+1e-6
    else: m,s=client_stats
    f=np.zeros((len(sub),4+len(CHANNELS)),dtype=np.float32)
    f[:,0]=(la-m)/s
    f[:,1]=sub.cross_border.values
    hrs=(sub.timestamp.values//3600)%24; f[:,2]=np.sin(2*np.pi*hrs/24); f[:,3]=np.cos(2*np.pi*hrs/24)
    for i,c in enumerate(sub.channel.astype(str).values):
        j=CH_IDX.get(c)
        if j is not None: f[i,4+j]=1.0
    return f
EDGE_DIM = 4+len(CHANNELS)
print("edge feature dim:", EDGE_DIM)

# ===== cell06_model.py =====
# ---------------- 5. Model: scatter-attention GNN + GRU memory (pure PyTorch) ----------------
import torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True, warn_only=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEV)

def scatter_softmax(scores, index, n):
    mx = torch.full((n,), -1e30, device=scores.device)
    mx = mx.scatter_reduce(0, index, scores, reduce="amax", include_self=True)
    ex = torch.exp(scores - mx[index])
    den = torch.zeros(n, device=scores.device).index_add_(0, index, ex) + 1e-12
    return ex / den[index]

class Backbone(nn.Module):
    """SHARED across clients (aggregated): 2 attention message-passing layers
    + GRUCell temporal memory per account."""
    def __init__(self, d=D_MODEL, ed=None):
        super().__init__()
        ed = ed or EDGE_DIM
        self.edge_in = nn.Linear(ed, d)
        self.att1 = nn.Linear(3*d, 1); self.upd1 = nn.Linear(2*d, d)
        self.att2 = nn.Linear(3*d, 1); self.upd2 = nn.Linear(2*d, d)
        self.gru  = nn.GRUCell(d, d)
    def mp(self, h, src, dst, e, att, upd, n):
        msg = torch.cat([h[src], h[dst], e], 1)
        a = scatter_softmax(att(msg).squeeze(-1), dst, n)
        agg = torch.zeros(n, h.size(1), device=h.device)
        agg.index_add_(0, dst, (h[src] + e) * a.unsqueeze(-1))
        return F.relu(upd(torch.cat([h, agg], 1)))
    def forward(self, mem, src, dst, efeat):
        n = mem.size(0)
        e = F.relu(self.edge_in(efeat))
        h = self.mp(mem, src, dst, e, self.att1, self.upd1, n)
        h = self.mp(h,  src, dst, e, self.att2, self.upd2, n)
        new_mem = self.gru(h, mem)
        edge_emb = torch.cat([new_mem[src], new_mem[dst], e], 1)   # [E, 3d]
        return new_mem, edge_emb

class Head(nn.Module):
    """PERSONALIZED per client (never aggregated)."""
    def __init__(self, d=D_MODEL):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3*d, d), nn.ReLU(), nn.Linear(d, 1))
    def forward(self, edge_emb): return self.net(edge_emb).squeeze(-1)

def focal_loss(logits, y, gamma=FOCAL_GAMMA, alpha=0.75):
    p = torch.sigmoid(logits)
    pt = torch.where(y > 0.5, p, 1-p)
    a = torch.where(y > 0.5, torch.full_like(p, alpha), torch.full_like(p, 1-alpha))
    l = -a * (1-pt).clamp_min(1e-6)**gamma * torch.log(pt.clamp_min(1e-6))
    return l.mean()

# ===== cell07_client.py =====
# ---------------- 6. Client ----------------
class Client:
    def __init__(self, cid, txns, initial_backbone_state, seed):
        self.id=cid
        self.seed=int(seed)
        self.txns=txns.sort_values("time_window").reset_index(drop=True)
        accts=pd.concat([self.txns.src_account,self.txns.dst_account]).unique()
        self.g2l={a:i for i,a in enumerate(accts)}
        self.n=len(accts)
        self.stats=None
        self.stat_count=0; self.stat_mean=0.0; self.stat_m2=0.0
        self.backbone=Backbone().to(DEV)
        self.backbone.load_state_dict(initial_backbone_state)
        self.head=Head().to(DEV)
        self.mem=torch.zeros(self.n,D_MODEL,device=DEV)
        self.reset_optimizer()
        self.protos=None; self.proto_tau=None
        self.s1_hist=[]; self.stability=1.0
        self.pos_seen=0
        self.last_train_count=1
        self.proto_purity=np.nan; self.proto_nmi=np.nan; self.proto_ari=np.nan
    def reset_optimizer(self):
        # Standard FL uses a fresh local optimizer after every server broadcast.
        self.opt=torch.optim.Adam(list(self.backbone.parameters())+list(self.head.parameters()),
                                  lr=LR)
    def _update_stats(self, sub):
        x=np.log1p(sub.amount.to_numpy(dtype=float))
        if len(x)==0: return
        n=len(x); mean=float(x.mean()); m2=float(((x-mean)**2).sum())
        if self.stat_count==0:
            self.stat_count=n; self.stat_mean=mean; self.stat_m2=m2
        else:
            delta=mean-self.stat_mean; total=self.stat_count+n
            self.stat_m2 += m2 + delta*delta*self.stat_count*n/total
            self.stat_mean += delta*n/total
            self.stat_count=total
        self.stats=(self.stat_mean, math.sqrt(self.stat_m2/max(1,self.stat_count))+1e-6)
    def _tensors(self, sub):
        src=torch.tensor([self.g2l[a] for a in sub.src_account.values],device=DEV)
        dst=torch.tensor([self.g2l[a] for a in sub.dst_account.values],device=DEV)
        ef=torch.tensor(edge_features(sub,self.stats),device=DEV)
        return src,dst,ef
    @torch.no_grad()
    def predict(self, w, registry=None):
        sub=self.txns[self.txns.time_window==w]
        if len(sub)==0: return sub, np.array([]), None
        src,dst,ef=self._tensors(sub)
        new_mem,emb=self.backbone(self.mem,src,dst,ef)
        score=torch.sigmoid(self.head(emb)).cpu().numpy()
        if registry is not None and len(registry)>0:
            R=F.normalize(torch.tensor(np.stack(registry),device=DEV,dtype=torch.float32),dim=1)
            E=F.normalize(emb,dim=1)
            boost=(E@R.T).max(1).values.clamp_min(0).cpu().numpy()
            score=np.clip(score+BETA_REGISTRY*boost,0,1)
        self.mem=new_mem                      # prequential temporal memory advance
        self._update_stats(sub)               # current unlabeled window, never future data
        return sub, score, emb.cpu()
    def train_round(self, w, mu_prox=0.0, global_state=None,
                    global_class_protos=None, mu_proto=0.0):
        vis_conf=self.txns[(self.txns.confirmed_window<=w)&(self.txns.time_window<=w)]
        vis_alert=self.txns[self.txns.time_window==w]
        frames=[]
        if len(vis_conf):  frames.append((vis_conf ,"label",1.0))
        if len(vis_alert): frames.append((vis_alert,"alert_label",LAMBDA_ALERT))
        self.last_train_count=1
        if not frames: return
        processed=0
        for _ in range(LOCAL_EPOCHS):
            for sub,col,wt in frames:
                pos=sub[sub[col]==1]; neg=sub[sub[col]==0]
                if len(neg)>NEG_SUBSAMPLE:
                    neg=neg.sample(NEG_SUBSAMPLE,random_state=self.seed+w)
                sub=pd.concat([pos,neg])
                if len(sub)==0: continue
                processed += len(sub)
                src,dst,ef=self._tensors(sub)
                _,emb=self.backbone(self.mem.detach(),src,dst,ef)
                logit=self.head(emb)
                y=torch.tensor(sub[col].values,dtype=torch.float32,device=DEV)
                loss=focal_loss(logit,y)*wt
                if mu_prox>0 and global_state is not None:
                    prox=sum((p-g.to(DEV)).pow(2).sum() for p,g in
                             zip(self.backbone.parameters(),global_state))
                    loss=loss+0.5*mu_prox*prox
                if (mu_proto>0 and global_class_protos and col=="label"):
                    norm_emb=F.normalize(emb,dim=1)
                    proto_loss=torch.zeros((),device=DEV)
                    present=0
                    for cls,proto in global_class_protos.items():
                        mask=(y==float(cls))
                        if mask.any():
                            target=torch.tensor(proto,dtype=torch.float32,
                                                device=DEV)
                            proto_loss += (norm_emb[mask]-target).pow(2).sum(1).mean()
                            present += 1
                    if present:
                        loss=loss+mu_proto*proto_loss/present
                self.opt.zero_grad(); loss.backward(); self.opt.step()
        self.last_train_count=max(1,processed)
        self.pos_seen=int(self.txns.loc[self.txns.confirmed_window<=w,"label"].sum())
    @torch.no_grad()
    def class_prototypes(self,w):
        """Confirmed-label class prototypes for the FedProto comparator."""
        conf=self.txns[(self.txns.confirmed_window<=w)&
                       (self.txns.time_window<=w)]
        if len(conf)==0: return {}
        out={}
        for cls in (0,1):
            sub=conf[conf.label==cls]
            if cls==0 and len(sub)>NEG_SUBSAMPLE:
                sub=sub.sample(NEG_SUBSAMPLE,random_state=self.seed+w+777)
            if len(sub)==0: continue
            src,dst,ef=self._tensors(sub)
            _,emb=self.backbone(self.mem,src,dst,ef)
            mean=F.normalize(emb,dim=1).mean(0)
            mean=F.normalize(mean.unsqueeze(0),dim=1).squeeze(0)
            out[cls]=(mean.cpu().numpy(),len(sub))
        return out
    # ---- prototypes & drift signals ----
    @torch.no_grad()
    def build_prototypes(self, w):
        conf=self.txns[(self.txns.confirmed_window<=w)&(self.txns.label==1)&(self.txns.time_window<=w)]
        if len(conf) < max(MIN_CONF_POS, M_PROTO*2): return         # v2 threshold
        src,dst,ef=self._tensors(conf)
        _,emb=self.backbone(self.mem,src,dst,ef)
        X=F.normalize(emb,dim=1).cpu().numpy()
        k=min(M_PROTO, max(2, len(X)//2))                            # v2 adaptive k
        km=KMeans(n_clusters=k,n_init=4,random_state=self.seed).fit(X)
        self.protos=km.cluster_centers_/(np.linalg.norm(km.cluster_centers_,axis=1,keepdims=True)+1e-9)
        dmin=1-(X@self.protos.T).max(1)
        self.proto_tau=float(np.quantile(dmin,0.95))
        truth=conf.typology_id.to_numpy()
        pred=km.labels_
        self.proto_nmi=float(normalized_mutual_info_score(truth,pred))
        self.proto_ari=float(adjusted_rand_score(truth,pred))
        self.proto_purity=float(sum(
            np.bincount(truth[pred==cluster].astype(int)).max()
            for cluster in np.unique(pred)
        )/len(truth))
    def signal1(self, emb_now, emb_ref):
        if emb_now is None or emb_ref is None or len(emb_now)<8 or len(emb_ref)<8: return 0.0
        r=np.random.default_rng(self.seed)
        A=emb_now[r.choice(len(emb_now),min(256,len(emb_now)),replace=False)]
        B=emb_ref[r.choice(len(emb_ref),min(256,len(emb_ref)),replace=False)]
        Z=np.vstack([A,B]); d2=((Z[:,None,:]-Z[None,:,:])**2).sum(-1)
        s=np.median(d2)+1e-9; Kk=np.exp(-d2/s)
        na=len(A)
        mmd=Kk[:na,:na].mean()+Kk[na:,na:].mean()-2*Kk[:na,na:].mean()
        self.s1_hist.append(float(mmd))
        if len(self.s1_hist)>=5:
            base=np.median(self.s1_hist[:-1])
            mad=np.median(np.abs(np.array(self.s1_hist[:-1])-base))+1e-9
            z=(mmd-base)/mad
            self.stability=0.7*self.stability+0.3*float(np.exp(-max(0.0,z-3.0)))
        return float(mmd)
    @torch.no_grad()
    def signal2_novelty(self, w):
        if self.protos is None: return 0.0, None
        new=self.txns[(self.txns.confirmed_window==w)&(self.txns.label==1)]
        if len(new)==0: return 0.0, None
        src,dst,ef=self._tensors(new)
        _,emb=self.backbone(self.mem,src,dst,ef)
        X=F.normalize(emb,dim=1).cpu().numpy()
        dmin=1-(X@self.protos.T).max(1)
        novel=dmin>self.proto_tau if self.proto_tau else dmin>0.5
        frac=float(novel.mean())
        newproto=None
        if novel.sum()>=3:
            v=X[novel].mean(0); newproto=v/(np.linalg.norm(v)+1e-9)
        return frac, newproto

# ===== cell08_server.py =====
# ---------------- 7. Server: typology clustering, multi-factor aggregation, registry ----------------
def chamfer(P,Q):
    S=P@Q.T
    return float(1-0.5*(S.max(1).mean()+S.max(0).mean()))

def cluster_clients(clients):
    have=[c for c in clients if c.protos is not None]
    if len(have)<3: return {c.id:0 for c in clients}                 # v2 guard
    n=len(have); D=np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            D[i,j]=D[j,i]=chamfer(have[i].protos,have[j].protos)
    lab=AgglomerativeClustering(n_clusters=2,metric="precomputed",
                                linkage="average").fit_predict(D)
    out={c.id:int(l) for c,l in zip(have,lab)}
    for c in clients: out.setdefault(c.id,0)
    return out

def fedavg_aggregate(clients, weights=None, groups=None):
    sds=[c.backbone.state_dict() for c in clients]
    if weights is None: weights=[1.0]*len(clients)
    wsum=sum(weights); weights=[w/wsum for w in weights]
    gmean={k:sum(w*sd[k].float() for w,sd in zip(weights,sds)) for k in sds[0]}
    if groups is None:
        for c in clients: c.backbone.load_state_dict(gmean)
        return [p.detach().clone() for p in clients[0].backbone.parameters()]
    for g in set(groups.values()):
        idx=[i for i,c in enumerate(clients) if groups[c.id]==g]
        wsub=[weights[i] for i in idx]; s=sum(wsub) or 1.0
        cmean={k:sum(w/s*sds[i][k].float() for w,i in zip(wsub,idx)) for k in sds[0]}
        mix={k:(1-RHO_DAMP)*cmean[k]+RHO_DAMP*gmean[k] for k in cmean}
        for i in idx: clients[i].backbone.load_state_dict(mix)
    return None

# ===== cell09_harness_cda.py =====
# ---------------- 8. Prequential harness (+ CDA-FedAvg baseline) ----------------
# CDA-FedAvg (Casado et al. 2022, "Concept drift detection and adaptation for
# federated and continual learning", arXiv:2105.13309) is reimplemented here
# preserving its two distinctive mechanisms:
#   (1) confidence-based Beta-distribution CUSUM drift detector (their Alg. 5):
#       a sliding window of per-round mean prediction confidence is split at
#       every candidate point, Beta params fit to each half via method of
#       moments, and a log-likelihood-ratio statistic compared to a threshold.
#   (2) rehearsal-based drift adaptation (their Alg. 6): a growing long-term
#       memory of past confirmed-labeled examples; when drift is detected,
#       R extra local training rounds run on this rehearsal buffer before
#       that round's aggregation.
# ADAPTATION FROM THE ORIGINAL PAPER (disclosed, not a literal port):
#   - The original operates per-instance with an asynchronous, event-driven
#     server. FedTypo's harness is synchronous and window-based (20 windows
#     total per run), so detection operates on one mean-confidence value per
#     window per client (not per instance), with the sliding-window padding
#     Delta shrunk from their Nmax=1000/instance setting to Delta=3 windows
#     (their lambda=0.05 sensitivity is kept unchanged). Aggregation stays
#     the plain FedAvg weighted average every window (not their fully async
#     server loop), so results remain directly comparable, under the exact
#     same prequential per-window evaluation, to every other method in this
#     comparison.
#   - Their per-class-balanced rehearsal quota (L/2M per class) assumes
#     roughly-balanced classes (7-way HAR); under ~0.1% AML prevalence this
#     is inverted: the rehearsal buffer retains ALL confirmed positives seen
#     (they are the scarce, valuable class) plus a capped recent sample of
#     confirmed negatives (REHEARSAL_NEG_CAP), mirroring the class-balanced
#     subsampling philosophy already used elsewhere in this codebase rather
#     than the original's literal per-class quota.
#   - Supervision regime (confirmed labels after SAR delay, immediate noisy
#     alerts) is identical to every other method here; CDA-FedAvg's own
#     detector does not require labels (confidence is unsupervised), only
#     the rehearsal buffer does, exactly as in the original.
from scipy.stats import beta as beta_dist

CDA_DELTA       = 3     # min sub-window size (windows, not instances -- see note above)
CDA_LAMBDA      = 0.05  # sensitivity to change, unchanged from the original paper
CDA_R_ROUNDS    = 5     # extra rehearsal rounds per detected drift, per the original paper
REHEARSAL_NEG_CAP = 2000  # capped negative rehearsal pool (positives are never capped: they're rare)

def estimate_beta_params(x):
    x = np.clip(np.asarray(x, dtype=float), 1e-4, 1 - 1e-4)
    m = x.mean(); v = x.var()
    if v <= 1e-8 or m <= 0 or m >= 1:
        return 1.0, 1.0
    common = m * (1 - m) / v - 1
    a = max(m * common, 1e-3)
    b = max((1 - m) * common, 1e-3)
    return a, b

def cda_drift_detected(Q, lam=CDA_LAMBDA, delta=CDA_DELTA):
    """Port of Algorithm 5 (Casado et al. 2022): Beta-CUSUM drift test on a
    sliding window of mean confidences. Returns True/False."""
    N = len(Q)
    if N < 2 * delta + 1:
        return False
    Th = -math.log(lam)
    sf = 0.0
    arr = np.asarray(Q, dtype=float)
    for k in range(delta, N - delta):
        mb = arr[:k].mean(); ma = arr[k:].mean()
        if ma <= (1 - lam) * mb:
            ab, bb = estimate_beta_params(arr[:k])
            aa, ba = estimate_beta_params(arr[k:])
            sk = np.sum(beta_dist.logpdf(np.clip(arr[k:], 1e-4, 1 - 1e-4), aa, ba)
                        - beta_dist.logpdf(np.clip(arr[k:], 1e-4, 1 - 1e-4), ab, bb))
            sf = max(sf, sk)
    return sf > Th

def run_method(method, df, assign, seed=SEED, verbose=True):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    N_WINDOWS = int(df.time_window.max())+1
    registry_log = []
    initial_backbone=Backbone().to(DEV)
    initial_state={k:v.detach().clone() for k,v in initial_backbone.state_dict().items()}
    clients=[Client(c, df[assign.loc[df.txn_id].values==c].copy(), initial_state,
                    seed+1000*c)
             for c in range(K_CLIENTS)]
    registry=[]; ref_emb={c.id:None for c in clients}
    global_class_protos={}
    rows=[]; proto_rows=[]; global_params=[
        p.detach().clone() for p in clients[0].backbone.parameters()
    ]
    groups={c.id:0 for c in clients}          # v4: persistent, refreshed periodically
    conf_windows = {c.id: [] for c in clients}          # CDA-FedAvg only
    rehearsal    = {c.id: pd.DataFrame() for c in clients}  # CDA-FedAvg only
    for w in range(N_WINDOWS):
        # ---- TEST window w before training on it ----
        window_scores = {c.id: None for c in clients}
        for c in clients:
            sub,score,emb=c.predict(w, registry if method=="fedtypo" else None)  # noreg: no boost
            if len(sub):
                rows.append(pd.DataFrame(dict(window=w,client=c.id,score=score,
                    label=sub.label.values,typ=sub.typology_id.values)))
                window_scores[c.id] = score
            if method in ("fedtypo","fedtypo_noreg") and emb is not None:
                e=emb.numpy(); c.signal1(e, ref_emb[c.id]); ref_emb[c.id]=e
        # ---- TRAIN on visible labels ----
        for c in clients:
            c.train_round(w, mu_prox=MU_PROX if method=="fedprox" else 0.0,
                          global_state=global_params,
                          global_class_protos=global_class_protos
                              if method=="fedproto" else None,
                          mu_proto=MU_PROTO if method=="fedproto" else 0.0)
        # ---- CDA-FedAvg: confidence tracking, drift detection, rehearsal ----
        if method == "cda_fedavg":
            for c in clients:
                sc = window_scores[c.id]
                if sc is not None and len(sc):
                    conf_windows[c.id].append(float(np.mean(np.maximum(sc, 1 - sc))))
                # grow rehearsal buffer from this window's newly confirmed labels
                newly_conf = c.txns[(c.txns.confirmed_window==w)]
                if len(newly_conf):
                    pos = newly_conf[newly_conf.label==1]
                    neg = newly_conf[newly_conf.label==0]
                    rehearsal[c.id] = pd.concat([rehearsal[c.id], pos, neg]).tail(
                        10**9)  # keep all positives; cap applied below
                    r = rehearsal[c.id]
                    pos_r = r[r.label==1]
                    neg_r = r[r.label==0].tail(REHEARSAL_NEG_CAP)
                    rehearsal[c.id] = pd.concat([pos_r, neg_r])
                if len(conf_windows[c.id]) >= 2*CDA_DELTA+1 and cda_drift_detected(conf_windows[c.id]):
                    conf_windows[c.id] = []   # reset sliding window, per Alg. 4
                    buf = rehearsal[c.id]
                    if len(buf) >= 8:
                        for _ in range(CDA_R_ROUNDS):
                            src,dst,ef = c._tensors(buf)
                            _,emb = c.backbone(c.mem.detach(), src, dst, ef)
                            logit = c.head(emb)
                            y = torch.tensor(buf.label.values, dtype=torch.float32, device=DEV)
                            loss = focal_loss(logit, y)
                            c.opt.zero_grad(); loss.backward(); c.opt.step()
        # ---- FEDERATE ----
        if method in ("fedavg","fedprox","cda_fedavg"):
            sample_weights=[c.last_train_count for c in clients]
            global_params=fedavg_aggregate(clients,weights=sample_weights)
            for c in clients: c.reset_optimizer()
        elif method=="fedproto":
            local=[c.class_prototypes(w) for c in clients]
            merged={}
            for cls in (0,1):
                available=[entry[cls] for entry in local if cls in entry]
                if available:
                    total=sum(count for _,count in available)
                    mean=sum(count*proto for proto,count in available)/total
                    merged[cls]=mean/(np.linalg.norm(mean)+1e-9)
            global_class_protos=merged
        elif method in ("fedtypo","fedtypo_noreg"):
            fracs=[]
            for c in clients:
                if c.pos_seen < 30 or w < 3:
                    fracs.append(0.0); continue
                frac,newp=c.signal2_novelty(w)
                fracs.append(round(frac,2))
                if newp is not None and frac>NOVELTY_GATE:
                    if not registry or max(float(newp@r) for r in registry)<0.9:
                        registry.append(newp); registry_log.append(dict(window=w,client=c.id,frac=frac))
            for c in clients:
                c.build_prototypes(w)
                if c.protos is not None:
                    proto_rows.append(dict(
                        window=w,client=c.id,purity=c.proto_purity,
                        nmi=c.proto_nmi,ari=c.proto_ari,
                        confirmed_positives=c.pos_seen))
            if w % 3 == 0:
                groups=cluster_clients(clients)
            wts=[max(1,c.pos_seen)*max(0.05,c.stability) for c in clients]
            fedavg_aggregate(clients,weights=wts,groups=groups)
            for c in clients: c.reset_optimizer()
            n_proto=sum(c.protos is not None for c in clients)
            conf_pos=[int(c.txns.loc[c.txns.confirmed_window<=w,"label"].sum()) for c in clients]
            print(f"  [diag w={w}] confirmed_pos/client={conf_pos} "
                  f"protos_built={n_proto}/{len(clients)} registry={len(registry)} "
                  f"novelty_frac={fracs} clusters={groups} "
                  f"stability={[round(c.stability,2) for c in clients]}")
        if verbose and w%5==0: print(f"  [{method}] window {w}/{N_WINDOWS}")
    return (pd.concat(rows,ignore_index=True), pd.DataFrame(registry_log),
            pd.DataFrame(proto_rows))

def metrics_table(res):
    out=[]
    for (w,client),g in res.groupby(["window","client"]):
        auprc=average_precision_score(g.label,g.score) if g.label.sum()>0 else np.nan
        topk=g.nlargest(min(BUDGET_K,len(g)),"score")
        out.append(dict(window=w,client=client,auprc=auprc,
                        p_at_budget=topk.label.mean() if len(topk) else np.nan,
                        positives=int(g.label.sum()),transactions=len(g)))
    return pd.DataFrame(out)

# ===== cell10_helpers_local.py =====
# ---------------- 9. Helpers needed for the sensitivity sweep ----------------
OUT = os.environ.get("FT_OUT", "/tmp/ml")
os.makedirs(f"{OUT}/results", exist_ok=True)

def per_window(r):
    client_metrics=metrics_table(r)
    client_metrics=client_metrics[
        (client_metrics.window>=EVAL_START_W)&
        (client_metrics.window<VALID_W)
    ]
    return (client_metrics.groupby("window",as_index=False)
            .agg(auprc=("auprc","mean"),
                 p_at_budget=("p_at_budget","mean"),
                 valid_clients=("auprc","count"),
                 positives=("positives","sum"),
                 transactions=("transactions","sum")))

# ---------------- 11. Submission experiment matrix ----------------
from scipy.stats import wilcoxon

METHODS_FULL = (["local_only","fedavg","fedproto","fedtypo"] if FAST_DEV else
                ["local_only","fedavg","fedprox","fedproto","fedtypo_noreg",
                 "fedtypo","cda_fedavg"])
EXPERIMENTS_FULL = [
    (condition,seed)
    for condition in ("control","drift")
    for seed in SUBMISSION_SEEDS
]
run_name=('tifs_submission_smoke_v3' if FAST_DEV else 'tifs_submission_v2')
RES2 = f"{OUT}/results/{run_name}_{DATASET}"
os.makedirs(RES2, exist_ok=True)
def file_sha256(path, block_size=8*1024*1024):
    digest=hashlib.sha256()
    with open(path,"rb") as handle:
        for block in iter(lambda: handle.read(block_size),b""):
            digest.update(block)
    return digest.hexdigest()

_input_paths=(
    [("transactions",IBM_PATH),("patterns",IBM_PATTERNS)]
    if DATASET=="ibm" else [("transactions",SAMLD_PATH)]
)
with open(f"{RES2}/environment.json","w") as f:
    json.dump({
        "dataset":DATASET,
        "inputs":[{
            "role":role,
            "file":os.path.basename(path),
            "bytes":os.path.getsize(path),
            "sha256":file_sha256(path),
        } for role,path in _input_paths],
        "script_sha256":file_sha256(__file__),
        "seeds":list(SUBMISSION_SEEDS),
        "python":os.sys.version,
        "torch":torch.__version__,
        "cuda":torch.version.cuda,
        "numpy":np.__version__,
        "pandas":pd.__version__,
        "sklearn":__import__("sklearn").__version__,
        "scipy":__import__("scipy").__version__,
        "networkx":nx.__version__,
        "device":torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "config":{
            "clients":K_CLIENTS,"dimension":D_MODEL,
            "natural_windows":N_WINDOWS,"reporting_target_windows":N_WINDOWS_CAP,
            "evaluation_start_window":EVAL_START_W,
            "window_frequency":WINDOW_FREQ,"local_epochs":LOCAL_EPOCHS,
            "learning_rate":LR,"focal_gamma":FOCAL_GAMMA,
            "alert_tpr":ALERT_TPR,"alert_fpr":ALERT_FPR,
            "delay_median_cycles":DELAY_MEDIAN_WINDOWS,
            "max_prototypes":M_PROTO,"min_confirmed_positive":MIN_CONF_POS,
            "rho":RHO_DAMP,"registry_beta":BETA_REGISTRY,
            "novelty_gate":NOVELTY_GATE,"fedproto_mu":MU_PROTO,
            "budget":BUDGET_K,
        }},f,indent=2)

def budget_table(res, budgets=(10,25,50,100,200)):
    res=res[(res.window>=EVAL_START_W)&(res.window<VALID_W)]
    rows=[]
    for (w,client),g in res.groupby(["window","client"]):
        for budget in budgets:
            top=g.nlargest(min(budget,len(g)),"score")
            rows.append(dict(window=w,client=client,budget=budget,
                             precision=top.label.mean(),
                             recall=(top.label.sum()/g.label.sum())
                                    if g.label.sum()>0 else np.nan))
    return pd.DataFrame(rows)

def typology_table(res, budget=BUDGET_K):
    # This is a full-stream diagnostic rather than a primary evaluation
    # metric; retain the warm-up positives to describe typology coverage.
    ranked=res.copy()
    ranked["rank"]=ranked.groupby(["window","client"]).score.rank(
        method="first",ascending=False)
    pos=ranked[(ranked.label==1)&(ranked.typ>=0)].copy()
    if len(pos)==0: return pd.DataFrame()
    pos["captured"]=(pos["rank"]<=budget).astype(int)
    return (pos.groupby("typ",as_index=False)
            .agg(positives=("label","size"),
                 captured=("captured","sum"),
                 capture_rate=("captured","mean"),
                 mean_score=("score","mean")))

def inoculation_table(res, events, budget=BUDGET_K):
    if events is None or len(events)==0: return pd.DataFrame()
    ranked=res.copy()
    ranked["rank"]=ranked.groupby(["window","client"]).score.rank(
        method="first",ascending=False)
    rows=[]
    d4=events[(events.kind=="D4")&(events.stagger>0)]
    for event in d4.itertuples():
        q=ranked[(ranked.client==event.client)&
                 (ranked.typ==event.typology)&
                 (ranked.label==1)&
                 (ranked.window>=event.window)]
        rows.append(dict(client=event.client,typology=event.typology,
                         onset=event.window,stagger=event.stagger,
                         positives=len(q),
                         captured=int((q["rank"]<=budget).sum()),
                         capture_rate=float((q["rank"]<=budget).mean())
                                      if len(q) else np.nan))
    return pd.DataFrame(rows)

for cond, seed in EXPERIMENTS_FULL:
    tag=f"{cond}_s{seed}"; d_out=f"{RES2}/{tag}"; os.makedirs(d_out, exist_ok=True)
    missing_methods=[
        method for method in METHODS_FULL
        if not os.path.exists(f"{d_out}/DONE_{method}")
    ]
    if not missing_methods:
        print(f"### {tag}: every method already complete, skipping data build")
        continue
    print(f"### building data for {tag}")
    df_e, assign_e, events_e = build_data(cond, seed)
    events_e.to_csv(f"{d_out}/drift_events.csv", index=False)
    client_values=assign_e.loc[df_e.txn_id].to_numpy()
    ownership=(pd.DataFrame({"account":df_e.src_account.to_numpy(),
                             "client":client_values})
               .groupby("account").client.nunique().max())
    if ownership!=1:
        raise RuntimeError("source-account ownership was split across clients")
    prof=(pd.DataFrame({"client":client_values,
                        "label":df_e.label.to_numpy(),
                        "confirmed_window":df_e.confirmed_window.to_numpy()})
          .groupby("client",as_index=False)
          .agg(transactions=("label","size"),positives=("label","sum"),
               median_confirmation=("confirmed_window","median")))
    prof.to_csv(f"{d_out}/client_profile.csv",index=False)
    typ=(pd.DataFrame({"client":client_values,
                       "label":df_e.label.to_numpy(),
                       "typology":df_e.typology_id.to_numpy()})
         .query("label == 1")
         .groupby(["client","typology"]).size().unstack(fill_value=0))
    typ.to_csv(f"{d_out}/client_typology_counts.csv")
    for m in METHODS_FULL:
        done=f"{d_out}/DONE_{m}"
        if os.path.exists(done):
            print(f"  {tag}/{m}: already complete, skipping"); continue
        print(f"### running {tag} / {m}",flush=True)
        r, rlog, pdiag = run_method(
            "local" if m=="local_only" else m,
            df_e,assign_e,seed=seed,verbose=False)
        metrics_table(r).to_csv(f"{d_out}/client_metrics_{m}.csv",index=False)
        per_window(r).to_csv(f"{d_out}/window_metrics_{m}.csv",index=False)
        budget_table(r).to_csv(f"{d_out}/budget_metrics_{m}.csv",index=False)
        typology_table(r).to_csv(f"{d_out}/typology_metrics_{m}.csv",index=False)
        inoculation_table(r,events_e).to_csv(
            f"{d_out}/inoculation_{m}.csv",index=False)
        if len(rlog):
            rlog.to_csv(f"{d_out}/registry_{m}.csv",index=False)
        if len(pdiag):
            pdiag.to_csv(f"{d_out}/prototype_{m}.csv",index=False)
        if os.environ.get("FT_SAVE_PREDICTIONS","0")=="1" and seed==42:
            r.to_csv(f"{d_out}/preds_{m}.csv.gz",index=False,compression="gzip")
        open(done,"w").write("ok")
        del r
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    print(f"### {tag} complete",flush=True)
print("ALL TIFS SUBMISSION EXPERIMENTS COMPLETE")

# ---------------- 12. Seed-level aggregation and inference ----------------
def bootstrap_ci(values,n_boot=10000,seed=20260725):
    x=np.asarray(values,dtype=float)
    x=x[np.isfinite(x)]
    if len(x)==0: return np.nan,np.nan
    br=np.random.default_rng(seed)
    means=np.array([br.choice(x,size=len(x),replace=True).mean()
                    for _ in range(n_boot)])
    return tuple(np.quantile(means,[0.025,0.975]))

seed_rows=[]
for cond in ("control","drift"):
    for seed in SUBMISSION_SEEDS:
        for method in METHODS_FULL:
            f=f"{RES2}/{cond}_s{seed}/window_metrics_{method}.csv"
            t=pd.read_csv(f)
            # Compatibility with resumable runs produced before window 0 was
            # formally marked as warm-up: never use it for inference even if
            # an existing aggregate CSV still contains that row.
            t=t[(t.window>=EVAL_START_W)&(t.window<VALID_W)]
            seed_rows.append(dict(
                condition=cond,seed=seed,method=method,
                auprc=t.auprc.mean(),
                p_at_budget=t.p_at_budget.mean()))
seed_df=pd.DataFrame(seed_rows)
seed_df.to_csv(f"{RES2}/seed_summary.csv",index=False)

summary_rows=[]
for (cond,method),g in seed_df.groupby(["condition","method"]):
    lo_a,hi_a=bootstrap_ci(g.auprc)
    lo_p,hi_p=bootstrap_ci(g.p_at_budget)
    summary_rows.append(dict(
        condition=cond,method=method,n_seeds=len(g),
        auprc_mean=g.auprc.mean(),auprc_std=g.auprc.std(ddof=1),
        auprc_ci_lo=lo_a,auprc_ci_hi=hi_a,
        p_at_budget_mean=g.p_at_budget.mean(),
        p_at_budget_std=g.p_at_budget.std(ddof=1),
        p_at_budget_ci_lo=lo_p,p_at_budget_ci_hi=hi_p))
summary_df=pd.DataFrame(summary_rows)
summary_df.to_csv(f"{RES2}/method_summary.csv",index=False)

tests=[]
for cond in ("control","drift"):
    ours=(seed_df[(seed_df.condition==cond)&(seed_df.method=="fedtypo")]
          .set_index("seed"))
    raw=[]
    for baseline in [m for m in METHODS_FULL if m!="fedtypo"]:
        other=(seed_df[(seed_df.condition==cond)&(seed_df.method==baseline)]
               .set_index("seed"))
        common=ours.index.intersection(other.index)
        if len(common)<2:
            stat,p=np.nan,1.0
        else:
            diff=(ours.loc[common,"auprc"]-other.loc[common,"auprc"]).to_numpy()
            if np.allclose(diff,0):
                stat,p=0.0,1.0
            else:
                stat,p=wilcoxon(ours.loc[common,"auprc"],
                                other.loc[common,"auprc"],
                                alternative="greater",method="auto")
        raw.append((baseline,float(stat),float(p),len(common)))
    order=np.argsort([x[2] for x in raw])
    adjusted=np.empty(len(raw))
    running=0.0
    for rank,idx in enumerate(order):
        candidate=min(1.0,(len(raw)-rank)*raw[idx][2])
        running=max(running,candidate)
        adjusted[idx]=running
    for idx,(baseline,stat,p,n) in enumerate(raw):
        ours_mean=ours.auprc.mean()
        base_mean=(seed_df[(seed_df.condition==cond)&
                           (seed_df.method==baseline)].auprc.mean())
        tests.append(dict(condition=cond,baseline=baseline,n_seeds=n,
                          statistic=stat,p_raw=p,p_holm=adjusted[idx],
                          relative_gain=(ours_mean/base_mean-1)
                                        if base_mean else np.nan))
pd.DataFrame(tests).to_csv(f"{RES2}/seed_level_tests.csv",index=False)

print(summary_df.to_string(index=False))
print(pd.DataFrame(tests).to_string(index=False))
