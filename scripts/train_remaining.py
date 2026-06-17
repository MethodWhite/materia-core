"""Continue training: unified, nano, science modules"""
import os, sys, json, time, csv, gzip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = '/home/methodwhite/MATERIA'
DEVICE = torch.device('cpu')
torch.set_num_threads(4)
PLOTS_DIR = os.path.join(BASE, 'docs', 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)
sys.path.insert(0, os.path.join(BASE, 'models'))
from materia_v3_full import count_params

def log(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

class CompleteModel(nn.Module):
    def __init__(self, vocab_size=800, dim=256, n_layers=3, n_heads=8, n_kv=4,
                 jepa_dim=256, synapsis_slots=128, use_snn=True, use_ssm=True,
                 use_jepa=True, use_synapsis=True, use_hsaq=True):
        super().__init__()
        self.dim=dim;self.use_snn=use_snn;self.use_ssm=use_ssm;self.use_jepa=use_jepa
        self.use_synapsis=use_synapsis;self.use_hsaq=use_hsaq
        from materia_v3_full import TransformerBlock,SNNLayer,SSMBlock,JEPA,SynapsisMemory,HSAQ
        self.tok_emb=nn.Embedding(vocab_size,dim)
        self.layers=nn.ModuleList([TransformerBlock(dim,n_heads,n_kv) for _ in range(n_layers)])
        if use_snn:self.snn=SNNLayer(dim)
        if use_ssm:self.ssm=SSMBlock(dim)
        if use_jepa:self.jepa=JEPA(dim,jepa_dim)
        if use_synapsis:self.synapsis=SynapsisMemory(dim,synapsis_slots)
        if use_hsaq:self.hsaq=HSAQ(sparsity=0.3)
        self.norm=nn.RMSNorm(dim);self.head=nn.Linear(dim,vocab_size,bias=False)
    def forward(self,x,mask=None):
        x=self.tok_emb(x)
        if self.use_hsaq:x=self.hsaq(x)
        for l in self.layers:x=l(x,mask)
        if self.use_synapsis:x=self.synapsis(x)
        if self.use_snn:x_enh,rate=self.snn(x[:,-1:]);x=torch.cat([x[:,:-1],x_enh],dim=1)
        if self.use_ssm:x=self.ssm(x)
        if self.use_jepa:_,x=self.jepa(x)
        return self.head(self.norm(x))

def build_vocab(texts,vocab_size=800):
    chars=set()
    for t in texts:
        for c in t:chars.add(c)
    chars=sorted(chars)[:vocab_size-4]
    stoi={c:i+4 for i,c in enumerate(chars)}
    stoi['<PAD>']=0;stoi['<BOS>']=1;stoi['<EOS>']=2;stoi['<UNK>']=3
    return stoi,{i:c for c,i in stoi.items()}

class TextDataset(Dataset):
    def __init__(self,texts,stoi,seq_len=64):
        self.seq_len=seq_len;self.data=[]
        for text in texts:
            ids=[stoi.get(c,3) for c in text]
            actual_seq = min(seq_len, max(8, len(ids)-1))
            if len(ids)>actual_seq+1:
                for i in range(0,len(ids)-actual_seq,actual_seq//2):
                    self.data.append(ids[i:i+actual_seq+1])
            elif len(ids)>2:
                self.data.append(ids)
    def __len__(self):return max(1,len(self.data))
    def __getitem__(self,idx):
        seq=self.seq_len
        ids=self.data[idx%len(self.data)]
        if len(ids)<seq+1:
            pad=[0]*(seq+1-len(ids))
            ids=ids+pad
        else:
            ids=ids[:seq+1]
        ids=torch.tensor(ids[:seq+1],dtype=torch.long)
        return ids[:-1],ids[1:]

def compute_accuracy(logits,targets,ignore_idx=0):
    preds=logits.argmax(dim=-1);mask=targets!=ignore_idx
    correct=(preds[mask]==targets[mask]).float().sum();total=mask.sum()
    return (correct/total).item() if total>0 else 0.0

def load_texts(filepath,max_lines=5000,min_len=10):
    texts=[]
    with open(filepath,'r',encoding='utf-8',errors='ignore') as f:
        for line in f:
            line=line.strip()
            if len(line)>min_len:
                texts.append(line)
                if len(texts)>=max_lines:break
    return texts

REMOTE_SPECS = {
    "materia-v3-unified.materia": {
        "name":"MATERIA V3 - Unified Fine-tune","dim":256,"n_layers":2,"n_heads":8,"n_kv":4,
        "jepa_dim":128,"synapsis_slots":64,"use_snn":True,"use_ssm":True,"use_jepa":True,
        "use_synapsis":True,"use_hsaq":True,"dataset":"Wikipedia ES/EN","max_samples":2000,
        "epochs":3,"batch_size":8,"seq_len":64,"ollama_model":"materia-v3-unified:latest",
        "domain":"general","capabilities":["llm","snn","ssm","jepa"]
    },
    "materia-v3-nano.materia": {
        "name":"MATERIA V3 - Nano Fine-tune","dim":128,"n_layers":2,"n_heads":4,"n_kv":2,
        "jepa_dim":64,"synapsis_slots":32,"use_snn":True,"use_ssm":False,"use_jepa":True,
        "use_synapsis":True,"use_hsaq":True,"dataset":"C4 EN (1000 textos)","max_samples":1000,
        "epochs":3,"batch_size":16,"seq_len":64,"ollama_model":"materia-v3-nano:latest",
        "domain":"general","capabilities":["inferencia rapida","razonamiento basico"]
    },
    "science-v3.materia": {
        "name":"MATERIA Science V3 - Fine-tune","dim":256,"n_layers":2,"n_heads":8,"n_kv":4,
        "jepa_dim":128,"synapsis_slots":64,"use_snn":False,"use_ssm":False,"use_jepa":True,
        "use_synapsis":True,"use_hsaq":True,"dataset":"reasoning_dataset.txt","max_samples":168,
        "epochs":20,"batch_size":8,"seq_len":128,"ollama_model":"materia-science:latest",
        "domain":"science","capabilities":["conocimiento cientifico","razonamiento logico"]
    },
}

def train_model(model,loader,val_loader,model_key,spec,stoi):
    csv_path=os.path.join(BASE,'logs',f'{model_key.replace(".","_")}_log.csv')
    with open(csv_path,'w',newline='') as cf:
        w=csv.writer(cf);w.writerow(['step','epoch','loss','accuracy','grad_norm','spike_rate','lr'])
        step=0
        opt=optim.AdamW(model.parameters(),lr=5e-4,weight_decay=0.01)
        sch=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=spec['epochs']*len(loader))
        for epoch in range(spec['epochs']):
            model.train();tl,ta=0.0,0.0
            for i,(x,y) in enumerate(loader):
                opt.zero_grad()
                logits=model(x)
                loss=F.cross_entropy(logits.view(-1,logits.size(-1)),y.view(-1),ignore_index=0)
                loss.backward()
                gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                opt.step();sch.step()
                acc=compute_accuracy(logits,y)
                sr=0.0
                if hasattr(model,'snn')and model.use_snn:
                    with torch.no_grad():
                        _,sr=model.snn(model.tok_emb(x[:,-1:]))
                        sr=sr.item() if torch.is_tensor(sr) else sr
                tl+=loss.item();ta+=acc;step+=1
                w.writerow([step,epoch+1,f'{loss.item():.4f}',f'{acc:.4f}',f'{gn:.4f}',f'{sr:.3f}',f'{sch.get_last_lr()[0]:.2e}'])
                if (i+1)%50==0:log(f"  E{epoch+1}/{spec['epochs']} [{i+1}/{len(loader)}] loss={loss.item():.4f} acc={acc:.4f} gn={gn:.2f} sr={sr:.3f}")
            tl/=len(loader);ta/=len(loader)
            log(f"  -> E{epoch+1}/{spec['epochs']}: train_loss={tl:.4f} train_acc={ta:.4f}")
            if val_loader:
                model.eval();vl,va,vs=0.0,0.0,0
                with torch.no_grad():
                    for x,y in val_loader:
                        lo=model(x);l=F.cross_entropy(lo.view(-1,lo.size(-1)),y.view(-1),ignore_index=0)
                        vl+=l.item();va+=compute_accuracy(lo,y);vs+=1
                if vs>0:log(f"       val_loss={vl/vs:.4f} val_acc={va/vs:.4f}")
    # Plot
    import pandas as pd
    df=pd.read_csv(csv_path)
    if len(df)>2:
        fig,ax=plt.subplots(2,2,figsize=(14,10))
        ax[0,0].plot(df['step'],df['loss']);ax[0,0].set_title('Loss');ax[0,0].grid(True,alpha=0.3)
        ax[0,1].plot(df['step'],df['accuracy']);ax[0,1].set_title('Accuracy');ax[0,1].grid(True,alpha=0.3)
        ax[1,0].plot(df['step'],df['grad_norm']);ax[1,0].set_title('Gradient Norm');ax[1,0].grid(True,alpha=0.3)
        ax[1,1].plot(df['step'],df['spike_rate']);ax[1,1].set_title('Spike Rate');ax[1,1].grid(True,alpha=0.3)
        plt.suptitle(f'Training: {model_key}',fontsize=14);plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR,f'{model_key.replace(".","_")}_training.png'),dpi=150);plt.close()
    return tl,ta

if __name__=='__main__':
    log("Continuing MATERIA V3 training (unified, nano, science)...")
    c4=load_texts(os.path.join(BASE,'data/multilingual/tokenizer/c4_en.txt'),2000,10)
    es=load_texts(os.path.join(BASE,'data/multilingual/tokenizer/wiki_es.txt'),1000,10)
    en=load_texts(os.path.join(BASE,'data/multilingual/tokenizer/wiki_en.txt'),1000,10)
    rs=load_texts(os.path.join(BASE,'data/reasoning_dataset.txt'),168,10)
    all_texts=c4+es+en+rs
    stoi,itos=build_vocab(all_texts,800)
    log(f"Vocab: {len(stoi)} | Total texts: {len(all_texts)}")

    for mk,spec in REMOTE_SPECS.items():
        log(f"\n{'='*60}\nTRAINING: {mk}")
        if 'unified' in mk:
            texts=load_texts(os.path.join(BASE,'data/multilingual/tokenizer/wiki_es.txt'),1000,10)+load_texts(os.path.join(BASE,'data/multilingual/tokenizer/wiki_en.txt'),1000,10)
        elif 'nano' in mk:
            texts=load_texts(os.path.join(BASE,'data/multilingual/tokenizer/c4_en.txt'),1000,10)
        elif 'science' in mk:
            texts=load_texts(os.path.join(BASE,'data/reasoning_dataset.txt'),168,10)
        log(f"  Textos: {len(texts):,}")
        if len(texts)<5:
            log(f"  SKIP: too few texts ({len(texts)})")
            continue
        split=int(len(texts)*0.9)
        train_ds=TextDataset(texts[:split],stoi,spec['seq_len'])
        val_ds=TextDataset(texts[split:],stoi,spec['seq_len'])
        train_loader=DataLoader(train_ds,batch_size=spec['batch_size'],shuffle=True,drop_last=True)
        val_loader=DataLoader(val_ds,batch_size=spec['batch_size'],shuffle=False,drop_last=True) if len(val_ds)>spec['batch_size'] else None

        model=CompleteModel(vocab_size=len(stoi),dim=spec['dim'],n_layers=spec['n_layers'],
            n_heads=spec['n_heads'],n_kv=spec['n_kv'],jepa_dim=spec['jepa_dim'],
            synapsis_slots=spec['synapsis_slots'],use_snn=spec['use_snn'],
            use_ssm=spec['use_ssm'],use_jepa=spec['use_jepa'],
            use_synapsis=spec['use_synapsis'],use_hsaq=spec['use_hsaq'])
        params=count_params(model);log(f"  Params: {params:,}")

        final_loss,final_acc=train_model(model,train_loader,val_loader,mk,spec,stoi)

        # Save .materia
        mp=os.path.join(BASE,'models',mk)
        module={
            "materia":"umbra_sub_agent","name":spec['name'],"version":"3.0.0",
            "architecture":"gqa+rope+swiglu+lif_snn+ssm+jepa+synapsis+hsaq",
            "config":{
                "vocab_size":len(stoi),"params":params,"snn_type":"LIF real",
                "dim":spec['dim'],"n_layers":spec['n_layers'],"n_heads":spec['n_heads'],
                "n_kv":spec['n_kv'],"jepa_dim":spec['jepa_dim'],"synapsis_slots":spec['synapsis_slots'],
                "use_snn":spec['use_snn'],"use_ssm":spec['use_ssm'],"use_jepa":spec['use_jepa'],
                "use_synapsis":spec['use_synapsis'],"use_hsaq":spec['use_hsaq'],"max_seq_len":spec['seq_len'],
            },
            "ollama_model":spec['ollama_model'],"domain":spec['domain'],
            "capabilities":spec['capabilities'],
            "training":{"date":time.strftime('%Y-%m-%d'),"dataset":spec['dataset'],
                       "epochs":spec['epochs'],"final_loss":round(final_loss,4),"final_accuracy":round(final_acc,4)},
            "weights":{k:v.cpu().numpy() for k,v in model.state_dict().items()},"tokenizer":stoi
        }
        class NumpyEncoder(json.JSONEncoder):
            def default(self,o):
                if isinstance(o,np.ndarray):return o.tolist()
                if isinstance(o,np.floating):return float(o)
                if isinstance(o,np.integer):return int(o)
                return super().default(o)
        with gzip.open(mp,'wb') as f:
            f.write(json.dumps(module,cls=NumpyEncoder).encode('utf-8'))
        log(f"  Saved: {mp} ({os.path.getsize(mp)//1024}KB)")
        log(f"  ✓ {mk} completo (loss={final_loss:.4f}, acc={final_acc:.4f})")
    log("\nDone!")
