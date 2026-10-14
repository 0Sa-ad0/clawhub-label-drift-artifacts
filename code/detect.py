"""detect.py - detectors trained on scanner labels, evaluated against pre- and post-rescan labels.

Experiments (one per label source):
  ss  SkillSpector (rescanned 14-21 Sep 2026)   <- the label-drift experiment
  vt  VirusTotal                                <- control
  st  static analysis                           <- control
For each: train on *_old labels (pre-rescan) of the train split; select threshold on valid (*_old);
score the test split and report metrics against *_old AND *_new labels for the same test rows.
Also trains on *_new (post-rescan) to show what an updated detector would achieve.
Secondary: registry verdict (verdict_any), rare positives -> PR-AUC.

Models: 'tfidf' (char+word TF-IDF + logistic regression, CPU) and 'bert' (fine-tuned transformer, GPU).
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def load(path):
    df = pd.read_parquet(path)
    df["text"] = df["md_text"].fillna("") + "\n\n#### BUNDLE\n" + df["bundle_text"].fillna("")
    return df


def metrics(y, s, thr):
    y = np.asarray(y).astype(int)
    p = (np.asarray(s) >= thr).astype(int)
    out = {"n": int(len(y)), "pos": int(y.sum()), "pos_rate": float(y.mean()) if len(y) else float("nan")}
    if len(np.unique(y)) == 2:
        out.update(roc_auc=float(roc_auc_score(y, s)), pr_auc=float(average_precision_score(y, s)))
    out.update(f1=float(f1_score(y, p, zero_division=0)), precision=float(precision_score(y, p, zero_division=0)),
               recall=float(recall_score(y, p, zero_division=0)), accuracy=float((p == y).mean()))
    return out


def best_threshold(y, s):
    """Threshold maximising F1 on the validation set."""
    y = np.asarray(y).astype(int)
    qs = np.unique(np.quantile(s, np.linspace(0.01, 0.99, 197)))
    f1s = [f1_score(y, (s >= t).astype(int), zero_division=0) for t in qs]
    return float(qs[int(np.argmax(f1s))])


def bootstrap_ci(y, s, fn, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y), np.asarray(s)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) == 2:
            vals.append(fn(y[i], s[i]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else [float("nan")] * 2


# ----------------------------------------------------------------------------- models

class TfidfModel:
    name = "tfidf"

    def fit(self, tr_text, tr_y, seed=0):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        self.vw = TfidfVectorizer(max_features=300_000, ngram_range=(1, 2), min_df=3, sublinear_tf=True,
                                  token_pattern=r"(?u)\b\w[\w\-\.]+\b", dtype=np.float32)
        X = self.vw.fit_transform(tr_text)
        self.clf = LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced", random_state=seed)
        self.clf.fit(X, tr_y)
        return self

    def score(self, text):
        X = self.vw.transform(text)
        return self.clf.predict_proba(X)[:, 1]


class BertModel:
    name = "bert"

    def __init__(self, model_name="answerdotai/ModernBERT-base", max_len=1024, epochs=2, batch=16, lr=3e-5,
                 tiny=False):
        self.model_name, self.max_len, self.epochs, self.batch, self.lr = model_name, max_len, epochs, batch, lr
        self.tiny = tiny  # local smoke test: random tiny network, same tokenizer and code path

    def fit(self, tr_text, tr_y, seed=0):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(self.model_name)
        if self.tiny:
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(self.model_name, num_labels=2, hidden_size=32, intermediate_size=64,
                                             num_hidden_layers=2, num_attention_heads=2)
            self.model = AutoModelForSequenceClassification.from_config(cfg).to(self.dev)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2).to(self.dev)
        y = np.asarray(tr_y).astype(int)
        w = torch.tensor([1.0, float(max(1.0, (1 - y.mean()) / max(y.mean(), 1e-6)) ** 0.5)], dtype=torch.float32, device=self.dev)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        steps = self.epochs * int(np.ceil(len(y) / self.batch))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=steps, pct_start=0.1)
        amp = self.dev == "cuda"
        dt = torch.bfloat16 if (amp and torch.cuda.is_bf16_supported()) else torch.float16
        texts = list(tr_text)
        self.model.train()
        step = 0
        for ep in range(self.epochs):
            order = np.random.permutation(len(y))
            t0, losses = time.time(), []
            for b in range(0, len(order), self.batch):
                idx = order[b:b + self.batch]
                enc = self.tok([texts[i] for i in idx], truncation=True, max_length=self.max_len, padding=True,
                               return_tensors="pt").to(self.dev)
                with torch.autocast(device_type="cuda", dtype=dt, enabled=amp):
                    logits = self.model(**enc).logits.float()
                loss = torch.nn.functional.cross_entropy(logits, torch.tensor(y[idx], device=self.dev), weight=w)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                sched.step()
                step += 1
                losses.append(float(loss))
                if step % 200 == 0:
                    log(f"  epoch {ep + 1} step {step}/{steps} loss {np.mean(losses[-200:]):.4f}")
            log(f"  epoch {ep + 1} done, mean loss {np.mean(losses):.4f}, {time.time() - t0:.0f}s")
        return self

    def score(self, text):
        import torch
        self.model.eval()
        out, texts = [], list(text)
        amp = self.dev == "cuda"
        dt = torch.bfloat16 if (amp and torch.cuda.is_bf16_supported()) else torch.float16
        with torch.inference_mode():
            for b in range(0, len(texts), self.batch * 2):
                enc = self.tok(texts[b:b + self.batch * 2], truncation=True, max_length=self.max_len, padding=True,
                               return_tensors="pt").to(self.dev)
                with torch.autocast(device_type="cuda", dtype=dt, enabled=amp):
                    out.append(torch.softmax(self.model(**enc).logits.float(), -1)[:, 1].cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)


# ----------------------------------------------------------------------------- experiments

def run_label_drift(df, make_model, source, seed=0, max_train=None):
    """Train on {source}_old (and separately {source}_new); test against both label versions."""
    res = {"source": source, "seed": seed}
    both = df[df[f"{source}_old"].notna() & df[f"{source}_new"].notna()]
    tr, va, te = (both[both.split == s] for s in ("train", "valid", "test"))
    if max_train and len(tr) > max_train:
        tr = tr.sample(max_train, random_state=seed)
    res["sizes"] = {"train": len(tr), "valid": len(va), "test": len(te)}
    res["test_label_changed"] = int((te[f"{source}_old"] != te[f"{source}_new"]).sum())
    for train_on in ("old", "new"):
        col = f"{source}_{train_on}"
        t0 = time.time()
        m = make_model().fit(tr.text, tr[col].astype(int), seed=seed)
        s_va, s_te = m.score(va.text), m.score(te.text)
        thr = best_threshold(va[col].astype(int), s_va)
        r = {"threshold": thr, "minutes": round((time.time() - t0) / 60, 2)}
        for eval_on in ("old", "new"):
            y = te[f"{source}_{eval_on}"].astype(int).values
            r[f"test_vs_{eval_on}"] = metrics(y, s_te, thr)
            r[f"test_vs_{eval_on}"]["roc_auc_ci"] = bootstrap_ci(y, s_te, roc_auc_score, n=300, seed=seed)
        res[f"trained_on_{train_on}"] = r
        res.setdefault("scores", {})[f"trained_on_{train_on}"] = s_te.astype(float).tolist()
        log(f"{m.name}/{source}/train={train_on}: "
            f"AUC vs old {r['test_vs_old'].get('roc_auc', float('nan')):.3f}, "
            f"vs new {r['test_vs_new'].get('roc_auc', float('nan')):.3f} | "
            f"F1 vs old {r['test_vs_old']['f1']:.3f}, vs new {r['test_vs_new']['f1']:.3f}")
    res["test_keys"] = te.k.tolist()
    return res


def run_verdict(df, make_model, seed=0):
    d = df.copy()
    tr, va, te = (d[d.split == s] for s in ("train", "valid", "test"))
    y_tr = tr.verdict_any.astype(int)
    m = make_model().fit(tr.text, y_tr, seed=seed)
    s_va, s_te = m.score(va.text), m.score(te.text)
    thr = best_threshold(va.verdict_any.astype(int), s_va) if va.verdict_any.sum() else 0.5
    r = {"source": "verdict", "seed": seed, "sizes": {"train": len(tr), "valid": len(va), "test": len(te)},
         "test": metrics(te.verdict_any.astype(int), s_te, thr), "threshold": thr}
    log(f"{m.name}/verdict: test pos {r['test']['pos']}, PR-AUC {r['test'].get('pr_auc', float('nan')):.3f}, "
        f"AUC {r['test'].get('roc_auc', float('nan')):.3f}")
    return r


def run_all(data_path, out_dir, model="tfidf", seeds=(0,), sources=("ss", "vt", "st"), max_train=None,
            bert_kwargs=None, verdict=True):
    os.makedirs(out_dir, exist_ok=True)
    df = load(data_path)
    log(f"data: {len(df)} skill versions; splits {df.split.value_counts().to_dict()}")
    make = (lambda: TfidfModel()) if model == "tfidf" else (lambda: BertModel(**(bert_kwargs or {})))
    for seed in seeds:
        for src in sources:
            path = os.path.join(out_dir, f"{model}_{src}_s{seed}.json")
            if os.path.exists(path):
                log(f"skip {path} (done)")
                continue
            r = run_label_drift(df, make, src, seed=seed, max_train=max_train)
            json.dump(r, open(path, "w"))
        if verdict:
            path = os.path.join(out_dir, f"{model}_verdict_s{seed}.json")
            if not os.path.exists(path):
                json.dump(run_verdict(df, make, seed=seed), open(path, "w"))
    log("all done")


if __name__ == "__main__":
    import sys
    run_all(sys.argv[1], sys.argv[2], model=sys.argv[3] if len(sys.argv) > 3 else "tfidf",
            max_train=int(sys.argv[4]) if len(sys.argv) > 4 else None)
