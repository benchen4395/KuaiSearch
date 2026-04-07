import json
import numpy as np
from typing import List, Dict, Tuple
from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score

def load_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data

def load_dataset(path: str):
    anchors, sentences, labels = [], [], []

    with open(path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            q = str(obj.get("query", "")).strip()
            t = str(obj.get("text", "")).strip()
            r = obj.get("label", None)
            if not q or not t or r is None:
                continue
            try:
                y = int(r)
            except Exception:
                continue
            anchors.append(q)
            sentences.append(t)
            labels.append(y)
    dataset = Dataset.from_dict({
        "anchor": anchors,
        "sentence": sentences,
        "label": labels
    })
    return dataset
def compute_pair_sims_official(
    model: SentenceTransformer,
    pairs: List[Dict],
    query_prompt_name: str = "query",
    batch_size: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute aligned pair similarities using SentenceTransformer official API:
      sims[i] = model.similarity(q_emb[i], d_emb[i]) (diagonal of sim matrix)
    """
    queries = [x["query"] for x in pairs]
    texts = [x["text"] for x in pairs]
    labels = np.array([int(x["label"]) for x in pairs], dtype=int)

    q_emb = model.encode(
        queries,
        prompt_name=query_prompt_name,
        batch_size=batch_size,
        show_progress_bar=True,
    )
    t_emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
    )

    # Official similarity: returns a tensor-like matrix [n, n] for (q_emb, t_emb)
    sim_mat = model.similarity(q_emb, t_emb)

    # Convert to numpy and take diagonal (aligned pairs)
    if hasattr(sim_mat, "detach"):  # torch Tensor
        sim_mat = sim_mat.detach().cpu().numpy()
    else:
        sim_mat = np.asarray(sim_mat)

    sims = np.diag(sim_mat)
    return sims, labels
    
def safe_auc_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    """
    Compute AUC metrics safely:
    - roc_auc: ROC-AUC
    - pr_auc:  PR-AUC (Average Precision)
    If y_true has only one class, AUC is undefined; return NaN.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    if np.unique(y_true).size < 2:
        return {"roc_auc": float("nan"), "pr_auc": float("nan")}

    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
    }

    
def find_best_threshold(
    sims: np.ndarray,
    y_true: np.ndarray,
    metric: str = "macro_f1",
    num_candidates: int = 2001,
) -> Tuple[float, Dict[str, float]]:
    lo = float(np.min(sims))
    hi = float(np.max(sims))
    if hi - lo < 1e-12:
        thr = lo
        y_pred = (sims >= thr).astype(int)
        return thr, {
            "acc": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        }

    thresholds = np.linspace(lo, hi, num_candidates)

    best_thr = thresholds[0]
    best_score = -1.0
    best_stats = None

    for thr in thresholds:
        y_pred = (sims >= thr).astype(int)
        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        if metric == "acc":
            score = acc
        elif metric == "weighted_f1":
            score = weighted_f1
        else:
            score = macro_f1

        if score > best_score:
            best_score = score
            best_thr = float(thr)
            best_stats = {"acc": acc, "macro_f1": macro_f1, "weighted_f1": weighted_f1}
            
    best_stats.update(safe_auc_metrics(y_true, sims))
    return best_thr, best_stats


def evaluate_with_threshold(sims: np.ndarray, y_true: np.ndarray, thr: float) -> Dict[str, float]:
    y_pred = (sims >= thr).astype(int)
    stats = {
        "acc": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    # AUC uses continuous score; include it here for completeness
    stats.update(safe_auc_metrics(y_true, sims))
    return stats



def test(
    train_path,
    test_path,
    model,
    query_prompt_name,
    optimize_metric,
    batch_size,
):
    train_data = load_jsonl(train_path)
    test_data = load_jsonl(test_path)

    #model = SentenceTransformer(model_name)

    train_sims, y_train = compute_pair_sims_official(
        model, train_data, query_prompt_name=query_prompt_name, batch_size=batch_size
    )
    test_sims, y_test = compute_pair_sims_official(
        model, test_data, query_prompt_name=query_prompt_name, batch_size=batch_size
    )

    best_thr, train_best = find_best_threshold(train_sims, y_train, metric=optimize_metric)
    print(f"[Train] Best threshold = {best_thr:.6f} (optimized for {optimize_metric})")
    print(f"[Train] Accuracy     : {train_best['acc']:.6f}")
    print(f"[Train] Macro F1     : {train_best['macro_f1']:.6f}")
    print(f"[Train] Weighted F1  : {train_best['weighted_f1']:.6f}")
    print(f"[Train] ROC-AUC      : {train_best['roc_auc']:.6f}")
    print(f"[Train] PR-AUC       : {train_best['pr_auc']:.6f}")

    test_stats = evaluate_with_threshold(test_sims, y_test, best_thr)
    acc = test_stats["acc"]
    macro_f1 = test_stats["macro_f1"]
    weighted_f1 = test_stats["weighted_f1"]

    print("\n[Test] Using train-derived threshold (official similarity)")
    print(f"Accuracy     : {test_stats['acc']:.6f}")
    print(f"Macro F1     : {test_stats['macro_f1']:.6f}")
    print(f"Weighted F1  : {test_stats['weighted_f1']:.6f}")
    print(f"ROC-AUC      : {test_stats['roc_auc']:.6f}")
    print(f"PR-AUC       : {test_stats['pr_auc']:.6f}")



