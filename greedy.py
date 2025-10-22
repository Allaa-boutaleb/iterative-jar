import argparse
import pickle
import time
from typing import List, Sequence, Dict, Tuple

import numpy as np
from tqdm import tqdm

from utils import read_json, write_json, get_corpus, get_skip_idxs
from compatibility import get_cr
from metrics import eval_preds

def load_fine_scores(dataset: str):
    with open(f"./data/{dataset}/contriever/score_decomp.pkl", "rb") as fh:
        flat = pickle.load(fh)
    decomp = read_json(f"./data/{dataset}/decomp.json")
    cum = [0]
    for subs in decomp:
        cum.append(cum[-1] + len(subs))
    corpus = get_corpus(dataset)
    mats: List[np.ndarray] = []
    for qi in range(len(decomp)):
        st, ed = cum[qi], cum[qi + 1]
        block = flat[st:ed]
        n_subq = ed - st
        mat = np.zeros((n_subq, len(corpus)), dtype=np.float32)
        for sub_idx, per_table_arrays in enumerate(block):
            for table_idx, arr in enumerate(per_table_arrays):
                if isinstance(arr, np.ndarray):
                    value = float(arr.max())
                elif isinstance(arr, list):
                    value = max(x.max() for x in arr if isinstance(x, np.ndarray))
                else:
                    value = 0.0
                mat[sub_idx, table_idx] = value
        mats.append(mat)
    return mats

def _coverage_gain(F: np.ndarray, best_cov: np.ndarray, idx: int) -> float:
    return float(np.clip(F[:, idx] - best_cov, 0, None).sum())

def _join_gain(cr: Dict[Tuple[int, int], np.ndarray], selected: Sequence[int], idx: int) -> float:
    if not selected:
        return 0.0
    gain = 0.0
    for j in selected:
        mat = cr[(idx, j)] if (idx, j) in cr else cr[(j, idx)]
        gain += float(mat.max())
    return gain

def greedy_selection(
    F: np.ndarray,
    coarse: np.ndarray,
    cr: Dict[Tuple[int, int], np.ndarray],
    K: int,
    lambda_cov: float,
    lambda_join: float,
    lambda_coarse: float,
) -> List[int]:
    M = F.shape[1]
    avail = set(range(M))
    selected: List[int] = []
    best_cov = np.zeros(F.shape[0], dtype=np.float32)

    # initial seed
    best_idx, best_gain = None, -np.inf
    for idx in avail:
        gain = lambda_cov * _coverage_gain(F, best_cov, idx) + lambda_coarse * coarse[idx]
        if gain > best_gain:
            best_gain, best_idx = gain, idx
    selected.append(best_idx)
    avail.remove(best_idx)
    best_cov = np.maximum(best_cov, F[:, best_idx])

    # greedy add
    while len(selected) < K and avail:
        gains = []
        for idx in avail:
            cov_g = _coverage_gain(F, best_cov, idx)
            join_g = _join_gain(cr, selected, idx)
            total_gain = lambda_cov * cov_g + lambda_join * join_g + lambda_coarse * coarse[idx]
            gains.append((total_gain, idx, cov_g))
        gains.sort(reverse=True)
        best_gain, best_idx, cov_g = gains[0]
        if cov_g <= 0 and len(selected) >= K:
            break
        selected.append(best_idx)
        avail.remove(best_idx)
        best_cov = np.maximum(best_cov, F[:, best_idx])

    return selected

def format_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"Total execution time: {minutes} minutes {secs} seconds"

def main():
    parser = argparse.ArgumentParser(description="Greedy table retrieval")
    parser.add_argument("--dataset", required=True, choices=["bird", "spider", "beaver_dw", "beaver_nw"])
    parser.add_argument("--model", default="contriever", choices=["contriever", "tapas"])
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--K", type=int, default=2)
    parser.add_argument("--lambda_cov", type=float, default=2.0)
    parser.add_argument("--lambda_join", type=float, default=1.0)
    parser.add_argument("--lambda_coarse", type=float, default=3.0)
    args = parser.parse_args()

    start_time = time.time()
    print(f"Starting Greedy Selection | {time.strftime('%Y-%m-%d %H:%M:%S')}")

    skip = set(get_skip_idxs(args.dataset))
    preds_topk = read_json(f"./data/{args.dataset}/{args.model}/preds_{args.topk}.json")
    fine_mats = load_fine_scores(args.dataset)
    coarse_all = np.load(f"./data/{args.dataset}/{args.model}/score.npy")
    corpus = get_corpus(args.dataset)

    load_time = time.time() - start_time
    print(f"Data loaded in {load_time:.2f}s")

    all_preds: List[List[str]] = []
    query_times = []
    proc_start = time.time()

    for qi, top_tables in enumerate(tqdm(preds_topk, desc="Queries")):
        q_start = time.time()
        if qi in skip:
            all_preds.append([])
            continue

        idxs = [corpus.index(t) for t in top_tables]
        Fsub = fine_mats[qi][:, idxs]
        coarse_sub = coarse_all[qi][idxs]
        cr_sub = get_cr(args.dataset, top_tables)

        sel_idxs = greedy_selection(
            Fsub, coarse_sub, cr_sub,
            K=args.K,
            lambda_cov=args.lambda_cov,
            lambda_join=args.lambda_join,
            lambda_coarse=args.lambda_coarse,
        )
        all_preds.append([top_tables[i] for i in sel_idxs])
        query_times.append(time.time() - q_start)

    process_time = time.time() - proc_start

    out_fn = f"./data/{args.dataset}/greedy_k{args.K}.json"
    write_json(all_preds, out_fn)
    print(f"Saved → {out_fn}")

    total_time = time.time() - start_time
    print(f"\nTIMING:")
    print(f"  Load time:      {load_time:.2f}s")
    print(f"  Process time:   {process_time:.2f}s")
    print(f"    Avg/query:    {np.mean(query_times):.4f}s")
    print(format_time(total_time))  # Modified line

    print("\nEvaluating:")
    eval_preds(args.dataset, all_preds)

if __name__ == "__main__":
    main()
