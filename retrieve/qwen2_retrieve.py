"""Zero-shot multimodal retrieval with Qwen2-VL-7B-Instruct on M-BEIR.

Evaluates both vanilla mean-pooled embeddings (BASE) and embeddings transformed
by the ReAlign whitening (SHRINKAGE) on a single M-BEIR task. See
``retrieve/qwen2_retrieve.sh`` for the recommended launcher.
"""
import os
import glob
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from accelerate import Accelerator
import accelerate
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from collators.eval_collator import EvalDataCollator
from utils.embeddings import mean_pooling

from dataset.t2i import Text2ImageDataset           # task 1
from dataset.t2t import Text2TextDataset            # task 2
from dataset.t2it import Text2ImageTextDataset      # task 3
from dataset.i2t import Image2TextDataset           # task 4
from dataset.i2i import Image2ImageDataset          # task 5
from dataset.it2t import ImageText2TextDataset      # task 6
from dataset.it2i import ImageText2ImageDataset     # task 7
from dataset.it2it import ImageText2ImageTextDataset  # task 8


TASK_MAP = {
    1: Text2ImageDataset,
    2: Text2TextDataset,
    3: Text2ImageTextDataset,
    4: Image2TextDataset,
    5: Image2ImageDataset,
    6: ImageText2TextDataset,
    7: ImageText2ImageDataset,
    8: ImageText2ImageTextDataset,
}


def get_datasets(task_num, query_df, corpus_df, qrels_df):
    if task_num not in TASK_MAP:
        raise ValueError(f"Unsupported task_num: {task_num}")
    DatasetClass = TASK_MAP[task_num]
    query_dataset = DatasetClass(query_df, corpus_df, qrels_df, type="query")
    cand_dataset = DatasetClass(query_df, corpus_df, qrels_df, type="corpus")
    return query_dataset, cand_dataset


def tensors_to_device(data, device, dtype=torch.float16):
    for key in list(data.keys()):
        if isinstance(data[key], torch.Tensor):
            if key == 'pixel_values':
                data[key] = data[key].to(device).to(dtype)
            else:
                data[key] = data[key].to(device)
    return data


def extract_embeddings(dataloader, model, accelerator, device):
    features, ids = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, disable=not accelerator.is_main_process):
            batch = tensors_to_device(batch, device, model.dtype)
            outputs = model(**batch, inference=True, output_hidden_states=True)
            last_hidden_states = outputs.hidden_states[-1]  # (B, L, D)
            pooled = mean_pooling(last_hidden_states, batch['attention_mask'])  # (B, D)
            pooled = F.normalize(pooled, dim=-1)

            pooled = accelerator.gather_for_metrics(pooled)
            batch_ids = accelerate.utils.gather_object(batch['ids'])[:len(pooled)]

            features.append(pooled)
            ids.extend(batch_ids)

    features = torch.cat(features, dim=0) if len(features) > 0 else torch.empty(0)
    return features, ids


def shrinkage_whiten(X, alpha=0.1, batch_size=51200):
    """ReAlign: ZCA whitening with a shrinkage covariance estimator.

    Computed in float32 for numerical stability and applied in chunks of
    ``batch_size`` rows to keep peak memory bounded for large candidate pools.
    """
    device = X.device
    N, D = X.shape

    mean = torch.zeros((1, D), device=device)
    for i in tqdm(range(0, N, batch_size), desc="Calculating mean"):
        batch = X[i:i + batch_size].to(torch.float32)
        mean += torch.sum(batch, dim=0, keepdim=True)
    mean /= N

    cov = torch.zeros((D, D), device=device)
    for i in tqdm(range(0, N, batch_size), desc="Calculating covariance"):
        batch = X[i:i + batch_size].to(torch.float32)
        batch_centered = batch - mean
        cov += batch_centered.T @ batch_centered
    cov /= (N - 1)

    shrink_cov = (1 - alpha) * cov + alpha * torch.trace(cov) / D * torch.eye(D, device=device)
    U, S, _ = torch.linalg.svd(shrink_cov)
    W = U @ torch.diag(1.0 / torch.sqrt(S + 1e-5)) @ U.T

    X_white = torch.empty_like(X, dtype=torch.float32)
    for i in tqdm(range(0, N, batch_size), desc="Applying whitening"):
        batch = X[i:i + batch_size].to(torch.float32)
        batch_centered = batch - mean
        transformed = batch_centered @ W
        X_white[i:i + batch_size] = F.normalize(transformed, dim=1)

    return X_white


def evaluate_torch_optimized(query_features, cand_features, query_ids, cand_ids,
                             qrels_df, topk=(1, 5, 10), batch_size=512):
    num_queries = query_features.shape[0]
    max_k = max(topk)

    id2index = {cid: idx for idx, cid in enumerate(cand_ids)}
    relevant_indices_list = []
    for qid in query_ids:
        relevant_cids = qrels_df[qrels_df['query-id'] == qid]['corpus-id'].tolist()
        indices = [id2index[cid] for cid in relevant_cids if cid in id2index]
        relevant_indices_list.append(indices)

    all_topk_idx = []
    print(f"Starting retrieval in batches (size={batch_size})...")
    for i in tqdm(range(0, num_queries, batch_size), desc="Retrieving"):
        batch_queries = query_features[i:i + batch_size]
        batch_sims = torch.matmul(batch_queries, cand_features.T)
        _, batch_topk_idx = torch.topk(batch_sims, max_k, dim=1)
        all_topk_idx.append(batch_topk_idx.cpu())
    all_topk_idx = torch.cat(all_topk_idx, dim=0).numpy()

    results = {}
    for k in topk:
        hit = 0
        for q_idx, relevant_indices in enumerate(relevant_indices_list):
            if not relevant_indices:
                continue
            predicted_indices = all_topk_idx[q_idx, :k]
            if set(predicted_indices) & set(relevant_indices):
                hit += 1
        results[f"Recall@{k}"] = hit / num_queries
    return results


def evaluate_retrieve(model, processor, query_df, corpus_df, qrels_df,
                      batch_size, accelerator, device, task_num=1):

    query_dataset, cand_dataset = get_datasets(task_num, query_df, corpus_df, qrels_df)

    query_loader = DataLoader(
        query_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=EvalDataCollator(tokenizer=processor.tokenizer, processor=processor),
    )
    cand_loader = DataLoader(
        cand_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=EvalDataCollator(tokenizer=processor.tokenizer, processor=processor),
    )

    query_loader, cand_loader, model = accelerator.prepare(query_loader, cand_loader, model)

    query_features, query_ids = extract_embeddings(query_loader, model, accelerator, device)
    cand_features, cand_ids = extract_embeddings(cand_loader, model, accelerator, device)

    print(f"Query features: {query_features.size()}, Candidate features: {cand_features.size()}")

    for method in ["shrinkage", "base"]:
        if method == "shrinkage":
            q_proc = shrinkage_whiten(query_features)
            c_proc = shrinkage_whiten(cand_features)
            print("Finished shrinkage whitening (ReAlign)")
        else:
            q_proc, c_proc = query_features, cand_features

        result = evaluate_torch_optimized(q_proc, c_proc, query_ids, cand_ids, qrels_df)
        print(f"{method.upper()}:", result)


def main(args):
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(args.model_path)

    accelerator = Accelerator()
    device = accelerator.device

    print(f"Loading data from: {args.dataset_path}")
    query_df = pd.read_parquet(os.path.join(args.dataset_path, "query-00000-of-00001.parquet"))
    qrels_df = pd.read_parquet(os.path.join(args.dataset_path, "qrels-00000-of-00001.parquet"))

    corpus_files = sorted(glob.glob(os.path.join(args.dataset_path, "corpus-*.parquet")))
    corpus_df = pd.concat([pd.read_parquet(f) for f in corpus_files], ignore_index=True)

    print(f"Loaded Task {args.task_num}: corpus {len(corpus_df)}, queries {len(query_df)}")

    evaluate_retrieve(
        model, processor, query_df, corpus_df, qrels_df, args.batch_size,
        accelerator, device, task_num=args.task_num,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--task_num', type=int, required=True)
    args = parser.parse_args()
    main(args)
