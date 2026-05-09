"""Train a Top-K sparse autoencoder on the last-layer hidden states of
Qwen3-VL-8B-Instruct, using activations collected on-the-fly from the
COCO-Caption dataset. See ``sae_train/qwen3_sae_train.sh`` for the launcher.
"""
import os
import argparse
import glob
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from collators.eval_collator import EvalDataCollator
from dataset.i2t import Image2TextDataset

from overcomplete.sae.topk_sae import TopKSAE
from overcomplete.sae.trackers import DeadCodeTracker
from overcomplete.metrics import r2_score


def split_multimodal_dataframe(df: pd.DataFrame):
    """Split a COCO-Caption parquet into (query, candidate, qrels) frames."""
    query_rows, cand_rows, qrels_rows = [], [], []
    query_counter = 0
    cand_global_counter = 0

    for _, row in df.iterrows():
        query_id = f"{query_counter}:1"
        query_rows.append({
            "id": query_id, "modality": "image",
            "text": None, "image": row["image"],
        })

        answers = row["answer"]
        if isinstance(answers, np.ndarray):
            answers = answers.tolist()
        elif not isinstance(answers, (list, tuple)):
            answers = [answers]

        for ans in answers:
            if ans is None or str(ans).strip() == "":
                continue
            cand_id = f"{query_counter}:{cand_global_counter}"
            cand_rows.append({
                "id": cand_id, "modality": "text",
                "text": str(ans).strip(), "image": None,
            })
            qrels_rows.append({
                "query-id": query_id, "Q0": 0,
                "corpus-id": cand_id, "score": 1,
            })
            cand_global_counter += 1
        query_counter += 1

    return (pd.DataFrame(query_rows),
            pd.DataFrame(cand_rows),
            pd.DataFrame(qrels_rows))


def collect_hidden_states(model, dataloader, device, mask_ids):
    hidden_states_pool = []

    for batch in tqdm(dataloader, desc="Collecting hidden states", position=1, leave=False):
        batch = tensors_to_device(batch, device, model.dtype)

        with torch.no_grad():
            hidden = model(**batch, output_hidden_states=True).hidden_states[-1]

        data_attention_mask = batch['attention_mask'].bool()
        data_mask = torch.zeros_like(data_attention_mask)

        system_ids = mask_ids[0]
        data_mask[:, :len(system_ids)] = 1
        data_mask = data_mask.bool()

        filtered_hidden = hidden[~data_mask & data_attention_mask]
        filtered_hidden = filtered_hidden.view(1, -1, hidden.shape[-1])

        if filtered_hidden.numel() > 0:
            hidden_states_pool.append(filtered_hidden.cpu())

    return hidden_states_pool


def tensors_to_device(data, device, dtype=torch.float16):
    for key in data.keys():
        if isinstance(data[key], torch.Tensor):
            if key == 'pixel_values':
                data[key] = data[key].to(device).to(dtype)
            else:
                data[key] = data[key].to(device)
    return data


def process_one_parquet_file(parquet_file, model, processor, tokenizer, device,
                             system_prompt_ids, user_query_ids, user_cand_ids,
                             llm_batch_size, seed):
    print(f"\n  Loading {os.path.basename(parquet_file)}...")

    df = pd.read_parquet(parquet_file)
    query_df, cand_df, qrels_df = split_multimodal_dataframe(df)

    query_dataset = Image2TextDataset(query_df, cand_df, qrels_df, type="query")
    cand_dataset = Image2TextDataset(query_df, cand_df, qrels_df, type="corpus")

    query_collator = EvalDataCollator(tokenizer=tokenizer, processor=processor)
    cand_collator = EvalDataCollator(tokenizer=tokenizer, processor=processor)

    query_loader = DataLoader(query_dataset, batch_size=llm_batch_size,
                              num_workers=1, shuffle=False, collate_fn=query_collator)
    cand_loader = DataLoader(cand_dataset, batch_size=llm_batch_size,
                             num_workers=1, shuffle=False, collate_fn=cand_collator)

    query_hidden = collect_hidden_states(
        model, query_loader, device, (system_prompt_ids, user_query_ids))
    cand_hidden = collect_hidden_states(
        model, cand_loader, device, (system_prompt_ids, user_cand_ids))

    local_pool = query_hidden + cand_hidden
    if len(local_pool) == 0:
        return None

    local_pool = [h.view(-1, h.shape[-1]) for h in local_pool]
    all_hidden = torch.cat(local_pool, dim=0)
    indices = torch.randperm(all_hidden.shape[0])
    return all_hidden[indices]


def train_on_pool(sae, optimizer, criterion, pool_hidden, sae_batch_size, device,
                  dead_tracker, pbar_position=1):
    num_batches = (pool_hidden.shape[0] + sae_batch_size - 1) // sae_batch_size

    pool_loss = 0.0
    pool_error = 0.0
    pool_sparsity = 0.0
    batch_count = 0

    pbar = tqdm(range(num_batches), desc="    Training on pool",
                position=pbar_position, leave=False)

    for batch_idx in pbar:
        start = batch_idx * sae_batch_size
        end = min(start + sae_batch_size, pool_hidden.shape[0])
        batch_hidden = pool_hidden[start:end].to(device).to(
            sae.parameters().__next__().dtype)

        optimizer.zero_grad()
        pre_codes, codes = sae.encode(batch_hidden)
        x_hat = sae.decode(codes)
        loss = criterion(batch_hidden, x_hat, pre_codes, codes, sae.get_dictionary())

        dead_tracker.update(codes)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        optimizer.step()

        batch_loss = loss.item()
        batch_error = r2_score(batch_hidden, x_hat).item()
        non_zero = (codes.abs() > 1e-6).sum(dim=-1)
        batch_sparsity = non_zero.float().mean().item()

        pool_loss += batch_loss
        pool_error += batch_error
        pool_sparsity += batch_sparsity
        batch_count += 1

        pbar.set_postfix({
            'loss': f'{batch_loss:.4f}',
            'r2': f'{batch_error:.4f}',
            'l0': f'{batch_sparsity:.2f}',
        })

    pbar.close()

    if batch_count > 0:
        return (pool_loss / batch_count,
                pool_error / batch_count,
                pool_sparsity / batch_count,
                batch_count)
    return 0.0, 0.0, 0.0, 0


def train(args):
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True,
    )
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoProcessor.from_pretrained(args.model_id)
    tokenizer = processor.tokenizer

    parquet_pattern = os.path.join(args.data_dir, "val-*.parquet")
    parquet_files = sorted(glob.glob(parquet_pattern))
    if not parquet_files:
        raise ValueError(f"No parquet files found matching {parquet_pattern}")

    accelerator = Accelerator()
    device = accelerator.device
    model.eval()

    sae = TopKSAE(input_shape=args.hidden_dim, nb_concepts=args.d_sae,
                  top_k=50, device=device).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=8e-4,
                                 betas=(0.9, 0.999), eps=1e-8)

    SPARSITY_COEFF = 1e-2
    def criterion(x, x_hat, pre_codes, codes, dictionary):
        reconstruction_loss = (x - x_hat).square().mean()
        sparsity_loss = codes.abs().mean()
        return reconstruction_loss + SPARSITY_COEFF * sparsity_loss

    logs = defaultdict(list)
    system_prompt_ids = None
    user_query_ids = []
    user_cand_ids = []

    for epoch in range(args.n_epochs):
        print(f"\n{'='*80}\nEpoch {epoch+1}/{args.n_epochs}\n{'='*80}")

        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_error = 0.0
        epoch_sparsity = 0.0
        epoch_batch_count = 0
        dead_tracker = DeadCodeTracker(args.d_sae, device)

        file_pbar = tqdm(parquet_files, desc="Processing parquet files",
                         position=0, leave=False)

        for file_idx, parquet_file in enumerate(file_pbar):
            file_pbar.set_description(f"File {file_idx+1}/{len(parquet_files)}")

            pool_hidden = process_one_parquet_file(
                parquet_file, model, processor, tokenizer, device,
                system_prompt_ids, user_query_ids, user_cand_ids,
                args.llm_batch_size, seed=42 + epoch + file_idx,
            )

            if pool_hidden is None or pool_hidden.shape[0] == 0:
                print("    Warning: Empty pool, skipping...")
                continue

            pool_loss, pool_error, pool_sparsity, pool_batch_count = train_on_pool(
                sae, optimizer, criterion, pool_hidden, args.sae_batch_size,
                device, dead_tracker, pbar_position=1,
            )

            epoch_loss += pool_loss * pool_batch_count
            epoch_error += pool_error * pool_batch_count
            epoch_sparsity += pool_sparsity * pool_batch_count
            epoch_batch_count += pool_batch_count

            dead_ratio = dead_tracker.get_dead_ratio()
            print(f"    Pool trained: Loss={pool_loss:.4f}, R2={pool_error:.4f}, "
                  f"L0={pool_sparsity:.2f}, Dead={dead_ratio*100:.1f}%")

            if (file_idx + 1) % 5 == 0 and epoch_batch_count > 0:
                cum_loss = epoch_loss / epoch_batch_count
                cum_err = epoch_error / epoch_batch_count
                cum_spar = epoch_sparsity / epoch_batch_count
                print(f"\n  [Cumulative after {file_idx+1} files]"
                      f" Loss={cum_loss:.4f}, R2={cum_err:.4f}, "
                      f"L0={cum_spar:.2f}, Dead={dead_ratio*100:.1f}%\n")

        file_pbar.close()

        if epoch_batch_count > 0:
            avg_loss = epoch_loss / epoch_batch_count
            avg_error = epoch_error / epoch_batch_count
            avg_sparsity = epoch_sparsity / epoch_batch_count
            dead_ratio = dead_tracker.get_dead_ratio()
            duration = time.time() - epoch_start

            logs['avg_loss'].append(avg_loss)
            logs['r2'].append(avg_error)
            logs['time_epoch'].append(duration)
            logs['z_sparsity'].append(avg_sparsity)
            logs['dead_features'].append(dead_ratio)

            print(f"\n{'='*80}\nEpoch {epoch+1} Summary:")
            print(f"  Loss: {avg_loss:.4f}")
            print(f"  R2 Score: {avg_error:.4f}")
            print(f"  L0 Sparsity: {avg_sparsity:.2f}")
            print(f"  Dead Features: {dead_ratio*100:.1f}%")
            print(f"  Total Batches: {epoch_batch_count}")
            print(f"  Time: {duration:.2f}s ({duration/60:.2f}min)")
            print(f"{'='*80}")

        os.makedirs(args.save_dir, exist_ok=True)
        ckpt_path = os.path.join(args.save_dir, f"sae_qwen3vl_coco_epoch{epoch+1}.pt")
        torch.save({
            "epoch": epoch + 1,
            "model_state": sae.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "logs": logs,
        }, ckpt_path)
        print(f"Checkpoint saved at {ckpt_path}")

    final_ckpt = os.path.join(args.save_dir, "sae_qwen3vl_coco_final.pt")
    torch.save({
        "epoch": args.n_epochs,
        "model_state": sae.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "logs": logs,
    }, final_ckpt)
    print(f"\nFinal SAE model saved at {final_ckpt}")

    return logs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./sae_checkpoints")
    parser.add_argument("--llm_batch_size", type=int, default=1)
    parser.add_argument("--sae_batch_size", type=int, default=4096)
    parser.add_argument("--hidden_dim", type=int, default=4096)
    parser.add_argument("--d_sae", type=int, default=32768)
    parser.add_argument("--n_epochs", type=int, default=2)
    args = parser.parse_args()

    train(args)
