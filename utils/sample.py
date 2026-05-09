import pandas as pd
import os
import numpy as np

def sample_dataset(query_df, corpus_df, qrels_df, num_samples=100, seed=42):
    np.random.seed(seed)

    sampled_query_ids = np.random.choice(query_df["id"].unique(), size=min(num_samples, len(query_df)), replace=False)

    sampled_qrels_df = qrels_df[qrels_df["query-id"].isin(sampled_query_ids)].copy()

    sampled_query_df = query_df[query_df["id"].isin(sampled_query_ids)].copy()
    sampled_corpus_ids = sampled_qrels_df["corpus-id"].unique()
    sampled_corpus_df = corpus_df[corpus_df["id"].isin(sampled_corpus_ids)].copy()

    return sampled_query_df, sampled_corpus_df, sampled_qrels_df

def sample_dataset_one_to_one(query_df, corpus_df, qrels_df, num_samples=100, seed=42):
    np.random.seed(seed)
    sampled_query_ids = np.random.choice(query_df["id"].unique(), size=min(num_samples, len(query_df)), replace=False)
    sampled_query_df = query_df[query_df["id"].isin(sampled_query_ids)].copy()
    sampled_qrels_df = qrels_df[qrels_df["query-id"].isin(sampled_query_ids)].copy()

    sampled_qrels_df = sampled_qrels_df.groupby("query-id").first().reset_index()
    sampled_qrels_df = sampled_qrels_df.drop_duplicates(subset=["corpus-id"])
    sampled_qrels_df = sampled_qrels_df.drop_duplicates(subset=["query-id", "corpus-id"])
    sampled_query_df = query_df[query_df["id"].isin(sampled_qrels_df["query-id"])].copy()
    sampled_corpus_df = corpus_df[corpus_df["id"].isin(sampled_qrels_df["corpus-id"])].copy()

    return sampled_query_df, sampled_corpus_df, sampled_qrels_df
