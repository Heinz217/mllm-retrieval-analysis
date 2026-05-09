import json
from typing import Dict, List
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import io
import tqdm
from torch.utils.data import Dataset

class ImageText2ImageTextDataset(Dataset):
    def __init__(
        self,
        query_df: pd.DataFrame,
        corpus_df: pd.DataFrame,
        qrels_df: pd.DataFrame,
        type: str
    ) -> None:
        self.query_df = query_df
        self.corpus_df = corpus_df
        self.qrels_df = qrels_df
        self.type = type

    def construct_messages(self, type: str, text: str = None, image: Image.Image = None):
        if type == "query":
            message = {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text},
                    {"type": "text", "text": f"\nSummarize the above image and the sentence in one word: "}
                ]
            }
        elif type == "corpus":
            message = {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text},
                    {"type": "text", "text": f"\nSummarize the above image and the sentence in one word: "}
                ]
            }
        else:
            raise ValueError("Only 'query' and 'corpus' types are supported.")
        return message

    def __len__(self) -> int:
        if self.type == "query":
            return len(self.query_df)
        elif self.type == "corpus":
            return len(self.corpus_df)
        elif self.type == "clip-query":
            return len(self.query_df)
        elif self.type == "clip-corpus":
            return len(self.corpus_df)
        else:
            raise ValueError(f"Unknown data: {self.type}")

    def __getitem__(self, idx: int):
        if self.type == "query":
            row = self.query_df.iloc[idx]
            query_id = row['id']
            query_text = row['text']
            bytes_image = row['image']  # {'bytes': ...}
            image = Image.open(io.BytesIO(bytes_image['bytes'])).convert('RGB')
            # image = Image.open(io.BytesIO(bytes_image['bytes']))

            messages = self.construct_messages(type="query", text=query_text, image=image)
            relevant_corpus_ids = self.qrels_df[self.qrels_df['query-id'] == query_id]['corpus-id'].tolist()
            return messages, query_id

        elif self.type == "corpus":
            row = self.corpus_df.iloc[idx]
            corpus_id = row['id']
            corpus_text = row['text']
            bytes_image = row['image']  # {'bytes': ...}
            image = Image.open(io.BytesIO(bytes_image['bytes'])).convert('RGB')
            # image = Image.open(io.BytesIO(bytes_image['bytes']))

            messages = self.construct_messages(type="corpus", text=corpus_text, image=image)
            return messages, corpus_id

        elif self.type == "clip-query":
            row = self.query_df.iloc[idx]
            query_id = row['id']
            query_text = row['text']
            bytes_image = row['image']  # {'bytes': ...}
            image = Image.open(io.BytesIO(bytes_image['bytes'])).convert('RGB')
            # image = Image.open(io.BytesIO(bytes_image['bytes']))

            messages = self.construct_messages(type="query", text=query_text, image=image)
            relevant_corpus_ids = self.qrels_df[self.qrels_df['query-id'] == query_id]['corpus-id'].tolist()

            return {
                "image": image,
                "text": query_text,
                "query_id": query_id,
                "corpus_id": None,
            }

        elif self.type == "clip-corpus":
            row = self.corpus_df.iloc[idx]
            corpus_id = row['id']
            corpus_text = row['text']
            bytes_image = row['image']  # {'bytes': ...}
            image = Image.open(io.BytesIO(bytes_image['bytes'])).convert('RGB')
            # image = Image.open(io.BytesIO(bytes_image['bytes']))

            messages = self.construct_messages(type="corpus", text=corpus_text, image=image)

            return {
                "image": image,
                "text": corpus_text,
                "query_id": None,
                "corpus_id": corpus_id,
            }
