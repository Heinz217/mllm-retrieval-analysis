import torch
import torch.nn.functional as F

def mean_pooling(logits, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(logits.size()).float()
    sum_embeddings = torch.sum(logits * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask
