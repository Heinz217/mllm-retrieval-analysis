<div align="center">

# Generative Giants, Retrieval Weaklings: Why do Multimodal Large Language Models Fail at Multimodal Retrieval?

[![ACL 2026](https://img.shields.io/badge/ACL-2026-b31b1b?style=flat-square)](#)
[![arXiv](https://img.shields.io/badge/arXiv-2512.19115-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2512.19115)
[![GitHub](https://img.shields.io/badge/GitHub-mllm--retrieval--analysis-1f6feb?style=flat-square&logo=github)](https://github.com/Heinz217/mllm-retrieval-analysis)
[![HuggingFace SAE](https://img.shields.io/badge/%F0%9F%A4%97%20SAE%20Checkpoints-mllm--retrieval--analysis--sae-yellow?style=flat-square)](https://huggingface.co/Heinz217/mllm-retrieval-analysis-sae)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)](LICENSE)

</div>

---

This repository hosts the official code for our paper:

> **Generative Giants, Retrieval Weaklings: Why do Multimodal Large Language Models Fail at Multimodal Retrieval?**
> *Accepted to ACL 2026.*

We use **Sparse Autoencoders (SAEs)** to dissect the representation space of MLLMs and identify three intrinsic factors that prevent them from being effective zero-shot multimodal retrievers. Building on these insights, we propose **ReAlign**, a *training-free* test-time adaptation method that consistently improves zero-shot multimodal retrieval performance across diverse MLLM architectures.

## ✨ Highlights

- **Mechanistic analysis with SAEs.** We train Top-K sparse autoencoders on the last-layer hidden states of four representative MLLMs (Qwen2-VL-7B, Qwen3-VL-8B, PaliGemma2-3B, Chameleon-7B). Four metrics are defined to probe the geometry of MLLM representations: *energy*, *modality score*, *bridge score*, and *retrieval attribution score*.
- **Three failure modes of MLLMs in retrieval.** (i) The representation space is dominated by textual semantics. (ii) MLLMs spend most of their representational budget on bridging modalities, which homogenizes embeddings. (iii) The components that contribute most to similarity computation actually behave as distractors that hurt retrieval.
- **ReAlign: a training-free fix.** A ZCA whitening with a shrinkage covariance estimator that realigns the geometry of MLLM embeddings on-the-fly, with no fine-tuning and no parameter updates. On M-BEIR, it yields up to about +34\% absolute Recall@5 on InfoSeek (Qwen3-VL-8B, Task 6) and is competitive with fine-tuned baselines such as VLM2Vec and GME on several tasks.

## 📁 Repository Layout

```
mllm-retrieval-analysis/
├── collators/          # Per-architecture HF processor wrappers (Qwen-VL, PaliGemma, Chameleon, ...)
├── dataset/            # PyTorch datasets for the 8 M-BEIR query/candidate modality combinations
├── overcomplete/       # Top-K SAE implementation (adapted from KempnerInstitute/overcomplete)
├── retrieve/           # ReAlign retrieval evaluators + launchers, one per MLLM
├── sae_train/          # SAE training scripts + launchers, one per MLLM
├── utils/              # Shared utilities (mean pooling, dataset sampling)
├── requirements.txt
└── README.md
```

> Pre-trained SAE checkpoints are hosted on Hugging Face at
> [`Heinz217/mllm-retrieval-analysis-sae`](https://huggingface.co/Heinz217/mllm-retrieval-analysis-sae). They are not required for running ReAlign retrieval — they are only needed to reproduce the SAE-based analysis in Section 3 of the paper.

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Heinz217/mllm-retrieval-analysis.git
cd mllm-retrieval-analysis
```

### 2. Set up the environment

```bash
conda create -n mllm-retrieve python=3.10 -y
conda activate mllm-retrieve

pip install -r requirements.txt
pip install ninja
```

### 3. Download MLLM weights

ReAlign and the SAE training pipeline both rely on the original Hugging Face checkpoints of the analyzed MLLMs. Download whichever ones you plan to use:

| Model                       | Hugging Face repo                                               | Hidden dim |
|-----------------------------|-----------------------------------------------------------------|-----------:|
| Qwen2-VL-7B-Instruct        | [`Qwen/Qwen2-VL-7B-Instruct`](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)         |       3584 |
| Qwen3-VL-8B-Instruct        | [`Qwen/Qwen3-VL-8B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)         |       4096 |
| PaliGemma2-3B-Mix-224       | [`google/paligemma2-3b-mix-224`](https://huggingface.co/google/paligemma2-3b-mix-224)   |       2304 |
| Chameleon-7B                | [`facebook/chameleon-7b`](https://huggingface.co/facebook/chameleon-7b)                 |       4096 |

For example:

```bash
huggingface-cli download Qwen/Qwen2-VL-7B-Instruct --local-dir ./checkpoints/Qwen2-VL-7B-Instruct
```

## 🔬 The SAE-based Analysis

### Data preparation: COCO

We train all SAEs on activations collected from the
[`lmms-lab/COCO-Caption`](https://huggingface.co/datasets/lmms-lab/COCO-Caption)
dataset. Download it and lay it out as:

```bash
huggingface-cli download lmms-lab/COCO-Caption \
    --repo-type dataset \
    --local-dir ./coco-caption
```

```
coco-caption/
├── val-00000-of-00013.parquet
├── val-00001-of-00013.parquet
├── ...
└── val-00012-of-00013.parquet
```

### Train SAEs from scratch

Each MLLM has its own launcher under `sae_train/` (`qwen2_sae_train.sh`,
`qwen3_sae_train.sh`, `paligemma_sae_train.sh`, `chameleon_sae_train.sh`).
Edit the two paths at the top (`MODEL_PATH`, `DATA_DIR`) and run, for example:

```bash
bash sae_train/qwen3_sae_train.sh
```

Each run streams ~28B activations through the model on-the-fly (no pre-cached
activation tensors) and saves checkpoints under `./sae_checkpoints/` by
default. Training was done on 4× NVIDIA H20 GPUs in our experiments.

### Use our released SAE checkpoints

If you only want to reproduce the SAE-based analysis without retraining, pull
the pre-trained checkpoints from
[`Heinz217/mllm-retrieval-analysis-sae`](https://huggingface.co/Heinz217/mllm-retrieval-analysis-sae)
on Hugging Face:

```bash
huggingface-cli download Heinz217/mllm-retrieval-analysis-sae \
    --local-dir ./sae_checkpoints
```

The repository ships one `.pt` file per backbone (also covering CLIP and
SigLIP2 in addition to the four MLLMs):

| Filename                          | Backbone                       |
|-----------------------------------|--------------------------------|
| `sae_Qwen2-VL-7B-Instruct.pt`     | Qwen2-VL-7B-Instruct           |
| `sae_Qwen3-VL-8B-Instruct.pt`     | Qwen3-VL-8B-Instruct           |
| `sae_paligemma-3b-mix-224.pt`     | PaliGemma2-3B-Mix-224          |
| `sae_chameleon-7b.pt`             | Chameleon-7B                   |
| `sae_clip.pt`                     | CLIP (ViT-Base-Patch32)        |
| `sae_siglip2.pt`                  | SigLIP2 (Base-Patch16-512)     |

A machine-readable index is provided as `model_index.json` in the same
repository.

Each checkpoint stores `model_state` (and optionally `optimizer_state` /
training `logs`) and can be loaded into a `TopKSAE` instance:

```python
import torch
from overcomplete.sae.topk_sae import TopKSAE

ckpt = torch.load("./sae_checkpoints/sae_Qwen2-VL-7B-Instruct.pt", map_location="cuda")
sae = TopKSAE(input_shape=..., nb_concepts=..., top_k=50, device="cuda").to("cuda")
sae.load_state_dict(ckpt["model_state"])
```

## 🧪 Zero-shot Multimodal Retrieval (ReAlign)

### Data preparation: M-BEIR

Download the test set of the M-BEIR benchmark from the Hugging Face mirror at
[`MRBench`](https://huggingface.co/MRBench). Each task is stored in its own
directory; organize the data on disk as below:

```
M-BEIR/
├── mbeir_mscoco_task4/
│   ├── query-00000-of-00001.parquet
│   ├── corpus-00000-of-00001.parquet
│   └── qrels-00000-of-00001.parquet
├── mbeir_cirr_task7/
│   ├── ...
├── mbeir_infoseek_task6/
│   ├── ...
└── ...
```

The benchmark covers 8 retrieval tasks, indexed by `--task_num`:

| `--task_num` | Query → Candidate           | Example datasets                  |
|:------------:|:----------------------------|:----------------------------------|
| 1            | text → image                | VisualNews, MSCOCO, Fashion200K   |
| 2            | text → text                 | WebQA                             |
| 3            | text → (image, text)        | EDIS, WebQA                       |
| 4            | image → text                | VisualNews, MSCOCO, Fashion200K   |
| 5            | image → image               | NIGHTS                            |
| 6            | (image, text) → text        | OVEN, InfoSeek                    |
| 7            | (image, text) → image       | FashionIQ, CIRR                   |
| 8            | (image, text) → (image, text) | OVEN, InfoSeek                  |

### Run ReAlign evaluation

Each MLLM has its own launcher under `retrieve/` (`qwen2_retrieve.sh`,
`qwen3_retrieve.sh`, `paligemma_retrieve.sh`, `chameleon_retrieve.sh`). Open
the launcher, set the two paths at the top, and run, for example:

```bash
bash retrieve/qwen3_retrieve.sh
```

The script prints both the vanilla MLLM (`BASE`) and the ReAlign-whitened
(`SHRINKAGE`) Recall@{1, 5, 10} for the chosen task. To switch tasks, edit the
`TASK` variable inside the launcher (1–8, see the table above) and update
`DATASET_DIR` to point to the corresponding `mbeir_*_task*/` directory.

## 📜 License

This codebase is released under the [MIT License](LICENSE).

## 🙏 Acknowledgments

This repository builds on the excellent work of [LamRA](https://github.com/Code-kunkun/LamRA) for the multimodal retrieval pipeline with MLLMs, [overcomplete](https://github.com/KempnerInstitute/overcomplete) for the Top-K sparse autoencoder framework, and the [M-BEIR](https://huggingface.co/MRBench) benchmark for evaluating multimodal retrieval. We are grateful to the authors and maintainers of these projects.

## 📑 Citation

If you find this work useful, please cite our paper:

```bibtex
@misc{feng2025generativegiantsretrievalweaklings,
      title={Generative Giants, Retrieval Weaklings: Why do Multimodal Large Language Models Fail at Multimodal Retrieval?}, 
      author={Hengyi Feng and Zeang Sheng and Meiyi Qiang and Wentao Zhang},
      year={2025},
      eprint={2512.19115},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.19115}, 
}
```

<!-- @misc{feng2025generativegiantsretrievalweaklings,
      title={Generative Giants, Retrieval Weaklings: Why do Multimodal Large Language Models Fail at Multimodal Retrieval?}, 
      author  = {Feng, Hengyi and Sheng, Zeang and Qiang, Meiyi and Li, Yang and Zhang, Wentao},
      year={2025},
      eprint={2512.19115},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.19115}, 
} -->
