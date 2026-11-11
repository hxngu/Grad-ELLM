# Grad-ELLM

Official implementation for **Faithfulness Evaluation for Decoder-only LLM Attributions with Controlled Retained Information** (Findings of EMNLP 2026).

**Paper:** https://arxiv.org/abs/2601.03089 & https://aclanthology.org/2026.findings-emnlp.1259/

This repository contains:

- **Grad-ELLM**, a gradient-based attribution method for decoder-only LLMs;
- generation of baseline attribution maps with **Inseq**;
- evaluation with **π-Soft-NC / π-Soft-NS** using controlled expected retaining probability.

## Repository structure

```text
Grad-ELLM/
├── grad_ellm.py                 # Grad-ELLM implementation
├── evaluate.py                  # π-Soft-NC / π-Soft-NS evaluator
├── utils.py                     # scaling, distances, and seeding
├── data/
│   ├── dataset.py               # dataset + attribution loader
│   └── README.md                # expected CSV format
├── scripts/
│   ├── generate_grad_ellm.py    # Grad-ELLM + Random heatmaps
│   ├── generate_baselines.py    # Inseq baselines, including ReAGent
│   └── eval_boolq.sh            # example evaluation command
└── examples/
    └── imdb_sample.csv          # format-only sampled-data example
```

## Installation

```bash
conda create -n grad-ellm python=3.10 -y
conda activate grad-ellm
pip install -r requirements.txt
```

Llama models on Hugging Face may require accepting the model license and logging in with `huggingface-cli login`.

## Data

Prepare a sampled CSV for each dataset. The expected columns are:

| Dataset | Columns |
| --- | --- |
| IMDb / SST2 | `text`, `label` |
| BoolQ | `passage`, `question`, `label` |
| TellMeWhy / WikiBio | `text` |

By default the evaluator looks for:

```text
data/<DATASET>/sampled_data.csv
```

You can instead pass an explicit file with `--data_path`.

## 1. Generate Grad-ELLM heatmaps

Example for IMDb with Llama-3.1-8B-Instruct:

```bash
python -m scripts.generate_grad_ellm \
  --dataset IMDb \
  --model llama3.1-8b-instruct \
  --num_layers 16 \
  --normalize \
  --device 0
```

The script also writes a Random attribution map with the same dimensions. By default outputs are stored under:

```text
outputs/<DATASET>/<MODEL>/
```

Use `--model_name_or_path /path/to/local/model` if you prefer a local checkpoint.

## 2. Generate baseline heatmaps

Baselines are generated through Inseq. For example:

```bash
python -m scripts.generate_baselines \
  --method attention \
  --dataset IMDb \
  --model llama3.1-8b-instruct \
  --device 0
```

Replace `attention` with the Inseq attribution method used in the experiment, e.g. `saliency`, `integrated_gradients`, `input_x_gradient`, `deeplift`, `vargrad`, or `reagent` when supported by your installed Inseq version.


## 3. Evaluate π-Soft-NC / π-Soft-NS

Example:

```bash
python evaluate.py \
  --method grad_ellm \
  --batch_size 1 \
  --evaluation_mode per_token \
  --scaling_method alpha \
  --distance_type hellinger \
  --model llama3.1-8b-instruct \
  --dataset IMDb \
  --num_layers 16 \
  --normalize \
  --device 0
```

For a custom attribution file:

```bash
python evaluate.py \
  --method attention \
  --dataset IMDb \
  --heatmap_path /path/to/attention.pth \
  --data_path /path/to/sampled_data.csv \
  --device 0
```

The evaluation uses `batch_size=1` by default and 15 Monte Carlo samples per retaining probability. The classification setting evaluates the first generated answer token; open-generation experiments use the per-sentence mode and stride configured in the command line.

A BoolQ shell example is provided at `scripts/eval_boolq.sh`.

## Notes on reproducibility

- Attribution tensors are stored in the same order as the sampled CSV.
- Sampling seeds are reset deterministically in the generation/evaluation scripts, matching the released experimental protocol.
- Grad-ELLM uses teacher-forced replay over the sampled output sequence to compute token-level gradients.
- `--freeze_attention` keeps clean attention probabilities fixed during perturbation while allowing the value/content stream to change.
- The evaluator currently scans retaining probabilities from `0.05` to `0.95` in steps of `0.05`, matching the released evaluation code. If reporting a restricted AUC interval, apply the same interval used by the corresponding paper experiment.

## Citation

```bibtex
@article{huang2026faithfulness,
  title={Faithfulness Evaluation for Decoder-only LLM Attributions with Controlled Retained Information},
  author={Huang, Xin and Chan, Antoni B.},
  journal={arXiv preprint arXiv:2601.03089},
  year={2026}
}
```

The proceedings citation can be updated after the EMNLP 2026 Findings version is published.
