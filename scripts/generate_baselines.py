import argparse
import os
import time

import inseq
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import reset_seeds


MODEL_SPECS = {
    "llama3-8b-instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama3.1-8b-instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Mistral-7B-Instruct-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate baseline attribution heatmaps with Inseq.")
    parser.add_argument("--method", default="attention")
    parser.add_argument("--model", default="llama3.1-8b-instruct", choices=list(MODEL_SPECS))
    parser.add_argument("--model_name_or_path", default=None)
    parser.add_argument("--dataset", default="IMDb", choices=["IMDb", "SST2", "BoolQ", "TellMeWhy", "WikiBio"])
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--output_root", default="outputs")
    parser.add_argument("--device", default="0")
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--ig_steps", type=int, default=16)
    parser.add_argument("--ig_internal_batch_size", type=int, default=8)
    parser.add_argument("--timing_csv", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def default_max_new_tokens(dataset):
    return 2 if dataset in {"IMDb", "SST2", "BoolQ"} else 64


def build_attr_kwargs(method, args):
    if method in {"integrated_gradients", "layer_integrated_gradients", "sequential_integrated_gradients"}:
        return {
            "attribution_args": {
                "n_steps": args.ig_steps,
                "internal_batch_size": args.ig_internal_batch_size,
                "return_convergence_delta": False,
            }
        }
    return {}


def build_messages(dataset, row):
    if dataset in {"IMDb", "SST2"}:
        return [
            {
                "role": "system",
                "content": (
                    "You are a helpful sentiment classifier. Please help to do the sentiment "
                    "classification of the given text and respond ONLY with the single word "
                    "Positive or Negative."
                ),
            },
            {"role": "user", "content": f"Text: {row['text']}"},
        ]
    if dataset == "BoolQ":
        return [
            {
                "role": "system",
                "content": (
                    "You are a factual question answering system. Determine if the given passage "
                    "supports the question being true. Respond ONLY with the single word 'true' "
                    "or 'false' in lowercase, without any punctuation or additional text."
                ),
            },
            {
                "role": "user",
                "content": f"Passage: {row['passage']}\n\nQuestion: {row['question']}",
            },
        ]
    return [{"role": "user", "content": row["text"]}]


def main():
    args = parse_args()
    reset_seeds(args.seed)

    model_name_or_path = args.model_name_or_path or MODEL_SPECS[args.model]
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map=f"cuda:{args.device}",
        attn_implementation="eager",
    ).eval()

    if args.method == "reagent":
        inseq_model = inseq.load_model(
            model,
            "reagent",
            keep_top_n=3,
            stopping_condition_top_k=3,
            replacing_ratio=0.25,
            max_probe_steps=32,
            num_probes=2,
        )
    else:
        inseq_model = inseq.load_model(model, args.method, tokenizer=tokenizer)

    max_new_tokens = args.max_new_tokens or default_max_new_tokens(args.dataset)
    generation_args = {
        "max_new_tokens": max_new_tokens,
        "temperature": 0.7,
        "do_sample": True,
        "top_p": 1.0,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }
    attr_kwargs = build_attr_kwargs(args.method, args)

    data_path = args.data_path or os.path.join("data", args.dataset, "sampled_data.csv")
    data = pd.read_csv(data_path)
    attributions = []

    start_time = time.perf_counter()
    for i, row in tqdm(data.iterrows(), total=len(data)):
        reset_seeds(args.seed)
        messages = build_messages(args.dataset, row)
        input_text = tokenizer.apply_chat_template(messages, tokenize=False)
        if args.model.startswith("llama"):
            input_text += "<|start_header_id|>assistant<|end_header_id|>\n"

        out = inseq_model.attribute(
            input_text,
            generation_args=generation_args,
            **attr_kwargs,
        )
        aggregated = out.sequence_attributions[0].aggregate()
        input_len = len(out.sequence_attributions[0].source)
        attribution = aggregated.target_attributions[:input_len].detach().cpu().float()
        attributions.append(attribution)

    elapsed = time.perf_counter() - start_time
    output_dir = os.path.join(args.output_root, args.dataset, args.model)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{args.method}.pth")
    torch.save(attributions, output_path)

    timing_path = args.timing_csv or os.path.join(output_dir, f"{args.method}_timing.csv")
    pd.DataFrame(
        [{
            "dataset": args.dataset,
            "model": args.model,
            "method": args.method,
            "total_time_sec": elapsed,
            "avg_time_per_sample_sec": elapsed / len(data),
        }]
    ).to_csv(timing_path, index=False)

    print(f"Saved {len(attributions)} attribution maps to {output_path}")


if __name__ == "__main__":
    main()
