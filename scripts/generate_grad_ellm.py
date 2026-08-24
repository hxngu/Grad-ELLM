import argparse
import os

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from grad_ellm import AttributionGenerator
from utils import reset_seeds


MODEL_SPECS = {
    "llama3-8b-instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama3.1-8b-instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Mistral-7B-Instruct-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Grad-ELLM attribution heatmaps.")
    parser.add_argument("--dataset", default="IMDb", choices=["IMDb", "SST2", "BoolQ", "TellMeWhy", "WikiBio"])
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--output_root", default="outputs")
    parser.add_argument("--model", default="llama3.1-8b-instruct", choices=list(MODEL_SPECS))
    parser.add_argument("--model_name_or_path", default=None)
    parser.add_argument("--device", default="0")
    parser.add_argument("--num_layers", type=int, default=16)
    parser.add_argument("--normalize", action="store_true", help="normalize q-k similarity")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


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

    data_path = args.data_path or os.path.join("data", args.dataset, "sampled_data.csv")
    data = pd.read_csv(data_path)
    model_name_or_path = args.model_name_or_path or MODEL_SPECS[args.model]

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map=f"cuda:{args.device}",
        attn_implementation="eager",
    ).eval()

    attributor = AttributionGenerator(
        model,
        tokenizer,
        n_layers=args.num_layers,
        normalize=args.normalize,
    )

    grad_ellm_maps = []
    random_maps = []

    for _, row in tqdm(data.iterrows(), total=len(data)):
        # The paper experiments reset the sampling seed for every example.
        reset_seeds(args.seed)
        messages = build_messages(args.dataset, row)
        input_text = tokenizer.apply_chat_template(messages, tokenize=False)
        if args.model.startswith("llama"):
            input_text += "<|start_header_id|>assistant<|end_header_id|>\n"

        emaps, _ = attributor.updated_generate(
            input_text,
            max_generated_length=args.max_new_tokens,
        )
        emaps = emaps.squeeze(0).transpose(0, 1).contiguous()  # [input, output]
        grad_ellm_maps.append(emaps)
        random_maps.append(torch.rand_like(emaps))

    output_dir = os.path.join(args.output_root, args.dataset, args.model)
    os.makedirs(output_dir, exist_ok=True)
    if args.normalize:
        grad_name = f"grad_ellm_{args.num_layers}.pth"
    else:
        grad_name = f"wo_normalized_grad_ellm_{args.num_layers}.pth"

    torch.save(grad_ellm_maps, os.path.join(output_dir, grad_name))
    torch.save(random_maps, os.path.join(output_dir, "random.pth"))
    print(f"Saved {len(grad_ellm_maps)} Grad-ELLM maps to {output_dir}")


if __name__ == "__main__":
    main()
