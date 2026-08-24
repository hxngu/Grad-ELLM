import argparse
import os
import warnings

import torch.nn.functional as F
import numpy as np
import torch
from sklearn.metrics import auc
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from contextlib import contextmanager
from grad_ellm import AttributionGenerator
from data.dataset import CustomizedDataset
from utils import (
    l1_scaling,
    alpha_scaling,
    cosine_distance,
    default_scaling,
    hellinger_distance,
    reset_seeds,
    softmax_scaling,
)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from torch.utils.tensorboard import SummaryWriter


MODEL_SPECS = {
    "llama3-8b-instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama3.1-8b-instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Mistral-7B-Instruct-v0.3": "mistralai/Mistral-7B-Instruct-v0.3",
}


PREFIX_MAP = {
    ("llama3-8b-instruct", "IMDb"): 41,
    ("llama3-8b-instruct", "SST2"): 41,
    ("llama3.1-8b-instruct", "IMDb"): 61,
    ("llama3.1-8b-instruct", "SST2"): 61,
    ("llama3.1-8b-instruct", "BoolQ"): 75,
    ("llama3.1-8b-instruct", "TellMeWhy"): 30,
    ("llama3.1-8b-instruct", "WikiBio"): 30,
    ("Mistral-7B-Instruct-v0.3", "IMDb"): 39,
    ("Mistral-7B-Instruct-v0.3", "SST2"): 39,
    ("Mistral-7B-Instruct-v0.3", "BoolQ"): 55,
    ("Mistral-7B-Instruct-v0.3", "TellMeWhy"): 2,
    ("Mistral-7B-Instruct-v0.3", "WikiBio"): 2,
}

SUPPORTED_CLASSIFICATION_DATASETS = {"IMDb", "SST2", "BoolQ"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate attribution faithfulness with pi-Soft-NC/NS.")
    parser.add_argument("--method", type=str, default="grad_ellm", help="attribution method name")
    parser.add_argument("--log_dir", type=str, default="results", help="TensorBoard output directory")
    parser.add_argument("--data_path", type=str, default=None, help="path to sampled_data.csv")
    parser.add_argument("--heatmap_path", type=str, default=None, help="path to precomputed attribution .pth")
    parser.add_argument("--output_root", type=str, default="outputs", help="default root for attribution files")
    parser.add_argument("--model_name_or_path", type=str, default=None, help="HF model id or local model path")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--model", type=str, default="llama3.1-8b-instruct")
    parser.add_argument("--num_layers", type=int, default=16)
    parser.add_argument("--normalize", action="store_true", help="use normalized q-k similarity in Grad-ELLM")
    parser.add_argument("--dataset", type=str, default="IMDb", choices=["IMDb", "SST2", "BoolQ", "TellMeWhy", "WikiBio"])
    parser.add_argument("--evaluation", type=str, default="soft_ns_nc")
    parser.add_argument("--evaluation_mode", type=str, default="per_token", choices=["per_token", "per_sentence"])
    parser.add_argument("--stride", type=int, default=5, help="stride for per-sentence evaluation")
    parser.add_argument("--scaling_method", type=str, default="alpha", choices=["alpha", "softmax", "L1", "default"])
    parser.add_argument("--distance_type", type=str, default="hellinger", choices=["hellinger", "cosine"])
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mc_samples", type=int, default=15)
    parser.add_argument("--num_samples", type=int, default=None, help="evaluate the last N samples; paper defaults are used when omitted")
    parser.add_argument(
        "--freeze_attention",
        action="store_true",
        help="freeze attention probabilities during perturbation evaluation",
    )
    return parser.parse_args()


def build_log_dir(args: argparse.Namespace) -> str:
    log_dir = os.path.join(
        args.log_dir,
        args.model,
        args.dataset,
        args.method,
        args.evaluation,
        args.scaling_method,
        args.distance_type,
    )
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def create_writer(args: argparse.Namespace, log_dir: str):
    if args.debug:
        return None

    print("saving the result.")
    if args.method == "grad_ellm" and args.normalize:
        return SummaryWriter(os.path.join(log_dir, f"normalized_{args.num_layers}"))
    if args.method == "grad_ellm":
        return SummaryWriter(os.path.join(log_dir, f"wo_normalized_{args.num_layers}"))
    return SummaryWriter(log_dir)


def build_dataset_paths(args: argparse.Namespace) -> tuple[str, str]:
    dataset_path = args.data_path or os.path.join("data", args.dataset, "sampled_data.csv")

    if args.method == "grad_ellm" and args.normalize:
        heatmap_name = f"grad_ellm_{args.num_layers}.pth"
    elif args.method == "grad_ellm":
        heatmap_name = f"wo_normalized_grad_ellm_{args.num_layers}.pth"
    else:
        heatmap_name = f"{args.method}.pth"

    heatmap_path = args.heatmap_path or os.path.join(
        args.output_root, args.dataset, args.model, heatmap_name
    )
    return dataset_path, heatmap_path


def resolve_model(args: argparse.Namespace):
    if args.model not in MODEL_SPECS:
        supported = ", ".join(MODEL_SPECS.keys())
        raise ValueError(f"Unsupported model '{args.model}'. Supported: {supported}")

    model_name_or_path = args.model_name_or_path or MODEL_SPECS[args.model]
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map=f"cuda:{args.device}",
        attn_implementation="eager",
    ).eval()
    return tokenizer, model


def unwrap_alpha_scaling_result(result: torch.Tensor | tuple) -> torch.Tensor:
    if isinstance(result, tuple):
        return result[0]
    return result


def get_scaling_function(name: str):
    scaling_map = {
        "alpha": lambda x, p: unwrap_alpha_scaling_result(alpha_scaling(x, p)),
        "softmax": lambda x, p: softmax_scaling(x),
        "L1": lambda x, p: l1_scaling(x),
        "default": lambda x, p: default_scaling(x),
    }

    if name == "quantile":
        raise ValueError("scaling_method='quantile' is configured but quantile_scaling is not available in util.py")

    if name not in scaling_map:
        supported = ", ".join(scaling_map.keys())
        raise ValueError(f"Unsupported scaling method '{name}'. Supported: {supported}")

    return scaling_map[name]


def get_distance_function(name: str):
    distance_map = {
        "hellinger": hellinger_distance,
        "cosine": cosine_distance,
    }
    if name not in distance_map:
        supported = ", ".join(distance_map.keys())
        raise ValueError(f"Unsupported distance type '{name}'. Supported: {supported}")
    return distance_map[name]


def get_prefix(model_name: str, dataset: str) -> int:
    key = (model_name, dataset)
    if key not in PREFIX_MAP:
        raise ValueError(f"No prefix configured for model={model_name}, dataset={dataset}")
    return PREFIX_MAP[key]

def build_position_ids(attn_mask: torch.Tensor) -> torch.Tensor:
    """
    Make input_ids path and inputs_embeds path use the same position_ids.
    This is safer for left-padding.
    """
    position_ids = attn_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attn_mask == 0, 0)
    return position_ids


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Equivalent to HF Llama/Mistral repeat_kv.
    hidden_states: [bsz, num_kv_heads, seq_len, head_dim]
    return: [bsz, num_heads, seq_len, head_dim]
    """
    if n_rep == 1:
        return hidden_states

    bsz, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        bsz, num_kv_heads, n_rep, seq_len, head_dim
    )
    return hidden_states.reshape(bsz, num_kv_heads * n_rep, seq_len, head_dim)

def _get_attention_module_config(module):
    """
    Get attention config for Llama/Mistral attention module.
    Compatible with different HuggingFace versions.
    """
    config = getattr(module, "config", None)

    num_heads = getattr(module, "num_heads", None)
    if num_heads is None:
        if config is None:
            raise ValueError("Cannot find num_heads and module.config is None.")
        num_heads = config.num_attention_heads

    num_kv_heads = getattr(module, "num_key_value_heads", None)
    if num_kv_heads is None:
        if config is not None and hasattr(config, "num_key_value_heads"):
            num_kv_heads = config.num_key_value_heads
        else:
            num_kv_heads = num_heads

    num_kv_groups = getattr(module, "num_key_value_groups", None)
    if num_kv_groups is None:
        num_kv_groups = num_heads // num_kv_heads

    head_dim = getattr(module, "head_dim", None)
    if head_dim is None:
        hidden_size = getattr(module, "hidden_size", None)
        if hidden_size is None:
            if config is None:
                raise ValueError("Cannot find hidden_size and module.config is None.")
            hidden_size = config.hidden_size
        head_dim = hidden_size // num_heads

    return num_heads, num_kv_heads, num_kv_groups, head_dim


@contextmanager
def freeze_attention_context(hf_model, clean_attentions):
    """
    Temporarily freeze all decoder self-attention probabilities.

    Clean forward:
        A_clean = softmax(Q_clean K_clean^T)

    Perturbed forward inside this context:
        output = A_clean @ V_perturbed

    So attention routing is fixed, but value/content stream still changes.
    """
    handles = []
    layers = hf_model.model.layers

    if clean_attentions is None:
        raise ValueError("clean_attentions is None, but freeze_attention is enabled.")

    if len(clean_attentions) != len(layers):
        raise ValueError(
            f"clean_attentions length {len(clean_attentions)} != number of layers {len(layers)}"
        )

    def make_hook(layer_idx):
        clean_attn = clean_attentions[layer_idx].detach()

        def hook(module, inputs, kwargs, output):
            # New HF Llama/Mistral often passes hidden_states by keyword.
            if "hidden_states" in kwargs:
                hidden_states = kwargs["hidden_states"]
            elif len(inputs) > 0:
                hidden_states = inputs[0]
            else:
                raise ValueError(
                    f"Cannot find hidden_states in attention hook at layer {layer_idx}. "
                    f"inputs={inputs}, kwargs keys={list(kwargs.keys())}"
                )

            batch_size, q_len, _ = hidden_states.shape
            device = hidden_states.device

            num_heads, num_kv_heads, num_kv_groups, head_dim = _get_attention_module_config(module)

            # Recompute V from perturbed hidden_states.
            # Use F.linear instead of module.v_proj(...) to avoid extra hook side effects.
            value_states = F.linear(
                hidden_states,
                module.v_proj.weight,
                module.v_proj.bias,
            )

            value_states = value_states.view(
                batch_size, q_len, num_kv_heads, head_dim
            ).transpose(1, 2)

            value_states = repeat_kv(value_states, num_kv_groups)
            # value_states: [batch, num_heads, seq_len, head_dim]

            attn_probs = clean_attn.to(device=device, dtype=value_states.dtype)

            # If perturbed batch is mc_samples * original_batch,
            # repeat clean attention along batch dimension.
            if attn_probs.shape[0] != batch_size:
                if batch_size % attn_probs.shape[0] != 0:
                    raise ValueError(
                        f"Cannot repeat clean attention from batch={attn_probs.shape[0]} "
                        f"to perturbed batch={batch_size}"
                    )
                repeat_factor = batch_size // attn_probs.shape[0]
                attn_probs = attn_probs.repeat(repeat_factor, 1, 1, 1)

            kv_len = value_states.shape[-2]

            # Usually q_len == kv_len in full forward.
            # Slice only for safety.
            attn_probs = attn_probs[:, :, :q_len, :kv_len]

            if attn_probs.shape[-2] != q_len or attn_probs.shape[-1] != kv_len:
                raise ValueError(
                    f"Frozen attention shape mismatch at layer {layer_idx}: "
                    f"attn_probs={attn_probs.shape}, value_states={value_states.shape}"
                )

            # Frozen routing with perturbed values.
            attn_output = torch.matmul(attn_probs, value_states)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(batch_size, q_len, num_heads * head_dim)

            # Output projection.
            attn_output = F.linear(
                attn_output,
                module.o_proj.weight,
                module.o_proj.bias,
            )

            if isinstance(output, tuple):
                output = list(output)
                output[0] = attn_output

                # Keep attention output compatible.
                if len(output) > 1:
                    output[1] = attn_probs

                return tuple(output)

            return attn_output

        return hook

    for layer_idx, layer in enumerate(layers):
        handles.append(
            layer.self_attn.register_forward_hook(
                make_hook(layer_idx),
                with_kwargs=True,
            )
        )

    try:
        yield
    finally:
        for h in handles:
            h.remove()


def next_token_logits_with_optional_frozen_attention(
    attributor,
    input_ids: torch.Tensor,
    attn_mask: torch.Tensor,
    inputs_embeddings: torch.Tensor | None = None,
    frozen_attentions=None,
    return_attentions: bool = False,
):
    """
    One-step next-token forward.

    If frozen_attentions is None:
        normal forward.
    If frozen_attentions is not None:
        forward with frozen attention probabilities.
    """
    hf_model = attributor.model

    position_ids = build_position_ids(attn_mask).to(attn_mask.device)

    kwargs = {
        "attention_mask": attn_mask,
        "position_ids": position_ids,
        "use_cache": False,
        "return_dict": True,
        # Need attentions for clean pass, and also keeps attention hook output structure stable.
        "output_attentions": return_attentions or (frozen_attentions is not None),
    }

    if inputs_embeddings is None:
        kwargs["input_ids"] = input_ids
    else:
        kwargs["inputs_embeds"] = inputs_embeddings

    with torch.no_grad():
        if frozen_attentions is None:
            outputs = hf_model(**kwargs)
        else:
            with freeze_attention_context(hf_model, frozen_attentions):
                outputs = hf_model(**kwargs)

    next_logits = outputs.logits[:, -1, :]

    if return_attentions:
        return next_logits, outputs.attentions

    return next_logits, None



def get_suffix_and_class_ids(args: argparse.Namespace, tokenizer, model):
    model_device = next(model.parameters()).device

    if args.model.startswith("llama"):
        suffix = 5
        class_map = {
            "IMDb": ["Negative", "Positive"],
            "SST2": ["Negative", "Positive"],
            "BoolQ": ["false", "true"],
        }
    elif args.model.startswith("Mistral"):
        suffix = 1
        class_map = {
            "IMDb": ["Neg", "Pos"],
            "SST2": ["Neg", "Pos"],
            "BoolQ": ["false", "true"],
        }
    else:
        raise ValueError(f"Unsupported model family for suffix/class_ids: {args.model}")

    class_ids = None
    if args.dataset in class_map:
        class_ids = tokenizer(
            class_map[args.dataset],
            return_tensors="pt",
            add_special_tokens=False,
        ).to(model_device)["input_ids"].squeeze(1)

    return suffix, class_ids


def soft_ns_nc(
    args,
    input_ids,
    emap,
    attn_mask,
    prefix,
    suffix,
    model,
    label,
    padding_counts,
    evaluate_token_idx,
    feature_retained,
    scaling_function,
    distance_function,
    class_ids,
    seed=42,
    mc_samples=5,
):
    reset_seeds(seed)
    batch_size = input_ids.shape[0]
    device = model.model.device

    with torch.no_grad():
        embeddings = model.model.get_input_embeddings()(input_ids)

    clean_attentions = None

    if args.dataset in SUPPORTED_CLASSIFICATION_DATASETS:
        retaining_perturbed_embeddings = embeddings.clone()
        removing_perturbed_embeddings = embeddings.clone()
        baseline_embedding = embeddings.clone()
        emap_masks = None
    else:
        reset_seeds(seed)

        original_logits, total_sequence = model.forward(
            input_ids=input_ids,
            attn_mask=attn_mask,
            max_generated_length=64
        )

        # original_logit = original_logits[evaluate_token_idx]
        # generated_sequence_len = len(original_logits)
        # input_len = input_ids[0].shape[0]
        # if generated_sequence_len <= evaluate_token_idx:
        #     return 0.0, 0.0, False
        generated_sequence_len = len(original_logits)
        input_len = input_ids[0].shape[0]

        if not args.freeze_attention:
            original_logit = original_logits[evaluate_token_idx]




        input_ids = torch.cat((input_ids, total_sequence[:, input_len: input_len + evaluate_token_idx]), dim=1)
        attn_mask = torch.cat(
            (
                attn_mask,
                torch.ones((batch_size, evaluate_token_idx), dtype=attn_mask.dtype, device=device),
            ),
            dim=1,
        )

        with torch.no_grad():
            embeddings = model.model.get_input_embeddings()(input_ids)

        if args.freeze_attention:
            original_logit, clean_attentions = next_token_logits_with_optional_frozen_attention(
                attributor=model,
                input_ids=input_ids,
                attn_mask=attn_mask,
                inputs_embeddings=None,
                frozen_attentions=None,
                return_attentions=True,
            )

        retaining_perturbed_embeddings = embeddings.clone()
        removing_perturbed_embeddings = embeddings.clone()
        baseline_embedding = embeddings.clone()
        emap_masks = torch.ones(batch_size, device=device)

    hidden_dim = embeddings.shape[-1]
    new_batch_size = batch_size * mc_samples if mc_samples > 1 else batch_size

    if mc_samples > 1:
        repl_input_ids = input_ids.unsqueeze(0).expand(mc_samples, -1, -1).reshape(new_batch_size, -1)
        repl_attn_mask = attn_mask.unsqueeze(0).expand(mc_samples, -1, -1).reshape(new_batch_size, -1)
        retaining_perturbed_embeddings = embeddings.unsqueeze(0).expand(mc_samples, -1, -1, -1).clone()
        removing_perturbed_embeddings = embeddings.unsqueeze(0).expand(mc_samples, -1, -1, -1).clone()
    else:
        retaining_perturbed_embeddings = retaining_perturbed_embeddings.unsqueeze(0)
        removing_perturbed_embeddings = removing_perturbed_embeddings.unsqueeze(0)
        repl_input_ids = input_ids
        repl_attn_mask = attn_mask

    for idx in range(batch_size):
        if args.evaluation_mode == "per_sentence":
            if emap[idx].shape[1] < (evaluate_token_idx + 1):
                emap_masks[idx] = 0
                valid_emap = emap[idx][:, 0][prefix:-suffix].to(device)
            else:
                valid_emap = emap[idx][:, evaluate_token_idx][prefix:-suffix].to(device)
        else:
            if args.dataset == "BoolQ":
                passage_token_length = suffix[idx]
                valid_emap = emap[idx][prefix: prefix + passage_token_length].to(device)
            else:
                valid_emap = emap[idx][prefix:-suffix].to(device)

        min_val = valid_emap.min()
        max_val = valid_emap.max()
        valid_emap = (valid_emap - min_val) / (max_val - min_val + 1e-13)

        scaled_emap = scaling_function(valid_emap, feature_retained)

        p_retaining = scaled_emap.view(1, -1, 1).expand(mc_samples, -1, hidden_dim)
        retaining_bernoulli_mask = torch.bernoulli(p_retaining).float()

        p_removing = 1 - scaled_emap.view(1, -1, 1).expand(mc_samples, -1, hidden_dim)
        removing_bernoulli_mask = torch.bernoulli(p_removing).float()

        if args.dataset == "BoolQ":
            start = padding_counts[idx] + prefix
            end = start + passage_token_length
        else:
            start = padding_counts[idx] + prefix
            end = input_ids.shape[1] - (suffix + evaluate_token_idx)

        retaining_perturbed_embeddings[:, idx, start:end, :] = (
            retaining_perturbed_embeddings[:, idx, start:end, :] * retaining_bernoulli_mask
        )
        removing_perturbed_embeddings[:, idx, start:end, :] = (
            removing_perturbed_embeddings[:, idx, start:end, :] * removing_bernoulli_mask
        )
        baseline_embedding[idx, start:end, :] = 0

    if mc_samples > 1:
        retaining_perturbed_embeddings = retaining_perturbed_embeddings.reshape(new_batch_size, -1, hidden_dim)
        removing_perturbed_embeddings = removing_perturbed_embeddings.reshape(new_batch_size, -1, hidden_dim)
    else:
        retaining_perturbed_embeddings = retaining_perturbed_embeddings.squeeze(0)
        removing_perturbed_embeddings = removing_perturbed_embeddings.squeeze(0)

    # if args.dataset in SUPPORTED_CLASSIFICATION_DATASETS:
    #     original_logits, _ = model.forward(
    #         input_ids=input_ids,
    #         attn_mask=attn_mask,
    #         max_generated_length=1,
    #         inputs_embeddings=embeddings,
    #     )
    #     original_logit = original_logits[0]
    #     target_logits = original_logit[:, class_ids]
    #     pred_labels = torch.argmax(target_logits, dim=1)
    #     mask = (pred_labels == label).float()
    if args.dataset in SUPPORTED_CLASSIFICATION_DATASETS:
        if args.freeze_attention:
            original_logit, clean_attentions = next_token_logits_with_optional_frozen_attention(
                attributor=model,
                input_ids=input_ids,
                attn_mask=attn_mask,
                inputs_embeddings=None,
                frozen_attentions=None,
                return_attentions=True,
            )
        else:
            original_logits, _ = model.forward(
                input_ids=input_ids,
                attn_mask=attn_mask,
                max_generated_length=1,
                inputs_embeddings=embeddings,
            )
            original_logit = original_logits[0]

        target_logits = original_logit[:, class_ids]
        pred_labels = torch.argmax(target_logits, dim=1)
        mask = (pred_labels == label).float()

    original_prob = torch.softmax(original_logit, dim=1)
    if mc_samples > 1:
        original_prob = original_prob.unsqueeze(0).expand(mc_samples, -1, -1)

    # baseline_logits, _ = model.forward(
    #     input_ids=input_ids,
    #     attn_mask=attn_mask,
    #     max_generated_length=1,
    #     inputs_embeddings=baseline_embedding,
    # )
    # baseline_logit = baseline_logits[0]
    if args.freeze_attention:
        baseline_logit, _ = next_token_logits_with_optional_frozen_attention(
            attributor=model,
            input_ids=input_ids,
            attn_mask=attn_mask,
            inputs_embeddings=baseline_embedding,
            frozen_attentions=clean_attentions,
            return_attentions=False,
        )
    else:
        baseline_logits, _ = model.forward(
            input_ids=input_ids,
            attn_mask=attn_mask,
            max_generated_length=1,
            inputs_embeddings=baseline_embedding,
        )
        baseline_logit = baseline_logits[0]
    baseline_prob = torch.softmax(baseline_logit, dim=1)
    if mc_samples > 1:
        baseline_prob = baseline_prob.unsqueeze(0).expand(mc_samples, -1, -1)

    # retaining_logits, _ = model.forward(
    #     input_ids=repl_input_ids,
    #     attn_mask=repl_attn_mask,
    #     max_generated_length=1,
    #     inputs_embeddings=retaining_perturbed_embeddings,
    # )
    # retaining_logit = retaining_logits[0]
    if args.freeze_attention:
        retaining_logit, _ = next_token_logits_with_optional_frozen_attention(
            attributor=model,
            input_ids=repl_input_ids,
            attn_mask=repl_attn_mask,
            inputs_embeddings=retaining_perturbed_embeddings,
            frozen_attentions=clean_attentions,
            return_attentions=False,
        )
    else:
        retaining_logits, _ = model.forward(
            input_ids=repl_input_ids,
            attn_mask=repl_attn_mask,
            max_generated_length=1,
            inputs_embeddings=retaining_perturbed_embeddings,
        )
        retaining_logit = retaining_logits[0]
    if mc_samples > 1:
        retaining_logit = retaining_logit.view(mc_samples, batch_size, -1)
    retaining_prob = torch.softmax(retaining_logit, dim=-1)

    # removing_logits, _ = model.forward(
    #     input_ids=repl_input_ids,
    #     attn_mask=repl_attn_mask,
    #     max_generated_length=1,
    #     inputs_embeddings=removing_perturbed_embeddings,
    # )
    # removing_logit = removing_logits[0]
    if args.freeze_attention:
        removing_logit, _ = next_token_logits_with_optional_frozen_attention(
            attributor=model,
            input_ids=repl_input_ids,
            attn_mask=repl_attn_mask,
            inputs_embeddings=removing_perturbed_embeddings,
            frozen_attentions=clean_attentions,
            return_attentions=False,
        )
    else:
        removing_logits, _ = model.forward(
            input_ids=repl_input_ids,
            attn_mask=repl_attn_mask,
            max_generated_length=1,
            inputs_embeddings=removing_perturbed_embeddings,
        )
        removing_logit = removing_logits[0]
    if mc_samples > 1:
        removing_logit = removing_logit.view(mc_samples, batch_size, -1)
    removing_prob = torch.softmax(removing_logit, dim=-1)

    delta_p0 = distance_function(original_prob, baseline_prob)
    delta_p_retaining = distance_function(original_prob, retaining_prob)
    delta_p_removing = distance_function(original_prob, removing_prob)

    if mc_samples > 1:
        delta_p0 = delta_p0.mean(0)
        delta_p_retaining = delta_p_retaining.mean(0)
        delta_p_removing = delta_p_removing.mean(0)

    numerator_ns = torch.maximum(delta_p0 - delta_p_retaining, torch.tensor(0.0, device=device))
    soft_ns = numerator_ns / (delta_p0 + 1e-12)
    soft_nc = delta_p_removing / (delta_p0 + 1e-12)

    if args.dataset in SUPPORTED_CLASSIFICATION_DATASETS:
        soft_ns = (soft_ns * mask).mean().item()
        soft_nc = (soft_nc * mask).mean().item()

    else:
        soft_ns = soft_ns[emap_masks.bool()].mean().item()
        soft_nc = soft_nc[emap_masks.bool()].mean().item()

    return soft_ns, soft_nc


def evaluate(
    args,
    dataloader,
    attributor,
    prefix,
    suffix,
    class_ids,
    scaling_function,
    distance_function,
):
    p_retain_list = np.arange(0.05, 0.96, 0.05)
    results = {
        "soft_ns": {p: [] for p in p_retain_list},
        "soft_nc": {p: [] for p in p_retain_list},
        "soft_ns_auc": 0.0,
        "soft_nc_auc": 0.0,
    }

    for input_ids, attn_mask, labels, emaps, padding_counts, passage_token_length in tqdm(dataloader):


        input_ids = input_ids.to(attributor.model.device)
        attn_mask = attn_mask.to(attributor.model.device)
        labels = labels.to(attributor.model.device)
        padding_counts = padding_counts.to(attributor.model.device)

        for p_retain in p_retain_list:
            seed_idx = args.seed
            if args.evaluation_mode == "per_token":
                target_token_idx = 0
                emap = [tensor[:, target_token_idx] for tensor in emaps]

                runtime_suffix = passage_token_length if args.dataset == "BoolQ" else suffix
                soft_ns, soft_nc = soft_ns_nc(
                    args,
                    input_ids=input_ids,
                    emap=emap,
                    attn_mask=attn_mask,
                    prefix=prefix,
                    suffix=runtime_suffix,
                    model=attributor,
                    label=labels,
                    padding_counts=padding_counts,
                    evaluate_token_idx=target_token_idx,
                    feature_retained=p_retain,
                    scaling_function=scaling_function,
                    distance_function=distance_function,
                    class_ids=class_ids,
                    seed=seed_idx,
                    mc_samples=args.mc_samples,
                )
                results["soft_ns"][p_retain].append(soft_ns)
                results["soft_nc"][p_retain].append(soft_nc)

            elif args.evaluation_mode == "per_sentence":
                max_length = min(max([tensor.shape[1] for tensor in emaps]), 64)
                # max_length = 64

                for i in range(0, max_length, args.stride):
                    soft_ns, soft_nc = soft_ns_nc(
                        args,
                        input_ids=input_ids,
                        emap=emaps,
                        attn_mask=attn_mask,
                        prefix=prefix,
                        suffix=suffix,
                        model=attributor,
                        label=labels,
                        padding_counts=padding_counts,
                        evaluate_token_idx=i,
                        feature_retained=p_retain,
                        scaling_function=scaling_function,
                        distance_function=distance_function,
                        class_ids=class_ids,
                        seed=seed_idx,
                        mc_samples=args.mc_samples,
                    )
                    results["soft_ns"][p_retain].append(soft_ns)
                    results["soft_nc"][p_retain].append(soft_nc)

            else:
                raise ValueError(f"Unsupported evaluation_mode: {args.evaluation_mode}")

    return p_retain_list, results


def write_results(writer, p_retain_list, results):
    for p_retain in p_retain_list:
        results["soft_ns"][p_retain] = np.mean(results["soft_ns"][p_retain])
        results["soft_nc"][p_retain] = np.mean(results["soft_nc"][p_retain])

    soft_ns_values = [results["soft_ns"][p] for p in p_retain_list]
    soft_nc_values = [results["soft_nc"][p] for p in p_retain_list]

    results["soft_ns_auc"] = auc(p_retain_list, soft_ns_values)
    results["soft_nc_auc"] = auc(p_retain_list, soft_nc_values)

    writer.add_scalar("AUC/Soft_NS_AUC", results["soft_ns_auc"], 0)
    writer.add_scalar("AUC/Soft_NC_AUC", results["soft_nc_auc"], 0)

    for idx, p_retain in enumerate(p_retain_list):
        writer.add_scalar("Soft_NS", results["soft_ns"][p_retain], idx)
        writer.add_scalar("Soft_NC", results["soft_nc"][p_retain], idx)


def main():
    args = parse_args()
    print(args)

    log_dir = build_log_dir(args)
    writer = create_writer(args, log_dir)

    reset_seeds(args.seed)
    dataset_path, heatmap_path = build_dataset_paths(args)

    tokenizer, model = resolve_model(args)
    attributor = AttributionGenerator(model, tokenizer, n_layers=args.num_layers)

    datasets = CustomizedDataset(
        emap_path=heatmap_path,
        csv_path=dataset_path,
        tokenizer=tokenizer,
        dataset_tag=args.dataset,
        model_tag=args.model,
    )
    if args.num_samples is None:
        num_samples = 20 if args.dataset in {"TellMeWhy", "WikiBio"} else 200
    else:
        num_samples = args.num_samples
    num_samples = min(num_samples, len(datasets))
    start_idx = len(datasets) - num_samples
    subset = Subset(datasets, range(start_idx, len(datasets)))
    dataloader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=datasets.collate_fn,
    )

    scaling_function = get_scaling_function(args.scaling_method)
    distance_function = get_distance_function(args.distance_type)
    prefix = get_prefix(args.model, args.dataset)
    suffix, class_ids = get_suffix_and_class_ids(args, tokenizer, model)

    p_retain_list, results = evaluate(
        args=args,
        dataloader=dataloader,
        attributor=attributor,
        prefix=prefix,
        suffix=suffix,
        class_ids=class_ids,
        scaling_function=scaling_function,
        distance_function=distance_function,
    )

    # Always compute/print aggregate results. TensorBoard logging is optional.
    for p_retain in p_retain_list:
        results["soft_ns"][p_retain] = np.mean(results["soft_ns"][p_retain])
        results["soft_nc"][p_retain] = np.mean(results["soft_nc"][p_retain])
    soft_ns_values = [results["soft_ns"][p] for p in p_retain_list]
    soft_nc_values = [results["soft_nc"][p] for p in p_retain_list]
    results["soft_ns_auc"] = auc(p_retain_list, soft_ns_values)
    results["soft_nc_auc"] = auc(p_retain_list, soft_nc_values)
    print(f"Soft-NS AUC: {results['soft_ns_auc']:.6f}")
    print(f"Soft-NC AUC: {results['soft_nc_auc']:.6f}")

    if writer is not None:
        writer.add_scalar("AUC/Soft_NS_AUC", results["soft_ns_auc"], 0)
        writer.add_scalar("AUC/Soft_NC_AUC", results["soft_nc_auc"], 0)
        for idx, p_retain in enumerate(p_retain_list):
            writer.add_scalar("Soft_NS", results["soft_ns"][p_retain], idx)
            writer.add_scalar("Soft_NC", results["soft_nc"][p_retain], idx)
        writer.close()


if __name__ == "__main__":
    main()
