"""Dataset wrapper for precomputed token-attribution heatmaps."""

import pandas as pd
import torch
from torch.utils.data import Dataset



class CustomizedDataset(Dataset):
    def __init__(self, csv_path, emap_path, tokenizer, dataset_tag, model_tag):
        self.data = pd.read_csv(csv_path)
        self.emap_list = torch.load(emap_path, map_location="cpu")
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.dataset_tag = dataset_tag
        self.model_tag = model_tag

        if len(self.data) != len(self.emap_list):
            raise ValueError(
                f"Data/heatmap length mismatch: {len(self.data)} rows vs "
                f"{len(self.emap_list)} attribution maps."
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        emap = self.emap_list[idx]

        if self.dataset_tag == "BoolQ":
            passage_text = row["passage"]
            question_text = row["question"]
            label = torch.tensor(row["label"])
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a factual question answering system. Determine if the given "
                        "passage supports the question being true. Respond ONLY with the single "
                        "word 'true' or 'false' in lowercase, without any punctuation or "
                        "additional text."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Passage: {passage_text}\n\nQuestion: {question_text}",
                },
            ]
        elif self.dataset_tag in {"IMDb", "SST2"}:
            text = row["text"]
            label = torch.tensor(row["label"])
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful sentiment classifier. Please help to do the sentiment "
                        "classification of the given text and respond ONLY with the single word "
                        "Positive or Negative."
                    ),
                },
                {"role": "user", "content": f"Text: {text}"},
            ]
        elif self.dataset_tag in {"TellMeWhy", "WikiBio"}:
            text = row["text"]
            label = torch.tensor(0)
            messages = [{"role": "user", "content": text}]
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_tag}")

        formatted_input = self.tokenizer.apply_chat_template(messages, tokenize=False)
        if self.model_tag.startswith("llama"):
            formatted_input += "<|start_header_id|>assistant<|end_header_id|>\n"

        if self.dataset_tag == "BoolQ":
            return formatted_input, label, emap, " " + passage_text
        return formatted_input, label, emap

    def collate_fn(self, batch):
        if self.dataset_tag == "BoolQ":
            input_text, label, emap, passage_text = zip(*batch)
            passage_token = self.tokenizer(passage_text, add_special_tokens=False)["input_ids"]
            passage_token_length = [len(tokens) for tokens in passage_token]
        else:
            input_text, label, emap = zip(*batch)
            passage_token_length = None

        model_inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            add_special_tokens=False,
            padding="longest",
            padding_side="left",
        )
        input_ids = model_inputs["input_ids"]
        attention_masks = model_inputs["attention_mask"]
        labels = torch.stack(label)
        padding_counts = input_ids.shape[1] - attention_masks.sum(dim=1)

        return (
            input_ids,
            attention_masks,
            labels,
            emap,
            padding_counts,
            passage_token_length,
        )
