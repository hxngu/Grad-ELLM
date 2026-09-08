import torch
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np


class AttributionGenerator:
    def __init__(self, model, tokenizer, n_layers=1, normalize=True):
        self.model = model
        self.tokenizer = tokenizer
        self.n_layers = n_layers
        self.hooks = []
        self.normalize = normalize

        if self.normalize:
            self._sim_fuc = self._sim_qk
        else:
            self._sim_fuc = self._sim_qk_original

        self.attention_outputs = [[] for _ in range(self.n_layers)]
        self.attention_weights = [[] for _ in range(self.n_layers)]
        self.projection_q = [[] for _ in range(self.n_layers)]
        self.projection_k = [[] for _ in range(self.n_layers)]
        self.projection_v = [[] for _ in range(self.n_layers)]

        self._register_hooks(self.model.model.layers[-self.n_layers:])

    def _register_hooks(self, layers):
        for layer_idx, layer in enumerate(layers):
            # Register hooks for attention outputs
            def make_attn_hook(idx):
                def hook(module, input, output):
                    # Store attention outputs
                    pre_o_output = input[0]
                    if len(self.attention_outputs[idx]) > 0:
                        self.attention_outputs[idx].pop()
                    self.attention_outputs[idx].append(pre_o_output)
                    # Store attention weights
                    # if len(self.attention_weights[idx]) > 0:
                    #     self.attention_weights[idx].pop()
                    # self.attention_weights[idx].append(output[1].detach())
                return hook
            self.hooks.append(layer.self_attn.o_proj.register_forward_hook(make_attn_hook(layer_idx)))

            # Register hooks for projection layers
            def make_proj_hook(idx, proj_lst):
                def hook(module, input, output):
                    if len(proj_lst[idx]) > 0:
                        proj_lst[idx].pop()
                    proj_lst[idx].append(output.detach())
                return hook
            self.hooks.append(layer.self_attn.q_proj.register_forward_hook(make_proj_hook(layer_idx, self.projection_q)))
            self.hooks.append(layer.self_attn.k_proj.register_forward_hook(make_proj_hook(layer_idx, self.projection_k)))
            self.hooks.append(layer.self_attn.v_proj.register_forward_hook(make_proj_hook(layer_idx, self.projection_v)))

    @staticmethod
    def _sim_qk(q, k):
        q_norm = F.normalize(q, dim=-1)
        k_norm = F.normalize(k, dim=-1)

        cosine_qk = (q_norm * k_norm).sum(-1)
        cosine_qk = (cosine_qk-cosine_qk.min()) / (cosine_qk.max()-cosine_qk.min())
        return cosine_qk

    @staticmethod
    def _sim_qk_original(q, k):
        hidden_dimension = k.shape[-1]
        # q [batch, seq_len, hidden_dimension], k [batch, seq_len, hidden_dimension]
        dot_product = (q * k).sum(-1)  # [batch, seq_len]
        dot_product = dot_product / np.sqrt(hidden_dimension)  # scale
        return torch.softmax(dot_product, dim=-1)  # [batch, seq_len]

    def _grad_ellm(self, k, q, v, grad):
        # process the GQA
        k = k.repeat(1, 1, 4) if k.shape[-1] < q.shape[-1] else k
        v = v.repeat(1, 1, 4) if v.shape[-1] < q.shape[-1] else v

        # cosine_qk = self._sim_qk(q, k)
        cosine_qk = self._sim_fuc(q, k)

        emap = grad.unsqueeze(1) * v * cosine_qk.unsqueeze(-1)
        emap = F.relu(emap.sum(-1))
        return emap

    def _padding_emap(self, original_emap_lst):
        max_length = original_emap_lst[-1].shape[1]
        padded_tensors = []
        for tensor in original_emap_lst:
            pad_size = max_length - tensor.shape[1]
            padded_tensor = torch.nn.functional.pad(tensor, (0, pad_size), mode='constant', value=0)
            padded_tensors.append(padded_tensor)
        padded_tensors = torch.stack(padded_tensors, dim=1)  # [batch_size, output - input, output]

        return padded_tensors


    def generate(self, input_text, max_generated_length=300, temperature=0.7):
        # Tokenize input text
        # messages = [
        #                 {
        #                     "role": "system",
        #                     "content": "You are a helpful sentiment classifier. Please help to do the sentiment classification of the given text and respond ONLY with the single word Positive or Negative.",
        #                 },
        #                 {"role": "user", "content": "Text: {}".format(input_text)},
        #             ]
        # input_text = self.tokenizer.apply_chat_template(messages, return_tensors="pt", tokenize=False)

        input_ids = self.tokenizer.encode(input_text, return_tensors="pt", add_special_tokens=False).to(self.model.device)

        # input_ids = self.tokenizer.encode(input_text, return_tensors="pt").to(self.model.device)
        # input_ids = torch.cat((input_ids, torch.tensor([[self.tokenizer.eos_token_id]]).to(self.model.device)), dim=1)
        # self.tokenizer.pad_token = self.tokenizer.eos_token
        # model_inputs = self.tokenizer(input_text, return_tensors="pt", padding=True).to(self.model.device)
        # input_ids = model_inputs["input_ids"]

        input_length = input_ids.shape[1]
        emap_lst = []
        original_emap_lst = []

        # Generate tokens
        for _ in tqdm(range(max_generated_length)):
            outputs = self.model(input_ids)
            logits = outputs.logits[:, -1, :]

            next_token = torch.multinomial(torch.softmax(logits / temperature, dim=-1), num_samples=1)
            input_ids = torch.cat((input_ids, next_token), dim=1)

            # Get q, k, v, attn_outputs
            attn_outputs = [layer[0] for layer in self.attention_outputs]
            qs = [q[0] for q in self.projection_q]
            ks = [k[0] for k in self.projection_k]
            vs = [v[0] for v in self.projection_v]

            grads = torch.autograd.grad(
                logits[:, next_token],
                attn_outputs,
                retain_graph=False
            )

            combined_emap = []
            for layer_idx in range(self.n_layers):
                q = qs[layer_idx]
                q_eos = q[:, -1, :]
                k = ks[layer_idx]
                v = vs[layer_idx]
                grad = grads[layer_idx][:, -1, :].detach()
                emap = self._grad_ellm(k, q_eos, v, grad)
                combined_emap.append(emap)
            combined_emap = torch.stack(combined_emap, dim=0).sum(dim=0)
            truncated_emap = combined_emap[:, :input_length]
            emap_lst.append(truncated_emap)
            original_emap_lst.append(combined_emap)


            if next_token == self.tokenizer.eos_token_id:
                break

        # Pad the emap
        original_final_emap = self._padding_emap(original_emap_lst).detach().cpu().float()
        final_emap = torch.stack(emap_lst, dim=1).detach().cpu().float()

        # Remove hooks
        for hook in self.hooks:
            hook.remove()

        self.hooks.clear()

        self._step()

        return final_emap, original_final_emap, input_ids


    def updated_generate(self, input_text, max_generated_length=300, temperature=0.7):

        # 1) official generation
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        input_len = input_ids.shape[1]

        with torch.no_grad():
            gen = self.model.generate(
                input_ids=input_ids,
                do_sample=True,
                temperature=temperature,
                top_p=1.0,
                max_new_tokens=max_generated_length,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
            )

        seq = gen.sequences
        new_len = seq.shape[1] - input_len

        emap_lst = []

        # 2) replay for gradients (teacher forcing)
        for t in range(new_len):
            prefix = seq[:, :input_len + t]          # tokens already generated so far
            target = seq[:, input_len + t]           # the "next token" HF actually sampled

            outputs = self.model(prefix)             # <-- grad enabled
            logits = outputs.logits[:, -1, :]        # [bs, vocab]

            # select the logit of the chosen target token (scalar per batch)
            target_logit = logits.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # [bs]

            attn_outputs = [layer[0] for layer in self.attention_outputs]
            qs = [q[0] for q in self.projection_q]
            ks = [k[0] for k in self.projection_k]
            vs = [v[0] for v in self.projection_v]

            grads = torch.autograd.grad(
                target_logit.sum(),   # sum -> scalar
                attn_outputs,
                retain_graph=False
            )

            combined_emap = []
            for layer_idx in range(self.n_layers):
                q = qs[layer_idx][:, -1, :]                # q at last position
                k = ks[layer_idx]
                v = vs[layer_idx]
                grad = grads[layer_idx][:, -1, :].detach()
                emap = self._grad_ellm(k, q, v, grad)
                combined_emap.append(emap)

            combined_emap = torch.stack(combined_emap, dim=0).sum(dim=0)
            emap_lst.append(combined_emap[:, :input_len])

            if target.item() == self.tokenizer.eos_token_id:
                break

        for hook in self.hooks:
            hook.remove()

        self.hooks.clear()

        self._step()

        final_emap = torch.stack(emap_lst, dim=1).detach().cpu().float()

        return final_emap, seq




    def forward(self, input_ids, attn_mask, max_generated_length=300, temperature=0.7, inputs_embeddings=None):
        with torch.no_grad():
            if inputs_embeddings is not None:
                outputs = self.model.generate(attention_mask=attn_mask, inputs_embeds=inputs_embeddings, max_new_tokens=max_generated_length, do_sample=True, temperature=temperature, top_p=1, return_dict_in_generate=True, output_logits=True, use_cache=True, pad_token_id = self.tokenizer.eos_token_id)
            else:
                outputs = self.model.generate(input_ids=input_ids, attention_mask=attn_mask, max_new_tokens=max_generated_length, do_sample=True, temperature=temperature, top_p=1, return_dict_in_generate=True, output_logits=True, use_cache=True, pad_token_id = self.tokenizer.eos_token_id)
        sequences = outputs.sequences
        logits = outputs.logits
        # for _ in range(max_generated_length):
        #     outputs = self.model(input_ids=input_ids, attention_mask=attn_mask)
        #     logits = outputs.logits
        #     logit = logits[:, -1, :]

        #     next_token = torch.multinomial(torch.softmax(logit / temperature, dim=-1), num_samples=1)
        #     input_ids = torch.cat((input_ids, next_token), dim=1)
        #     attn_mask = torch.cat((attn_mask, torch.ones((attn_mask.shape[0], 1)).to(attn_mask.device)), dim=1)
        #     if next_token == self.tokenizer.eos_token_id:
        #         break


        return logits, sequences


    def _step(self):
        self.attention_outputs = [[] for _ in range(self.n_layers)]
        self.attention_weights = [[] for _ in range(self.n_layers)]
        self.projection_q = [[] for _ in range(self.n_layers)]
        self.projection_k = [[] for _ in range(self.n_layers)]
        self.projection_v = [[] for _ in range(self.n_layers)]

        self._register_hooks(self.model.model.layers[-self.n_layers:])
