"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect

import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

def get_logit_mask(b, t, causal_pos, last_loss_token, device):
    if type(causal_pos) == int:
        logit_mask = torch.zeros(b, t).bool()
        logit_mask[:, causal_pos:last_loss_token] = True
        return logit_mask

    indexed_rows = torch.arange(t, device=device)[None, :]
    mask1 = causal_pos[:, None] < indexed_rows
    mask2 = indexed_rows < last_loss_token[:, None]
    logit_mask = mask1 * mask2

    return logit_mask


def get_mask_for_batch(dev, sz, prefix_ind, last_loss_token=None, prefixlm=False, window=False, max_context_ratio=1):
    mask = torch.tril(torch.ones(sz, sz, device=dev))

    if type(prefix_ind) == int:
        #  TODO: fix the return mask here for the case that prefixlm=False
        mask[:prefix_ind, :prefix_ind] = 1
        if window:
            breakpoint()
            mask *= torch.triu(torch.ones((sz, sz), device=dev), diagonal=-int(sz/max_context_ratio))
        return mask.bool()

    #  constructing non-causal attention mask
    prefix_ind = torch.tensor(prefix_ind, device=dev)
    mask_non_causal = torch.arange(sz, device=dev).unsqueeze(0) < prefix_ind.unsqueeze(1)

    non_causal = mask_non_causal.unsqueeze(2) * mask_non_causal.unsqueeze(1)

    #  constructing causal attention mask for the tokens after the prefix

    mask_causal = torch.arange(sz, device=dev).unsqueeze(0) < last_loss_token.unsqueeze(1)
    causal = torch.tril(mask_causal.unsqueeze(2) * mask_causal.unsqueeze(1))

    if prefixlm == False:
        final_mask = causal.to(torch.float32)
    else:
        final_mask = (non_causal | causal).to(torch.float32)

    if window:
        final_mask *= torch.triu(torch.ones((sz, sz), device=dev), diagonal=-int(sz/max_context_ratio)).unsqueeze(0)

    final_mask[final_mask == 0] = -1e9
    final_mask[final_mask == 1] = 0
    # print('last tokens are:', last_loss_token)
    return final_mask.unsqueeze(1)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def _reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    freqs_cis: complex - (seq_len, head_dim / 2)
    x: complex - (bsz, seq_len, head_dim / 2)
    """
    ndim = x.ndim
    assert 1 < ndim
    assert freqs_cis.shape == (x.shape[-2], x.shape[-1]), (
        freqs_cis.shape,
        (x.shape[-2], x.shape[-1]),
    )
    shape = [d if i == 2 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(q, k, freqs_cis):
    # q, k: (B, nh, T, hs)
    # freq_cis: (T, hs)
    # return: (B, nh, T, hs), (B, nh, T, hs)

    q_ = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
    k_ = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
    freqs_cis = _reshape_for_broadcast(freqs_cis, q_)
    xq_out = torch.view_as_real(q_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(k_ * freqs_cis).flatten(3)
    # breakpoint()
    return xq_out.type_as(q), xk_out.type_as(k)


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        self.mask = torch.tril(torch.ones(config.sequence_length, config.sequence_length))
        self.mask[:config.sequence_length // 2, :config.sequence_length // 2] = 1
        self.mask = self.mask.view(1, 1, config.sequence_length, config.sequence_length)
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            mask = torch.tril(torch.ones(config.sequence_length, config.sequence_length))
            mask[:config.sequence_length//2, :config.sequence_length//2] = 1
            # self.register_buffer("bias", mask.view(1, 1, config.sequence_length, config.sequence_length))

            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.sequence_length, config.sequence_length))
                                 .view(1, 1, config.sequence_length, config.sequence_length))


    def forward(self, x, attn_mask, freqs_cis=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        if freqs_cis is not None:
            q, k = apply_rotary_emb(q, k, freqs_cis)
        # if pe == 'rope':
        #     q, k = RotaryEmbedding.apply_rotary_pos_emb(q, k)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout)
            if torch.sum(torch.isnan(y).int()) > 0:
                breakpoint()

            # y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, attn_mask, freqs_cis=None):
        x = x + self.attn(self.ln_1(x), attn_mask, freqs_cis=freqs_cis)
        x = x + self.mlp(self.ln_2(x))
        return x

class GPTBase(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.sequence_length is not None
        self.config = config
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.max_context_ratio = config.max_context_ratio

        effective_seq_len = config.sequence_length*config.max_context_ratio if config.long_context else config.sequence_length
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            # Here I assume we generalize until max_context_ratio! TBD later...
            wpe = nn.Embedding(effective_seq_len, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))

        self.freqs_cis = None
        if config.pe == 'rope' or config.pe == 'mixed':
            self.freqs_cis = precompute_freqs_cis(config.n_embd // config.n_head, effective_seq_len)
            # breakpoint()
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self,
                idx,
                targets=None,
                pe='learnable',
                prefixlm=False,
                last_loss_token=None,
                get_logits=False,
                causal_pos=0,
                eval_normalizer=None,
                window=False):

        device = idx.device
        b, t = idx.size()
        # assert t <= self.config.sequence_length, f"Cannot forward sequence of length {t}, block size is only {self.config.sequence_length}"

        if pe == 'random':
            # generating a sorted tensor with non-repeating integers between 0 and
            # self.config.max_context_ratio*self.config.sequence_length, with size (1, t)
            max_len = self.config.max_context_ratio*self.config.sequence_length
            rand_perm = torch.randperm(max_len, device=device, dtype=torch.long)[:t]
            sorted_pos = rand_perm.sort()[0]
            pos = sorted_pos.view(1, t)  # shape (1, t)
        else:
            pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)

        if pe == 'learnable' or pe == 'mixed':
            abs_pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (1, t, n_embd)
            x = self.transformer.drop(tok_emb + abs_pos_emb)
        else:
            x = tok_emb

        attn_mask = get_mask_for_batch(device, t, causal_pos, last_loss_token, prefixlm, window, t/self.max_context_ratio)

        freqs_cis = None
        if self.config.pe == 'rope' or self.config.pe == 'mixed':
            freqs_cis = self.freqs_cis.to(x.device)[pos[0]]

        # breakpoint()
        for block in self.transformer.h:
            x = block(x, attn_mask, freqs_cis=freqs_cis)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            assert last_loss_token is not None, "last_loss_token must be provided if targets are given"

            if eval_normalizer is not None:
                logit_mask = get_logit_mask(b, t, eval_normalizer, last_loss_token, device)
            else:
                logit_mask = get_logit_mask(b, t, causal_pos, last_loss_token, device)
            num_samples = torch.sum(logit_mask).item()
            logits = self.lm_head(x)

            loss = F.cross_entropy((logits[logit_mask, :]).reshape(-1, logits.size(-1)),
                                   (targets[logit_mask]).reshape(-1),
                                   ignore_index=-1,
                                   reduction='none') #TODO: take care of it for standard context setups

        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
            loss = None
            logit_mask = None
            num_samples = None
        logits = logits if get_logits else None
        return {'logits': logits,
                'loss': loss,
                'causal_pos': causal_pos,
                'logit_mask': logit_mask,
                'num_samples': num_samples}

    def crop_sequence_length(self, sequence_length):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert sequence_length <= self.config.sequence_length
        self.config.sequence_length = sequence_length
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:sequence_length])
        for block in self.transformer.h:
            block.attn.bias = block.attn.bias[:,:,:sequence_length,:sequence_length]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        # TODO
        pass

    def get_parameter_group_specs(self):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        # need to do import here to avoid circular import (since llama imports from base here)
        from .utils import BLACKLIST_WEIGHT_MODULES

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, BLACKLIST_WEIGHT_MODULES):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # subtle: 'transformer.wte.weight' and 'lm_head.weight' are tied, so they
        # will appear in the no_decay and decay sets respectively after the above.
        # In addition, because named_parameters() doesn't return duplicates, it
        # will only return the first occurence, key'd by 'transformer.wte.weight', below.
        # so let's manually remove 'lm_head.weight' from decay set. This will include
        # this tensor into optimization via transformer.wte.weight only, and not decayed.
        decay.remove("lm_head.weight")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        return [
            {"params": sorted(list(decay))},
            {"params": sorted(list(no_decay)), "weight_decay": 0.0},
        ]


    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, prefixlm=False, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at sequence_length
            idx_cond = idx if idx.size(1) <= self.config.sequence_length else idx[:, -self.config.sequence_length:]
            # forward the model to get the logits for the index in the sequence
            # breakpoint()
            logits = self(idx_cond,
                          get_logits=True,
                          prefixlm=prefixlm,
                          causal_pos=idx_cond.size(1)-2,
                          last_loss_token=idx_cond.size(1))['logits']
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # breakpoint()
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    @torch.no_grad()
    def generate_from_string(self, in_str, max_new_tokens, temperature=1.0, prefixlm=False, top_k=None):
        # breakpoint()
        idx = torch.tensor(self.tokenizer.encode(in_str, allowed_special={"<|endoftext|>"})).view(1,-1).to(self.lm_head.weight.device)
        out_idx = self.generate(idx, max_new_tokens, temperature, prefixlm, top_k).view(-1).to('cpu').numpy()
        try:
            decoded = self.tokenizer.decode(out_idx)
        except BaseException as e:
            print(f"failed to decode {out_idx}")
            print(e)
            breakpoint()

        return decoded
