import torch
import torch.nn as nn
from torch.nn import functional as F
from .attention import CausalSelfAttention

class GPT2Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
        # Fix: Ensure n_inner is always available
        n_inner = config.n_inner if hasattr(config, 'n_inner') and config.n_inner is not None else 4 * config.n_embd
        
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, n_inner),
            nn.GELU(),
            nn.Linear(n_inner, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.wte = nn.Embedding(config.vocab_size, config.n_embd) # Token embeddings
        self.wpe = nn.Embedding(config.n_positions, config.n_embd) # Position embeddings
        self.drop = nn.Dropout(config.embd_pdrop)
        
        self.blocks = nn.ModuleList([GPT2Block(config) for _ in range(config.n_layer)])
        
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, position_ids=None):
        B, T = input_ids.size()
        
        if position_ids is None:
            position_ids = torch.arange(0, T, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        
        # forward the GPT model
        token_embeddings = self.wte(input_ids)
        position_embeddings = self.wpe(position_ids)
        x = self.drop(token_embeddings + position_embeddings)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.ln_f(x)
        
        logits = torch.matmul(x, self.wte.weight.transpose(0, 1))
        
        return logits 