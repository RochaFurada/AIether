import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from aiether.utils import logger_setup

logger = logger_setup()

class GPTConfig(PretrainedConfig):
    """
    GPT model configuration class compatible with Hugging Face interface.
    """
    model_type = "gpt_custom"
    
    def __init__(
        self,
        block_size: int = 1024,
        vocab_size: int = 50304,
        n_layer: int = 12,
        n_head: int = 12,
        n_embd: int = 768,
        dropout: float = 0.0,
        bias: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.bias = bias

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # Query, key and value projections for all attention heads
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # Dropout layers for regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # Causal mask for autoregressive attention
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch, sequence, embedding

        # Calculate query, key, value and rearrange dimensions
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # Causal autoregressive attention
        if self.flash:
            # Uses optimized Flash Attention kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # Manual attention implementation
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Final output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd * 2, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = self.c_fc(x)
        x, gate = x.chunk(2, dim=-1) 
        x = F.silu(x) * gate 
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPTForHF(PreTrainedModel):
    """
    GPT model implementation compatible with Hugging Face Trainer interface.
    """
    def __init__(self, config):
        
        super().__init__(config)
        self._tied_weights_keys = ["lm_head.weight"]
        self.config = config
        # Model architecture construction
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd),
        ))
        
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        
        # Weight initialization
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # Zerofill initialization for inactive layers
        self._initialize_weights_layers_zerofill(self.transformer)

        self.active_layers = []
        for i, block in enumerate(self.transformer.h):
            block.requires_grad_(i == len(self.transformer.h) - 1)
            if i == len(self.transformer.h) - 1:
                self.active_layers.append(i)

        # Layer status logging
        for i, block in enumerate(self.transformer.h):
            has_grad = any(p.requires_grad for p in block.parameters())
            logger.infoG(f"Layer {i}: {has_grad}")

        logger.infoB(f"✅ GPTForHF model initialized with {len(self.transformer.h)} layers. Active layers: {self.active_layers}")

    def _init_weights(self, module):
        """
        Initializes all model weights.
        """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _initialize_weights_layers_zerofill(self, transformer):
        """
        Initializes all inactive layer weights with zeros.
        """
        logger.info("🔧 Initializing frozen layers with zerofill...")
        for i, block in enumerate(transformer.h):
            if i != len(transformer.h) - 1:
                for p in block.parameters():
                    if p.dim() > 1:
                        torch.nn.init.zeros_(p)
                logger.info(f"   Layer {i}: weights initialized with zerofill")

    def initialize_new_layer(self, layer_id: int, strategy: str, initializer, layer_manager, param_history=None):
        """
        Initializes layer using temporal parameter history.
        
        Args:
            layer_id: Index of layer to be initialized.
            strategy: Name of initialization strategy.
            initializer: GrowthManager instance.
            layer_manager: LayerStateManager instance.
            param_history: Dictionary mapping parameter names to temporal snapshot lists.
        """
        logger.infoB(f"🧬 Initializing layer {layer_id} weights via '{strategy}' strategy...")

        # History validation
        if param_history is None:
            logger.error(f"❌ param_history is None! Strategy {strategy} requires temporal history.")
            return

        # Weight extrapolation and initialization
        initializer.initialize_new_layer_from_history(
            new_layer=self.transformer.h[layer_id],
            param_history=param_history,
            strategy=strategy
        )

        # Layer metadata registration
        strategy_params = self._get_strategy_params(initializer)

        layer_manager.register_layer(
            layer_id=layer_id,
            strategy=strategy,
            strategy_params=strategy_params,
            generation_step=None,
        )

        # Layer initial state persistence
        layer_manager.save_state(
            layer_id=layer_id,
            state_name="L0",
            state_dict=self.transformer.h[layer_id].state_dict(),
            step=None,
            strategy=strategy,
            metrics=strategy_params
        )

        logger.infoG(f"  Layer {layer_id} initialized and registered successfully.")


    def _get_strategy_params(self, initializer) -> dict:
        """
        Extracts initialization strategy parameters from initializer.
        
        Args:
            initializer: GrowthManager instance
        
        Returns:
            Dictionary containing parameters of strategy used.
        """
        # Recover parameters stored in initializer
        if hasattr(initializer, 'last_strategy_params'):
            return initializer.last_strategy_params.copy()
        
        # Fallback if attribute not available
        logger.warning(f"⚠️  Initializer does not have 'last_strategy_params', returning empty dict")
        return {}
        
    def set_active_layers(self, new_layer_id: int):
            """
            Adds specified layer to active layers list.
            Does not modify requires_grad state of other layers.
            """
            # Update active layers registry
            if new_layer_id not in self.active_layers:
                self.active_layers.append(new_layer_id)
                self.active_layers.sort() 
            
            logger.info(f"✅ Layer {new_layer_id} activated for training. Current list: {self.active_layers}")

    def deactivate_layers(self, layer_id: int):
        """
        Deactivates specified layer by setting requires_grad to False.
        Preserves weights and parameter content.
        """
        logger.infoB(f"🔒 Deactivating layer {layer_id}...")

        if layer_id in self.active_layers:
            for p in self.transformer.h[layer_id].parameters():
                p.requires_grad = False
            self.active_layers.remove(layer_id)

        logger.infoG(f"🔒 Active layers now: {self.active_layers}")

        
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """
        Executes model forward pass compatible with Hugging Face Trainer interface.
        """
        device = input_ids.device
        b, t = input_ids.size()
        
        assert t <= self.config.block_size, \
            f"Sequence of size {t} exceeds block_size {self.config.block_size}"
        
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        
        # Token and position embeddings
        tok_emb = self.transformer.wte(input_ids)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        for i in self.active_layers:
            x = self.transformer.h[i](x)
        
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        
        # Loss calculation when labels are provided
        loss = None
        if labels is not None:
            # Shift for autoregressive prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
        
        # Return in CausalLMOutputWithPast format
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )
    
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generates token sequence autoregressively.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            outputs = self(idx_cond)
            logits = outputs.logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx
