import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import Mesh
from omegaconf.dictconfig import DictConfig
from rope import apply_rope
import functools

class TransformerDecoder(nnx.Module):
  def __init__(self, cfg: DictConfig, rngs: nnx.Rngs):
    self.embed = nnx.Embed(num_embeddings=cfg.V, features=cfg.D, embedding_init=fsdp_init('embedding', cfg), dtype=cfg.dtype, rngs=rngs)
    self.blocks = [TransformerBlock(cfg, rngs) for _ in range(cfg.N)]
    self.out_ln = nnx.RMSNorm(cfg.D, use_scale=False, dtype=cfg.dtype, rngs=rngs)
    self.readout = nnx.Linear(in_features=cfg.D, out_features=cfg.V, use_bias=False, kernel_init=fsdp_init('zero', cfg), rngs=rngs, dtype=cfg.dtype)
    self.gradient_checkpointing = cfg.gradient_checkpointing

    self.rope = cfg.rope
    if not cfg.rope:
        self.pos_embed = nnx.Embed(num_embeddings=cfg.L, features=cfg.D, embedding_init=fsdp_init('embedding', cfg), rngs=rngs, dtype=cfg.dtype)
  
  @nnx.jit
  def get_embedding(self, x):
    """Return token embeddings for input tokens (no positional embedding)."""
    return self.embed(x)
  
  @nnx.jit
  def get_features(self, x):
    # Token + positional embedding
    h = self.embed(x)  # [B, S, D]
    if not self.rope:
      h += self.pos_embed(jnp.arange(x.shape[1])[None, ...])
    for block in self.blocks:
      block = nnx.remat(block) if self.gradient_checkpointing else block
      h = block(h)
    return h
  
  @nnx.jit
  def get_features_and_logits(self, x):
    h = self.get_features(x)
    return h, self.readout(self.out_ln(h))

  @nnx.jit
  def __call__(self, x):  # [B, S]
    h = self.get_features(x)
    h = self.out_ln(h)
    return self.readout(h)  # [B, S, O] where O is either cfg.O or cfg.V

  def get_layer_inputs(self, x):
    """Collect input activations for each Linear layer per transformer block and readout.
    Returns a dict keyed by clean param-like paths, values with shape [B,S,in].
    """
    feats = {}
    # Token + positional embedding
    h = self.embed(x)
    if not self.rope:
      h = h + self.pos_embed(jnp.arange(x.shape[1])[None, ...])
    for i, block in enumerate(self.blocks):
      # Attention
      h_ln1 = block.ln1(h)
      attn_out, attn_inputs = block.attn.compute_io(h_ln1)
      for k, v in attn_inputs.items():
        feats[f'blocks.{i}.attn.{k}'] = v
      h = h + attn_out * block.branch_multiplier
      # MLP
      h_ln2 = block.ln2(h)
      mlp_out, mlp_inputs = block.mlp.compute_io(h_ln2)
      for k, v in mlp_inputs.items():
        feats[f'blocks.{i}.mlp.{k}'] = v
      h = h + mlp_out * block.branch_multiplier
    feats['readout'] = self.out_ln(h)
    return feats


class Attention(nnx.Module):
  """Custom multi-headed attention implementation with D x D projection matrices."""
  def __init__(self, cfg: DictConfig, rngs: nnx.Rngs):
    self.num_heads = cfg.D // cfg.dh
    self.head_dim = cfg.dh
    self.scale = (1 / self.head_dim) ** 0.5
    # D x D projection matrices
    self.query_proj = nnx.Linear(cfg.D, cfg.D, use_bias=False, kernel_init=fsdp_init('attn_proj', cfg), dtype=cfg.dtype, rngs=rngs)
    self.key_proj = nnx.Linear(cfg.D, cfg.D, use_bias=False, kernel_init=fsdp_init('attn_proj', cfg), dtype=cfg.dtype, rngs=rngs)
    self.value_proj = nnx.Linear(cfg.D, cfg.D, use_bias=False, kernel_init=fsdp_init('attn_proj', cfg), dtype=cfg.dtype, rngs=rngs)
    self.output_proj = nnx.Linear(cfg.D, cfg.D, use_bias=False, kernel_init=fsdp_init('zero', cfg), dtype=cfg.dtype, rngs=rngs)
    
    # Layer normalization for query-key normalization
    self.q_norm = nnx.RMSNorm(self.head_dim, use_scale=False, dtype=cfg.dtype, rngs=rngs)
    self.k_norm = nnx.RMSNorm(self.head_dim, use_scale=False, dtype=cfg.dtype, rngs=rngs)

    self.attention = functools.partial(jax.nn.dot_product_attention, is_causal=True)

    self.rope = cfg.rope

  def __call__(self, x): # [B, S, D]
    B, S, D = x.shape
    H = self.num_heads
    
    q = self.query_proj(x) # [B, S, D]
    k = self.key_proj(x) # [B, S, D]
    v = self.value_proj(x) # [B, S, D]
    
    # Fused reshape and transpose
    q = q.reshape(B, S, H, -1) # [B, S, H, D/H]
    k = k.reshape(B, S, H, -1) # [B, S, H, D/H]
    v = v.reshape(B, S, H, -1) # [B, S, H, D/H]
    
    q = self.q_norm(q)
    k = self.k_norm(k)

    # position embedding
    if self.rope:
        position = jnp.arange(S)
        q = apply_rope(q, position[None])
        k = apply_rope(k, position[None])

    # attention
    out = self.attention(q, k, v) # [B, S, H, D/H]

    # output projection
    out = self.output_proj(out.reshape(B, S, -1))
    return out

  def compute_io(self, x):
    """Run attention and return (out, inputs) where inputs holds per-linear inputs.
    Keys: 'query_proj', 'key_proj', 'value_proj', 'output_proj'.
    Each value has shape [B, S, in_features].
    """
    B, S, D = x.shape
    H = self.num_heads
    q_in = x; k_in = x; v_in = x
    q = self.query_proj(q_in)
    k = self.key_proj(k_in)
    v = self.value_proj(v_in)
    q = q.reshape(B, S, H, -1)
    k = k.reshape(B, S, H, -1)
    v = v.reshape(B, S, H, -1)
    q = self.q_norm(q)
    k = self.k_norm(k)
    if self.rope:
        position = jnp.arange(S)
        q = apply_rope(q, position[None])
        k = apply_rope(k, position[None])
    attn_out = self.attention(q, k, v)
    out_proj_in = attn_out.reshape(B, S, -1)
    out = self.output_proj(out_proj_in)
    inputs = {
        'query_proj': q_in,
        'key_proj': k_in,
        'value_proj': v_in,
        'output_proj': out_proj_in,
    }
    return out, inputs


class TransformerBlock(nnx.Module):
  def __init__(self, cfg: DictConfig, rngs: nnx.Rngs):
    self.ln1 = nnx.RMSNorm(cfg.D, use_scale=False, dtype=cfg.dtype, rngs=rngs)
    # Use our custom multi-headed attention implementation
    self.attn = Attention(cfg, rngs)
    self.ln2 = nnx.RMSNorm(cfg.D, use_scale=False, dtype=cfg.dtype, rngs=rngs)
    self.mlp = Mlp(cfg, rngs)
    self.branch_multiplier = 1 / (cfg.N / cfg.base_N) if cfg.depth_mup else 1
    
  def __call__(self, x):  # [B, S, D]
    # Pre-layernorm attention block
    h = self.ln1(x)

    # Attention and residual connection
    x = x + self.attn(h) * self.branch_multiplier
    
    # Pre-layernorm MLP block
    return x + self.mlp(self.ln2(x)) * self.branch_multiplier


class Mlp(nnx.Module):
  """Multilayer perceptron."""
  def __init__(self, cfg: DictConfig, rngs: nnx.Rngs):
    self.fc1 = nnx.Linear(in_features=cfg.D, out_features=cfg.mlp_expansion*cfg.D, use_bias=False, kernel_init=fsdp_init('mlp_kernel', cfg), dtype=cfg.dtype, rngs=rngs)
    self.fc2 = nnx.Linear(in_features=cfg.mlp_expansion*cfg.D, out_features=cfg.D, use_bias=False, kernel_init=fsdp_init('zero', cfg), dtype=cfg.dtype, rngs=rngs)

    self.swiglu = cfg.swiglu
    if cfg.swiglu:
        self.fc3 = nnx.Linear(in_features=cfg.D, out_features=cfg.mlp_expansion*cfg.D, use_bias=False, kernel_init=fsdp_init('mlp_kernel', cfg), dtype=cfg.dtype, rngs=rngs)
    
  def __call__(self, x):  # [B, S, D]
    # SwiGLU
    if self.swiglu:
        h = jax.nn.swish(self.fc1(x)) * self.fc3(x)       # [B, S, F]
    else:
        h = jax.nn.gelu(self.fc1(x))
    return self.fc2(h)  # [B, S, D]

  def compute_io(self, x):
    """Run MLP and return (out, inputs) with inputs for fc1, fc2, and optionally fc3."""
    if self.swiglu:
        h1_in = x
        h1 = jax.nn.swish(self.fc1(h1_in)) * self.fc3(h1_in)
    else:
        h1_in = x
        h1 = jax.nn.gelu(self.fc1(h1_in))
    out = self.fc2(h1)
    inputs = {'fc1': h1_in, 'fc2': h1}
    if self.swiglu:
        inputs['fc3'] = h1_in
    return out, inputs


def fsdp_init(layer_type: str, cfg: DictConfig):
  """Initialize weights with optional FSDP partitioning."""
  partition_fn = nnx.with_partitioning if cfg.fsdp_enabled else lambda x, _: x
  kernel_init = jax.nn.initializers.normal(stddev=cfg.init_std_mult*jnp.sqrt(1.0/cfg.D))
  embed_init = jax.nn.initializers.normal(stddev=cfg.init_std_mult*cfg.embed_init_std)
  zero_init = jax.nn.initializers.zeros
  match layer_type:
    case "embedding":  # [V, D]
      return partition_fn(embed_init, (None, "data"))
    case "attn_proj":  # [D, D]
      return partition_fn(kernel_init, ("data", None))
    case "mlp_kernel":  # [D, F]
      return partition_fn(kernel_init, ("data", None))
    case "zero":  # [D, O]
      return partition_fn(zero_init, ("data", None))
    case _:
      raise ValueError(f"unrecognized layer type: {layer_type}")


def create_sharded_model(c: DictConfig, mesh: Mesh, seed: int):
  """
  initialize sharded model without putting it on a single device
  https://flax.readthedocs.io/en/latest/guides/flax_gspmd.html
  """

  @nnx.jit
  def initialize_sharded_model():
    model = TransformerDecoder(c, rngs=nnx.Rngs(seed)) # unsharded at this moment
    state = nnx.state(model) # the model's state, a pure pytree
    pspecs = nnx.get_partition_spec(state) # get annotations from state
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    nnx.update(model, sharded_state) # the model is sharded now
    return model

  with mesh:
    model = initialize_sharded_model()

  return model
