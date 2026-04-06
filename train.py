import os
import jax
import jax.numpy as jnp
jnp.set_printoptions(precision=2)
import numpy as np
import optax
import wandb
import data, utils
import model as transformer
import mlp
from flax import nnx
from tqdm.auto import tqdm
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from omegaconf.dictconfig import DictConfig
import pickle
from math import prod
from functools import partial
from optax_optim import scale_by_shampoo, scale_by_muon, scale_by_adamuon, scale_by_adam, scale_by_momentum, scale_by_soap
import soap_debug_recorder as sdr
import orbax.checkpoint as ocp
from utils import get_vm_region, bucket_exists, get_flop_per_token, device_hardware_flops, clean_gcs_path, update_metrics_tree, clean_param_path
import time

def random_subspace(key, M: int, k: int, dtype, use_qr: bool = True):
  """Sample a random Mxk subspace basis."""
  A = jax.random.normal(key, (M, k), dtype=dtype)
  if use_qr:
    Q, _ = jnp.linalg.qr(A, mode='reduced')
    return Q
  return A / (jnp.linalg.norm(A, axis=0, keepdims=True) + 1e-8)

@nnx.jit
def eval_features(features, prev_features):
  delta = features - prev_features
  return {'h': jnp.sqrt(jnp.mean(features**2)), 'dh': jnp.sqrt(jnp.mean(delta**2))}

@nnx.jit
def eval_features_and_logits(features, logits, prev_features, prev_logits):
  delta_features = features - prev_features
  delta_logits = logits - prev_logits
  return {'h': jnp.sqrt(jnp.mean(features**2)), 'dh': jnp.sqrt(jnp.mean(delta_features**2)),
          'f': jnp.sqrt(jnp.mean(logits**2)), 'df': jnp.sqrt(jnp.mean(delta_logits**2))}

@nnx.jit
def eval_embeddings(embeddings, prev_embeddings):
  delta_embeddings = embeddings - prev_embeddings
  return {'e': jnp.sqrt(jnp.mean(embeddings**2)), 'de': jnp.sqrt(jnp.mean(delta_embeddings**2))}

def train_and_evaluate(cfg: DictConfig):
  # set seed
  np.random.seed(cfg.seed)

  # datasets
  tokens_per_step = cfg.opt.B * cfg.model.L
  # ── gradient-accumulation parameters ─────────────────────────────
  if cfg.opt.B_max is None:
    B_micro = cfg.opt.B
  else:
    B_micro = min(cfg.opt.B, cfg.opt.B_max)                 # per-device batch
    assert cfg.opt.B % B_micro == 0, \
        "`cfg.opt.B` must be an integer multiple of `cfg.opt.B_max`"
  accum_steps = cfg.opt.B // B_micro                     # micro-batches / update

  B_eval = min(cfg.T_eval // cfg.model.L, B_micro); tokens_per_step_eval = B_eval * cfg.model.L
  get_batch_train, total_train_tokens = data.make_loader(cfg.ds_path, cfg.model.L, B_micro, cfg.shuffle_data, 'train')
  try:
    get_batch_test, total_test_tokens = data.make_loader(cfg.ds_path, cfg.model.L, B_eval, cfg.shuffle_data, 'test')
  except:
    get_batch_test, total_test_tokens = data.make_loader(cfg.ds_path, cfg.model.L, B_eval, cfg.shuffle_data, 'val')
  print(f"Total train tokens: {total_train_tokens/1e9:.2g}B")
  print(f"Total test tokens: {total_test_tokens/1e9:.2g}B")

  # Process optimizer-specific overrides via "|" in opt.name (delegated)
  cfg.opt.name = utils.parse_opt_name_overrides(cfg)

  # model
  mesh = jax.make_mesh((jax.device_count(),), ('data',), axis_types=(jax.sharding.AxisType.Auto))
  print(f"Enable fsdp: {cfg.model.fsdp_enabled}")
  
  # mup logic
  if cfg.opt.scale_eps is None:
    cfg.opt.scale_eps = cfg.opt.mup
    print(f'Setting scale_eps to {cfg.opt.scale_eps}')
  if cfg.opt.depth_mup is None:
    cfg.opt.depth_mup = cfg.opt.mup
    print(f'Setting depth_mup to {cfg.opt.depth_mup}')

  # Create the base model and extract its shapes
  cfg.model.depth_mup = cfg.opt.depth_mup # to determine branch multiplier
  if cfg.model.aspect_ratio is not None:
    if cfg.model.P is not None:
      # P = 12 * N * D^2 = 12 * N^3 * A^2
      params = cfg.model.P * 1e6
      M = cfg.model.mlp_expansion * (2 + float(cfg.model.swiglu)) + 4
      cfg.model.N = round((params / M / cfg.model.aspect_ratio**2) ** (1/3))
      cfg.model.D = cfg.model.N * cfg.model.aspect_ratio
      print(f'Setting N to {cfg.model.N} and D to {cfg.model.D} from P = {cfg.model.P}M')
    else:
      cfg.model.D = cfg.model.N * cfg.model.aspect_ratio
    print(f'Setting D to {cfg.model.N} * {cfg.model.aspect_ratio} = {cfg.model.D}')

  # Auto-enable gradient checkpointing if model width exceeds threshold
  if not cfg.model.gradient_checkpointing and cfg.model.D >= cfg.model.D_gckpt:
    cfg.model.gradient_checkpointing = True
    print(f"Auto-enabled gradient checkpointing for D = {cfg.model.D} >= {cfg.model.D_gckpt}")
  print(f"Enable gradient checkpointing: {cfg.model.gradient_checkpointing}")
  if cfg.arch == 'transformer':
    create_model_fn = transformer.create_sharded_model
  elif cfg.arch == 'mlp':
    create_model_fn = mlp.create_sharded_model
  else:
    raise ValueError(f'Unknown model type: {cfg.model.type}')
  base_model_cfg = cfg.model.copy(); base_model_cfg.D = cfg.model.base_D
  if cfg.model.base_mlp_expansion is not None:
    base_model_cfg.mlp_expansion = cfg.model.base_mlp_expansion
  base_model = create_model_fn(base_model_cfg, mesh, cfg.seed)
  base_shapes = jax.tree.map(lambda x: jnp.array(x.shape), nnx.state(base_model))
  del base_model
  model = create_model_fn(cfg.model, mesh, cfg.seed)
  model_graphdef = nnx.graphdef(model)
  shapes = jax.tree.map(lambda x: jnp.array(x.shape), nnx.state(model))
  # Clean parameter path names for readable logging (drop brackets/quotes/kernel/value)
  paths = jax.tree_util.tree_map_with_path(lambda path, p: clean_param_path(path), nnx.state(model))
  flat_names = jax.tree_util.tree_leaves(paths)
  num_params = sum(prod(p.shape) for p in jax.tree_util.tree_leaves(nnx.state(model)))
  non_embed_params = sum((prod(p.shape) if cfg.model.V not in p.shape else 0) for p in jax.tree_util.tree_leaves(nnx.state(model)))
  print(f'Number of non-embedding parameters: {non_embed_params/1e9:.3g} B')
  print(f'Number of parameters: {num_params/1e9:.3g} B')
  print(f"Size of parameters: {num_params*4/1e9:.3g} GB")
  checkpoints_size = 2*cfg.opt.B*cfg.model.L*cfg.model.D*cfg.model.N
  print(f"Size of layer-only activations: {checkpoints_size/1e9:.3g} GB")


  # possibly auto-set T from fitted scale/exponent
  assert cfg.T is not None or (cfg.scale is not None and cfg.exponent is not None) \
        or (cfg.token_per_param is not None), \
      "Either T or fitted scale/exponent must be provided."
  if cfg.token_per_param:
    cfg.T = round(cfg.token_per_param * num_params)
    print(f"Setting T to {cfg.T:.2g} from {cfg.token_per_param} toks/param")
  elif cfg.scale is not None and cfg.exponent is not None:
    C = 1e15 * ((num_params / cfg.scale) ** cfg.exponent)
    cfg.T = round(C / (6 * num_params))
    print(f"Setting T to {cfg.T} from scale and exponent")
  num_train_steps = cfg.T // tokens_per_step
  num_valid_steps = cfg.T_eval // tokens_per_step_eval
  print(f"Number of train steps: {num_train_steps} = {cfg.T} / {tokens_per_step}")
  print(f"Number of valid steps: {num_valid_steps} = {cfg.T_eval} / {tokens_per_step_eval}")

  if cfg.opt.warmup_tokens < 1:
    cfg.opt.warmup_tokens = round(cfg.opt.warmup_tokens * cfg.T)
  num_warmup_steps = cfg.opt.warmup_tokens // tokens_per_step
  print(f"Number of warmup steps: {num_warmup_steps} = {cfg.opt.warmup_tokens} / {tokens_per_step}")
  

  # Tie lr multiplier
  if cfg.opt.readout_lr_mult is None:
    cfg.opt.readout_lr_mult = cfg.opt.embed_lr_mult

  # convert half life to beta
  if cfg.opt.t1 is not None:
    cfg.opt.b1 = max(1 - tokens_per_step / cfg.opt.t1, 0)
  if cfg.opt.t2 is not None:
    cfg.opt.b2 = max(1 - tokens_per_step / cfg.opt.t2, 0)

  # convert normalized weight decay half life to wd
  if cfg.opt.wd_half_life is not None:
    if cfg.opt.wd_half_life < 0:
      cfg.opt.wd_half_life = 2**cfg.opt.wd_half_life
    # wd_half_life = 1/(wd * total_steps) => wd = 1 / (wd_half_life * total_steps)
    cfg.opt.weight_decay = 1 / (cfg.opt.wd_half_life * num_train_steps)
  elif cfg.opt.wdxD is not None:
    if cfg.opt.wdxD < 0:
      cfg.opt.wdxD = 2**cfg.opt.wdxD
    cfg.opt.weight_decay = cfg.opt.wdxD / cfg.model.D

  # Convert negative values to 2^x
  if cfg.opt.lr < 0:
    cfg.opt.lr = 2**cfg.opt.lr
  if cfg.opt.weight_decay < 0:
    cfg.opt.weight_decay = 2**cfg.opt.weight_decay

  if os.path.exists(f'{cfg.ds_path}/meta.pkl'):
    with open(f'{cfg.ds_path}/meta.pkl', 'rb') as f:
      meta = pickle.load(f)
      cfg.model.V = int(jnp.ceil(meta['vocab_size'] / 32)) * 32

  # start wandb
  assert cfg.wandb_project is not None
  config = utils.flatten_dict(cfg)
  config['num_params'] = num_params
  config['non_embed_params'] = non_embed_params
  config['hardware'] = jax.devices()[0].device_kind.lower()
  config['elr'] = (cfg.opt.lr * cfg.opt.weight_decay) ** 0.5
  run = wandb.init(project=cfg.wandb_project, config=config, mode=cfg.wandb_mode, tags=[cfg.wandb_tag], resume="allow")

  if cfg.base_model_checkpoint_path is not None and cfg.ckpt:
    region = get_vm_region().replace("-", "_") # e.g. us_central1
    ckpt_path = f'gs://{cfg.base_model_checkpoint_path}_{region}/{run.sweep_id if run.sweep_id else "no_sweep"}/{run.id}/'
    bucket_path = ckpt_path.split("/")[2]
    if not bucket_exists(bucket_path):
      raise ValueError(f"Bucket gs://{bucket_path} doesn't exist")
  else:
    print("Checkpoint path is None. Skip checkpointing.")
    ckpt_path = None

  # Mark preemptable process for sweep
  if run.sweep_id:
     run.mark_preempting()

  # Early termination
  if not cfg.opt.mup and cfg.opt.scale_eps:
      print("Running SP with eps scaling is pointless, so exiting...")
      wandb.finish()
      exit()


  data_sharding = NamedSharding(mesh, P('data'))
  with mesh: ds_valid = jnp.stack([jax.device_put(next(get_batch_test), data_sharding) for i in range(num_valid_steps)])
  
  # ------------------------------------------------------------------------
  # 1. tag every parameter with the optimiser name you want
  do_spectral_norm = cfg.opt.name.startswith('spectral-')
  if do_spectral_norm:
    cfg.opt.name = cfg.opt.name.replace('spectral-', '')
  def assign_optimizer(p, path_str):
      opt = cfg.opt.name
      if opt.endswith('_adam'):
          opt = 'adam' if ('embed' in path_str or 'readout' in path_str) \
                        else opt.replace('_adam', '')
      elif opt.endswith('_adamr'):
          opt = 'adam' if 'readout' in path_str \
                        else opt.replace('_adamr', '')
      elif opt == 'gn_subspace_adamw':
          opt = 'adam'
      return opt

  optimizers = jax.tree.map(assign_optimizer, nnx.state(model), paths)

  # ------------------------------------------------------------------------
  # 2. learning-rate per-parameter (µP-aware)
  N_scale = cfg.model.N / cfg.model.base_N # depth scaling factor
  # true if not embedding or readout
  is_bulk = jax.tree.map(lambda x: not ('embed' in x or 'readout' in x), paths)
  N_scale = jax.tree.map(lambda x: N_scale if x else 1, is_bulk)

  # Precompute normalized block counts per-parameter for use in LR and optimizers
  def compute_nb(base_shape, shape):
      base_din, base_dout = base_shape
      din, dout = shape
      # one block if exceeding max precond dim
      base_nb_in = jnp.ceil(base_din / cfg.opt.block_size) if base_din <= cfg.opt.max_precond_dim else 1
      base_nb_out = jnp.ceil(base_dout / cfg.opt.block_size) if base_dout <= cfg.opt.max_precond_dim else 1
      nb_in = jnp.ceil(din / cfg.opt.block_size) if din <= cfg.opt.max_precond_dim else 1
      nb_out = jnp.ceil(dout / cfg.opt.block_size) if dout <= cfg.opt.max_precond_dim else 1
      # normalize relative to base
      return nb_in / base_nb_in, nb_out / base_nb_out

  nb_trees = jax.tree.map(compute_nb, base_shapes, shapes)
  is_pair = lambda x: isinstance(x, tuple)
  nb_in_tree  = jax.tree_util.tree_map(lambda t: t[0], nb_trees, is_leaf=is_pair)
  nb_out_tree = jax.tree_util.tree_map(lambda t: t[1], nb_trees, is_leaf=is_pair)

  def assign_lr(base_shape, shape, N_scale, nb_in, nb_out, opt_name, path_str):
      base_lr                   = cfg.opt.lr

      if   'embed'   in path_str: 
        base_lr *= cfg.opt.embed_lr_mult
      elif 'readout' in path_str: 
        base_lr *= cfg.opt.readout_lr_mult

      if not cfg.opt.mup:
        return base_lr

      if do_spectral_norm:
        # Normalization ensures Theta(1) updates, so no need to scale LR
        return base_lr

      base_din, base_dout       = base_shape
      din, dout                 = shape
      
      base_in_active = base_din <= cfg.opt.max_precond_dim
      base_out_active = base_dout <= cfg.opt.max_precond_dim
      in_active = din <= cfg.opt.max_precond_dim
      assert in_active == base_in_active, "Precondition input active must be the same as base"
      out_active = dout <= cfg.opt.max_precond_dim
      assert out_active == base_out_active, "Precondition output active must be the same as base"
      assert in_active or out_active, "At least one side must be active"

      # normalize realtive to base
      din, dout                 = din / base_din, dout / base_dout

      if not cfg.opt.depth_mup:
        N = 1
      else:
        N = N_scale

      if opt_name in ['adam', 'adamuon', 'adamuon_rms_align', 'muon_adam']: 
        mult = 1 / din
      elif opt_name == 'gn_subspace_adamw':
        mult = 1 / din
      elif opt_name == 'muon': 
        mult = (dout / din) ** 0.5
      elif opt_name in ['shampoo', 'shampoo2']:
        k = 1/4 if opt_name == 'shampoo' else 1/2
        if in_active and out_active:
          E = 2*k
        else:
          E = k
        mult =  N ** (1 - 2*E) * (dout / din) ** (1 - E) * (nb_in * nb_out) ** -E
      elif opt_name == 'soap':
        if in_active and out_active:
          mult = (dout / din) ** 0.5 * (nb_in * nb_out) ** -0.5
        elif in_active:
          mult = (din * nb_in) ** -0.5
        else:
          mult = (dout / nb_out) ** 0.5 / din
      elif opt_name.startswith('grafted_shampoo'):
        mult = 1 / din
      elif opt_name == "sgd": 
        mult = dout / din * N
      else: raise ValueError(f'Unknown optimiser {opt_name}')

      return base_lr * mult

  lrs = jax.tree.map(assign_lr, base_shapes, shapes, N_scale, nb_in_tree, nb_out_tree, optimizers, paths)

  # ------------------------------------------------------------------------
  # 3. build optimiser transforms lazily so unused ones never compile
  B_scale = cfg.opt.B / cfg.opt.base_B

  factory = {
      'adam':   lambda: scale_by_adam(
          b1=cfg.opt.b1, b2=cfg.opt.b2, eps=cfg.opt.eps,
          scale_eps=cfg.opt.scale_eps, base_shapes=base_shapes, B=B_scale, N=N_scale),
      'muon':   lambda: scale_by_muon(
          beta=cfg.opt.b1, eps=cfg.opt.eps, nesterov=True,
          scale_eps=cfg.opt.scale_eps, base_shapes=base_shapes, B=B_scale, N=N_scale),
      'adamuon': lambda: scale_by_adamuon(
          beta1=cfg.opt.b1, beta2=cfg.opt.b2, eps=cfg.opt.eps, adam_eps=cfg.opt.eps,
          scale_eps=cfg.opt.scale_eps, base_shapes=base_shapes, B=B_scale, N=N_scale, rms_align=False),
      'adamuon_rms_align': lambda: scale_by_adamuon(
          beta1=cfg.opt.b1, beta2=cfg.opt.b2, eps=cfg.opt.eps, adam_eps=cfg.opt.eps,
          scale_eps=cfg.opt.scale_eps, base_shapes=base_shapes, B=B_scale, N=N_scale, rms_align=True),
      'shampoo': lambda: scale_by_shampoo(
          b1=cfg.opt.b1, b2=cfg.opt.b2, adam_eps=cfg.opt.eps, matrix_eps=cfg.opt.matrx_eps,
          freq=cfg.opt.freq, scale_eps=cfg.opt.scale_eps,
          base_shapes=base_shapes, B=B_scale, N=N_scale, nb_in=nb_in_tree, nb_out=nb_out_tree, eigh=cfg.opt.eigh, rel_eps=cfg.opt.rel_eps, block_size=cfg.opt.block_size, max_precond_dim=cfg.opt.max_precond_dim),
      'shampoo2': lambda: scale_by_shampoo(
          b1=cfg.opt.b1, b2=cfg.opt.b2, adam_eps=cfg.opt.eps, matrix_eps=cfg.opt.matrx_eps,
          freq=cfg.opt.freq, kl=0.5, kr=0.5,
          scale_eps=cfg.opt.scale_eps, base_shapes=base_shapes, B=B_scale, N=N_scale, nb_in=nb_in_tree, nb_out=nb_out_tree, eigh=cfg.opt.eigh, rel_eps=cfg.opt.rel_eps, block_size=cfg.opt.block_size, max_precond_dim=cfg.opt.max_precond_dim),
      'grafted_shampoo': lambda: scale_by_shampoo(
          b1=cfg.opt.b1, b2=cfg.opt.b2, adam_eps=cfg.opt.eps, matrix_eps=cfg.opt.matrx_eps,
          freq=cfg.opt.freq, grafting=True,
          scale_eps=cfg.opt.scale_eps, base_shapes=base_shapes, B=B_scale, N=N_scale, nb_in=nb_in_tree, nb_out=nb_out_tree, eigh=cfg.opt.eigh, rel_eps=cfg.opt.rel_eps, block_size=cfg.opt.block_size, max_precond_dim=cfg.opt.max_precond_dim),
      'grafted_shampoo2': lambda: scale_by_shampoo(
          b1=cfg.opt.b1, b2=cfg.opt.b2, adam_eps=cfg.opt.eps, matrix_eps=cfg.opt.matrx_eps,
          freq=cfg.opt.freq, kl=0.5, kr=0.5, grafting=True,
          scale_eps=cfg.opt.scale_eps, base_shapes=base_shapes, B=B_scale, N=N_scale, nb_in=nb_in_tree, nb_out=nb_out_tree, eigh=cfg.opt.eigh, rel_eps=cfg.opt.rel_eps, block_size=cfg.opt.block_size, max_precond_dim=cfg.opt.max_precond_dim),
      'sgd': lambda: scale_by_momentum(b1=cfg.opt.b1),
      # bucketed variants: identical math, faster compile
      'soap': lambda: scale_by_soap(b1=cfg.opt.b1, b2=cfg.opt.b2, adam_eps=cfg.opt.eps, matrix_eps=cfg.opt.matrx_eps, freq=cfg.opt.freq, scale_eps=cfg.opt.scale_eps, base_shapes=base_shapes, B=B_scale, N=N_scale, nb_in=nb_in_tree, nb_out=nb_out_tree, eigh=cfg.opt.eigh, eigh_warmup_steps=cfg.opt.eigh_warmup_steps, align=True, rel_eps=cfg.opt.rel_eps, block_size=cfg.opt.block_size, max_precond_dim=cfg.opt.max_precond_dim, bf16_momentum=cfg.bf16_momentum),
  }

  use_gn_subspace_adamw = cfg.opt.name == 'gn_subspace_adamw'
  if use_gn_subspace_adamw and accum_steps != 1:
      raise ValueError("gn_subspace_adamw currently requires accum_steps == 1 (set opt.B_max to opt.B).")

  labels_in_use   = frozenset(jax.tree_util.tree_leaves(optimizers))
  transforms_dict = {name: make() for name, make in factory.items() if name in labels_in_use and name in factory}

  # single-label fast-path (skips optax.multi_transform completely)
  if len(labels_in_use) == 1:
      scale_by_optim = transforms_dict[next(iter(labels_in_use))]
  else:
      scale_by_optim = optax.multi_transform(
          transforms=transforms_dict,
          param_labels=optimizers
      )

  # ------------------------------------------------------------------------
  # 4. LR scaling & full optimiser
  scale_by_lr = optax.GradientTransformation(
      init   = lambda _: None,
      update = lambda upd, state, _: (jax.tree.map(
          lambda g, lr: g * lr, upd, lrs), state)
  )

  # Optional normalization by per-parameter spectral norm of the update.
  maybe_norm_by_spect = (
      utils.norm_by_spect(cfg.num_power_iter, seed=cfg.seed, emb_norm=cfg.opt.emb_norm, hidden_norm=cfg.opt.hidden_norm, readout_norm=cfg.opt.readout_norm)
      if cfg.num_power_iter > 0 and do_spectral_norm else optax.identity()
  )

  preconditioner = optax.chain(
    optax.clip_by_global_norm(cfg.opt.clip) 
        if cfg.opt.clip > 0 else optax.identity(),
    scale_by_optim,
    maybe_norm_by_spect,  # normalize update spectral norm before LR
    scale_by_lr,
    optax.transforms.add_decayed_weights(cfg.opt.weight_decay), # independent wd
  )

  schedule_fn = utils.get_scheduler(
      cfg.opt.schedule,
      cfg.opt.decay_frac,
      cfg.opt.warmup_tokens // tokens_per_step,
      num_train_steps
  )

  # Optional SOAP projected-tensor recorder
  soap_save_path = os.environ.get('SOAP_SAVE_PATH')
  if soap_save_path:
    sdr.init(flat_names, soap_save_path)

  if not use_gn_subspace_adamw:
    tx = optax.chain(
        preconditioner,
        optax.scale_by_schedule(schedule_fn),
        optax.scale(-1.0)
    )

    with mesh:
      optimizer = nnx.Optimizer(model, tx)
    opt_graphdef, opt_state = nnx.split(optimizer)
  else:
    opt_comp = optax.adamw(
        learning_rate=1.0,
        b1=cfg.opt.b1,
        b2=cfg.opt.b2,
        eps=cfg.opt.eps,
        weight_decay=cfg.opt.weight_decay,
    )
    opt_state_comp = opt_comp.init(nnx.state(model))
    params = nnx.state(model)
    flat_params, _ = jax.flatten_util.ravel_pytree(params)
    subspace_key, p_key = jax.random.split(jax.random.PRNGKey(cfg.seed))
    gn_basis = random_subspace(
        p_key,
        flat_params.shape[0],
        int(cfg.opt.gn_subspace_dim),
        flat_params.dtype,
        bool(cfg.opt.gn_use_qr),
    )


  del base_shapes, shapes

  get_in_out = partial(data.get_in_out, task='regression' if cfg.ds_path == 'fourier' else 'ntp')
  loss_fn = optax.l2_loss if cfg.ds_path == 'fourier' else optax.softmax_cross_entropy_with_integer_labels

  @nnx.jit
  def batch_loss_fn(model, batch):
    x, y, weights = get_in_out(batch)
    logits = model(x).astype(jnp.float32) # float32 for stability
    losses = loss_fn(logits, y).mean()
    mean_loss = jnp.sum(losses * weights) / (weights.sum() + 1e-6)
    return mean_loss

  def _micro_loss_and_grad(model, batch):
      return nnx.value_and_grad(batch_loss_fn)(model, batch)

  @partial(jax.jit, donate_argnames=('opt_state'), static_argnames=('track_update',))
  def train_step(opt_state, micro_batches, track_update: bool=False): #  micro_batches = list/tuple of length accum_steps   
    loss_sum, grads_sum = 0.0, None
    for batch in micro_batches:                    # python loop unrolled by jit, OK because accum_steps is small
        model = nnx.merge(model_graphdef, opt_state.model)
        loss, grads = _micro_loss_and_grad(model, batch)
        loss_sum += loss
        grads_sum = grads if grads_sum is None else jax.tree.map(jnp.add, grads_sum, grads)
    loss = loss_sum / accum_steps

    grads_mean = jax.tree.map(lambda g: g / accum_steps, grads_sum)

    optimizer = nnx.merge(opt_graphdef, opt_state)
    old_params = opt_state.model if track_update else None
    optimizer.update(grads_mean)
    opt_state = nnx.state(optimizer)
    if track_update:
      last_update = jax.tree.map(lambda new, old: new - old, opt_state.model, old_params)
      return opt_state, loss, last_update
    else:
      return opt_state, loss

  @nnx.jit
  def train_step_gn_subspace(params, opt_state_comp, batch, basis, lr_t, damping):
    x, y, weights = get_in_out(batch)

    def _residual_and_loss(p):
      model_local = nnx.merge(model_graphdef, p)
      logits = model_local(x).astype(jnp.float32)
      if cfg.ds_path == 'fourier':
        residual = (logits - y).reshape(-1)
        losses = loss_fn(logits, y).mean()
        mean_loss = jnp.mean(losses)
      else:
        onehot = jax.nn.one_hot(y.astype(jnp.int32), logits.shape[-1])
        probs = jax.nn.softmax(logits, axis=-1)
        residual = ((probs - onehot) * weights[..., None]).reshape(-1)
        losses = loss_fn(logits, y)
        mean_loss = jnp.sum(losses * weights) / (weights.sum() + 1e-6)
      return mean_loss, residual

    (loss, residual), grad_tree = jax.value_and_grad(_residual_and_loss, has_aux=True)(params)
    grad_vec, unravel = jax.flatten_util.ravel_pytree(grad_tree)
    flat_params, _ = jax.flatten_util.ravel_pytree(params)
    k = basis.shape[1]

    def jvp_col(v_col):
      v_tree = unravel(v_col)
      _, out = jax.jvp(
          lambda p: nnx.merge(model_graphdef, p)(x).astype(jnp.float32),
          (params,),
          (v_tree,),
      )
      return out.reshape(-1)

    jp = jax.vmap(jvp_col, in_axes=1, out_axes=1)(basis)
    bsz = x.shape[0]
    h_red = jp.T @ jp / bsz + damping * jnp.eye(k, dtype=jp.dtype)
    g_red = jp.T @ residual / bsz
    delta_u = -jnp.linalg.solve(h_red, g_red)
    delta_theta = basis @ delta_u

    complement_grad = grad_vec - basis @ (basis.T @ grad_vec)
    complement_grad_tree = unravel(complement_grad)
    updates, opt_state_comp = opt_comp.update(complement_grad_tree, opt_state_comp, params)
    adam_update_vec, _ = jax.flatten_util.ravel_pytree(updates)
    new_flat = flat_params + lr_t * delta_theta + adam_update_vec
    new_params = unravel(new_flat)
    return new_params, opt_state_comp, loss
  

  def loss_and_grad_fn(param, batch):
    model = nnx.merge(model_graphdef, param)
    return nnx.value_and_grad(batch_loss_fn)(model, batch)
  
  @nnx.jit
  def eval_step(params, dataset):
      def _body(total_loss, batch):
          loss, _ = loss_and_grad_fn(params, batch)
          return total_loss + loss, None
      total_loss, _ = jax.lax.scan(_body, 0.0, dataset)
      param_norm = jnp.sqrt(sum([jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params)]))
      return {"eval_loss": total_loss / len(dataset), "param_norm": param_norm}

  # training loop
  pending_train_metrics = None
  tau = 0
  prev_features = None
  prev_logits = None
  prev_embeddings = None
  last_update = None
  
  eval_steps = set(
      [int(i) for i in jnp.linspace(0, num_train_steps-1, cfg.num_evals)] + 
      ([] if cfg.no_geom_eval_step else [int(i) for i in jnp.geomspace(1, num_train_steps-1, 20)])
  )
  eval_steps.add(0)
  eval_steps.add(num_train_steps-1)

  # Creates checkpoint manager
  if use_gn_subspace_adamw and ckpt_path is not None:
    print("Checkpointing is disabled for gn_subspace_adamw.")
    ckpt_path = None

  if ckpt_path is not None:
    mngr = ocp.CheckpointManager(
        ckpt_path,
        # ocp.AsyncCheckpointer(ocp.CompositeCheckpointHandler("opt_state", "prev")),
        options=ocp.CheckpointManagerOptions(save_interval_steps=300, max_to_keep=1)
    )
  else:
    mngr = None

  start_step = -1

  # Loading from checkpoint
  if (not use_gn_subspace_adamw) and mngr is not None and mngr.latest_step() is not None:
    start_step = mngr.latest_step()
    print(f"Resume training from step {start_step}")
    # Filter abstract type to avoid explicit single device sharding
    def filter_single_device_sharding(spec):
        return jax.ShapeDtypeStruct(
            spec.shape, spec.dtype, 
            sharding=spec.sharding if not isinstance(spec.sharding, jax.sharding.SingleDeviceSharding) else None
            , weak_type=spec.weak_type
        )
    abstract_state = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, opt_state)
    abstract_state = jax.tree.map(lambda x: filter_single_device_sharding(x), abstract_state)
    try:
      ckpt_state = mngr.restore(start_step, args=ocp.args.Composite(
          opt_state=ocp.args.StandardRestore(abstract_state),
          prev_features=ocp.args.ArrayRestore(),
          prev_logits=ocp.args.ArrayRestore(),
          prev_embeddings=ocp.args.ArrayRestore(),
      ))
      opt_state = ckpt_state['opt_state']
      prev_features = ckpt_state['prev_features']
      prev_logits = ckpt_state['prev_logits']
      prev_embeddings = ckpt_state['prev_embeddings']
    except Exception:
      # Backward-compat for older checkpoints without prev_embeddings
      ckpt_state = mngr.restore(start_step, args=ocp.args.Composite(
          opt_state=ocp.args.StandardRestore(abstract_state),
          prev_features=ocp.args.ArrayRestore(),
          prev_logits=ocp.args.ArrayRestore(),
      ))
      opt_state = ckpt_state['opt_state']
      prev_features = ckpt_state['prev_features']
      prev_logits = ckpt_state['prev_logits']
      prev_embeddings = None

  pbar = tqdm(range(num_train_steps + 1)) # Add one more step for evaluating the final model

  last_time = None
  last_eval_step = None
  flop_per_token = get_flop_per_token(cfg.model)
  device_count = jax.device_count()
  device = jax.devices()[0]
  theoretical_flops = device_hardware_flops(device) * device_count


  with mesh:
    for step in pbar:
      # Fast forward to checkpoint
      if step <= start_step:
        next(get_batch_train)
        tau += cfg.opt.lr * schedule_fn(step)
        continue

      lr_t = cfg.opt.lr * schedule_fn(step)
      tokens_seen = step*tokens_per_step
      compute_spent = 3 * tokens_seen * flop_per_token
      non_embed_compute = 6 * tokens_seen * non_embed_params

      # eval step at linearly spaced intervals
      if step in eval_steps:
        if last_eval_step is not None:
          elapsed_time = time.perf_counter() - last_time
          elapsed_step = step - last_eval_step
          consumed_tokens = elapsed_step * cfg.opt.B*cfg.model.L
          throughput = consumed_tokens / elapsed_time
          achieved_flops = 3 * consumed_tokens * flop_per_token
          flops_per_sec = achieved_flops / elapsed_time
          mfu = flops_per_sec / theoretical_flops

        current_params = params if use_gn_subspace_adamw else opt_state.model
        wandb_eval_metrics = eval_step(current_params, ds_valid)
        merged_model = nnx.merge(model_graphdef, current_params)
        x_eval = get_in_out(ds_valid[0])[0]
        features, logits = merged_model.get_features_and_logits(x_eval)
        embeddings = merged_model.get_embedding(x_eval)
        # Per-layer input activations for alignment metrics
        layer_inputs = merged_model.get_layer_inputs(x_eval)
        if prev_features is not None and prev_logits is not None:
          pending_feature_metrics = eval_features_and_logits(features, logits, prev_features, prev_logits)
        else:
          pending_feature_metrics = {'h': jnp.sqrt(jnp.mean(features**2)), 'f': jnp.sqrt(jnp.mean(logits**2))}
        wandb_eval_metrics |= pending_feature_metrics
        # Embedding metrics (rms and update rms)
        if prev_embeddings is not None:
          wandb_eval_metrics |= eval_embeddings(embeddings, prev_embeddings)
        else:
          wandb_eval_metrics |= {'e': jnp.sqrt(jnp.mean(embeddings**2))}

        # Track per-parameter update metrics for the last update only
        if cfg.track_update_size and last_update is not None:
          # Map per-parameter paths to their input activations for alignment.
          flat_paths = jax.tree_util.tree_leaves(paths)
          _leaves, treedef = jax.tree_util.tree_flatten(paths)
          flat_inputs = [layer_inputs.get(name, None) for name in flat_paths]
          inputs_tree = jax.tree_util.tree_unflatten(treedef, flat_inputs)
          rms_tree, spec_tree, sr_tree, align_tree = update_metrics_tree(last_update, inputs=inputs_tree)
          flat_paths = jax.tree_util.tree_leaves(paths)
          flat_rms = jax.tree_util.tree_leaves(rms_tree)
          flat_spec = jax.tree_util.tree_leaves(spec_tree)
          flat_sr = jax.tree_util.tree_leaves(sr_tree)
          flat_align = jax.tree_util.tree_leaves(align_tree)
          for name, v in zip(flat_paths, flat_rms):
            wandb_eval_metrics[f'upd_rms/{name}'] = v
          for name, v in zip(flat_paths, flat_spec):
            wandb_eval_metrics[f'upd_spec/{name}'] = v
          for name, v in zip(flat_paths, flat_sr):
            wandb_eval_metrics[f'upd_srank/{name}'] = v
          for name, v in zip(flat_paths, flat_align):
            wandb_eval_metrics[f'upd_align/{name}'] = v

        # only log if average over more than 100 steps
        if last_eval_step is not None and elapsed_step > 100:
            wandb_eval_metrics |= {"throughput": throughput, "flops_per_sec": flops_per_sec, "mfu": mfu}

        prev_features = features
        prev_logits = logits
        prev_embeddings = embeddings
        wandb_eval_metrics |= {'step': step, 'tokens': tokens_seen, 'compute': compute_spent, 'non_embed_compute': non_embed_compute, 'tau': tau, 'lr': lr_t, 'progress': step / num_train_steps}
        wandb.log(wandb_eval_metrics)

        # Start computing throughput after the first 10 steps
        if step > start_step + 10:
            last_time = time.perf_counter()
            last_eval_step = step

      # training step
      micro_batches = [jax.device_put(next(get_batch_train), data_sharding)
                      for _ in range(accum_steps)]
      if use_gn_subspace_adamw:
        params, opt_state_comp, loss = train_step_gn_subspace(
            params,
            opt_state_comp,
            micro_batches[0],
            gn_basis,
            lr_t,
            cfg.opt.gn_damping,
        )
      elif cfg.track_update_size:
        opt_state, loss, last_update = train_step(opt_state, micro_batches, track_update=True)
      else:
        opt_state, loss = train_step(opt_state, micro_batches, track_update=False)
      pbar.set_postfix_str(f'L={loss:.2f}')

      # Checkpointing
      if (not use_gn_subspace_adamw) and mngr is not None:
        mngr.save(step, args=ocp.args.Composite(
            opt_state=ocp.args.StandardSave(opt_state),
            prev_features=ocp.args.ArraySave(prev_features),
            prev_logits=ocp.args.ArraySave(prev_logits),
            prev_embeddings=ocp.args.ArraySave(prev_embeddings),
        ))
      elif use_gn_subspace_adamw and cfg.opt.gn_refresh_every > 0 and (step + 1) % cfg.opt.gn_refresh_every == 0:
        subspace_key, p_key = jax.random.split(subspace_key)
        flat_params, _ = jax.flatten_util.ravel_pytree(params)
        gn_basis = random_subspace(
            p_key,
            flat_params.shape[0],
            int(cfg.opt.gn_subspace_dim),
            flat_params.dtype,
            bool(cfg.opt.gn_use_qr),
        )

      tau += lr_t
    # wandb.log(pending_train_metrics)
    wandb.finish()
    # Save SOAP debug tensors if enabled
    if sdr.is_active():
        out_path = sdr.finalize()
        if out_path:
            print(f"Saved SOAP debug tensors to {out_path}")
    if mngr is not None:
        mngr.close()
        clean_gcs_path(ckpt_path)

    
