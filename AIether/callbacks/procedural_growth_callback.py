import torch
import numpy as np
import gc
from collections import deque
from transformers.trainer_callback import TrainerCallback
from aiether.managers import GrowthManager, LayerStateManager
from aiether.utils import GeometricExtrapolator, logger_setup

logger = logger_setup()

class ProceduralGrowthCallback(TrainerCallback):
    """
    Callback compatible with Hugging Face Trainer that implements procedural layer growth
    based on tensor-wise granular history analysis.
    """
    
    def __init__(
        self, 
        patience: int, 
        threshold: float, 
        strategy='GeometricExtrapolator', 
        output_dir=None,
        warmup_steps: int = 500,
        history_window_size: int = 20,
        collect_interval_steps: int = 50,
        growth_params: dict = None,
        tau_crit: float = None,
        kappa_crit: float = None,
        **kwargs # Absorbs legacy arguments (dim, dropout) to avoid init errors
    ):
        super().__init__()
        self.patience = patience
        self.threshold = threshold # Reference for tau_trigger (if used dynamically)
        self.strategy = strategy
        self.warmup_steps = warmup_steps
        self.output_dir = output_dir
        
        # History configuration
        self.history_window_size = history_window_size
        self.collect_interval_steps = collect_interval_steps
        
        # Storage structure: { layer_id: { param_name: deque } }
        self.layer_histories = {}  
        
        # Internal state
        self.stagnation_counter = 0
        self.trainer = None
        self.in_warmup = False
        self.warmup_start_step = 0
        self.growth_complete_notified = False
        
        # Stagnation triggers (suggested empirical values)
        self.tau_trigger = tau_crit if tau_crit is not None else 0.0
        self.kappa_trigger = kappa_crit if kappa_crit is not None else 0.0
        
        # Internal instantiation
        self.initializer = GrowthManager(**(growth_params or {}))
        self.layer_manager = LayerStateManager(output_dir=output_dir)
        
        logger.info(f"✅ ProceduralGrowthCallback initialized")
        logger.info(f"   Window={history_window_size}, Interval={collect_interval_steps}")

    def attach_trainer(self, trainer):
        self.trainer = trainer

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        # Register L0 from first active layer
        if model is not None and len(model.transformer.h) > 0:
            if hasattr(model, 'active_layers') and model.active_layers:
                first_active = min(model.active_layers)
            else:
                first_active = len(model.transformer.h) - 1

            self.layer_manager.register_layer(
                layer_id=first_active, strategy='initial', generation_step=0
            )
            self.layer_manager.save_state(
                layer_id=first_active,
                state_name="L0",
                state_dict=model.transformer.h[first_active].state_dict(),
                step=0,
                strategy='initial'
            )
            logger.info(f"📝 Initial state L_0 registered for layer {first_active}")
        return control
    
    def _get_frontier_layer(self, model):
        """
        Returns the frontier layer ID, defined as the active layer with minimum index.
        """
        return min(model.active_layers) 

    def _can_grow_more(self, model) -> bool:
        """Checks if the model has additional layers available for activation. Returns False when frontier reaches index 0."""
        if model is None or not hasattr(model, 'active_layers'):
            return False

        if not hasattr(model, 'transformer') or not hasattr(model.transformer, 'h'):
            return False

        all_layers_active = len(model.active_layers) >= len(model.transformer.h)
        frontier_layer = self._get_frontier_layer(model)

        if all_layers_active and frontier_layer == 0:
            if not self.growth_complete_notified:
                logger.info("🧱 All layers already created; snapshots and growth disabled.")
                self.growth_complete_notified = True
            return False

        return True
    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        """
        Performs warmup monitoring and periodic parameter history collection
        from frontier layer according to configured interval.
        """
        # Heartbeat log every 500 steps to detect hangs
        if state.global_step % 500 == 0:
            logger.info(f"💓 Heartbeat: Step {state.global_step} | Warmup={self.in_warmup}")
        
        # Warmup logic
        if self.in_warmup:
            steps_since = state.global_step - self.warmup_start_step
            
            # LOG: Warmup progress every 100 steps
            if steps_since % 100 == 0:
                logger.info(f"🔥 Warmup in progress: {steps_since}/{self.warmup_steps} steps (Layers stabilizing)")

            if steps_since >= self.warmup_steps:
                self.unfreeze_all_layers(model)
                self.in_warmup = False

        # Data collection logic (Snapshot)
        if state.global_step % self.collect_interval_steps == 0 and self._can_grow_more(model):
            frontier_layer = self._get_frontier_layer(model)
            logger.info(f"📸 Attempting snapshot capture at step {state.global_step}...")
            self._capture_layer_snapshot(model, frontier_layer)
        
        return control

    def _capture_layer_snapshot(self, model, layer_id):
        """Stores a copy of specified layer parameters in temporal history."""
        if layer_id >= len(model.transformer.h): return

        logger.info(f"📥 Capturing snapshot from layer {layer_id} to history")
        
        try:
            # CRITICAL: Synchronize CUDA before CPU transfer to prevent deadlock
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            layer = model.transformer.h[layer_id]
            if layer_id not in self.layer_histories:
                self.layer_histories[layer_id] = {}

            for name, param in layer.named_parameters():
                if not param.requires_grad: continue
                
                if name not in self.layer_histories[layer_id]:
                    self.layer_histories[layer_id][name] = deque(maxlen=self.history_window_size)
                
                # Save in CPU numpy format with safety checks
                param_cpu = param.detach().cpu().float()
                param_numpy = param_cpu.numpy()
                self.layer_histories[layer_id][name].append(param_numpy)
            
            logger.info(f"✅ Snapshot captured successfully from layer {layer_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to capture snapshot from layer {layer_id}: {e}")
            # Don't crash training, just skip this snapshot
            return

    def _freeze_layer(self, model, layer_id):
        """Sets requires_grad to False for all parameters in specified layer."""
        for param in model.transformer.h[layer_id].parameters():
            param.requires_grad = False
        logger.infoB(f"   🔒 Layer {layer_id} frozen")

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        """
        Computes stagnation metrics based on frontier layer parameter averages
        and triggers growth when criteria are met.
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"📊 EVALUATION CALLBACK - Step {state.global_step}")
        logger.info(f"{'='*70}\n")
        
        if model is None or not hasattr(model, 'active_layers') or not model.active_layers:
            logger.warning("⚠️ No active layers to evaluate")
            return control

        if not self._can_grow_more(model):
            logger.info("ℹ️ Cannot grow more layers, skipping growth check")
            return control

        frontier_layer = min(model.active_layers)
        logger.info(f"🎯 Frontier layer: {frontier_layer}")

        # Check history availability
        if frontier_layer not in self.layer_histories:
            return control
        
        histories = self.layer_histories[frontier_layer]
        if not histories: return control
        
        # Check if window is full enough (min 3 snapshots for kappa)
        first_param_name = next(iter(histories))
        if len(histories[first_param_name]) < 3:
            logger.info(f"📏 Insufficient history for analysis ({len(histories[first_param_name])} steps)")
            return control

        logger.info(f"🔍 Starting metric computation for layer {frontier_layer}...")
        
        try:
            # Calculate layer average metrics
            geom = GeometricExtrapolator()
            total_tau = 0.0
            total_kappa = 0.0
            count = 0
            
            # Calculate metrics for each layer parameter
            for name, deque_hist in histories.items():
                logger.info(f"   📊 Processing {name}...")
                
                # CRITICAL: Stack operation can hang - add protection
                try:
                    W_hist = np.stack(list(deque_hist), axis=0)
                    logger.info(f"   ✅ Stacked {name}: shape={W_hist.shape}")
                except Exception as e:
                    logger.error(f"   ❌ Failed to stack {name}: {e}")
                    continue
                
                # Compute metrics with safety check
                try:
                    m = geom.compute_metrics(W_hist) # Returns {'tau': ..., 'kappa_eff': ...}
                    total_tau += m['tau']
                    total_kappa += m['kappa_eff']
                    count += 1
                    logger.info(f"   ✅ Metrics computed for {name}")
                except Exception as e:
                    logger.error(f"   ❌ Failed to compute metrics for {name}: {e}")
                    continue
            
            if count == 0:
                logger.warning("⚠️ No metrics computed successfully")
                return control
            
            logger.info(f"✅ Metric computation completed ({count} params processed)")
            
        except Exception as e:
            logger.error(f"❌ Critical error in metric computation: {e}")
            return control

        avg_tau = total_tau / count
        avg_kappa = total_kappa / count
        
        logger.info(f"📐 Aggregated Metrics (Layer {frontier_layer}): tau={avg_tau:.4f}, kappa={avg_kappa:.4f}")

        # Check stagnation criteria
        if avg_tau >= self.tau_trigger and avg_kappa >= self.kappa_trigger:
            self.stagnation_counter += 1
            logger.info(f"⚠️ Stagnation detected: {self.stagnation_counter}/{self.patience}")
        else:
            if self.stagnation_counter > 0:
                logger.info("✅ Trajectory recovered. Resetting counter.")
            self.stagnation_counter = 0

        # Start growth process when patience is reached
        if self.stagnation_counter >= self.patience:
            logger.info("🚨 STARTING PROCEDURAL GROWTH")
            device = args.device if hasattr(args, 'device') else torch.device("cuda")
            
            self.grow_and_replace(model, device, state)
            
            self.stagnation_counter = 0
            # Clear old history to save RAM and start fresh
            self.layer_histories.pop(frontier_layer, None)
            control.should_save = True

        return control

    def grow_and_replace(self, model, device, state):
        logger.info(f"\n{'='*70}")
        logger.info("🌱 STARTING GROWTH PROCESS")
        logger.info(f"{'='*70}\n")
        
        try:
            # CRITICAL: Synchronize before major model modification
            if torch.cuda.is_available():
                logger.info("🔄 Synchronizing CUDA before growth...")
                torch.cuda.synchronize()
                logger.info("✅ CUDA synchronized")
            
            if not hasattr(model, 'active_layers') or not model.active_layers:
                logger.error("❌ Model without active_layers defined.")
                return

            current_frontier = min(model.active_layers)
            new_layer_id = current_frontier - 1
            
            # Safety guard
            if new_layer_id < 0:
                logger.info(f"✅ Architecture limit reached. Layer {current_frontier} is already 0.")
                return

            # Retrieve history from BASE layer for extrapolation (current minimum frontier)
            if current_frontier not in self.layer_histories:
                logger.error(f"❌ Critical Error: Growth triggered without history in memory for layer {current_frontier}.")
                return
            
            logger.info(f"📦 Preparing history from layer {current_frontier}...")
            base_history_deque = self.layer_histories[current_frontier]   
            param_history = {k: list(v) for k, v in base_history_deque.items()}
            logger.info(f"✅ History prepared: {len(param_history)} parameters")
        
        except Exception as e:
            logger.error(f"❌ Failed during growth preparation: {e}")
            return

        # Save L1
        self.layer_manager.save_state(
            layer_id=current_frontier, state_name="L1",
            state_dict=model.transformer.h[current_frontier].state_dict(),
            step=state.global_step, strategy='pre_growth'
        )

        logger.info(f"🌱 Growing: Base={current_frontier} -> New={new_layer_id}")
        logger.info(f"🔒 Freezing {len(model.active_layers)} old layers during warmup...")

        # Freeze all active layers
        for old_layer_id in model.active_layers:
            self._freeze_layer(model, old_layer_id)

        # Initialize New Layer
        model.initialize_new_layer(
            layer_id=new_layer_id,
            strategy=self.strategy,
            initializer=self.initializer,
            layer_manager=self.layer_manager,
            param_history=param_history
        )

        # Update Setup
        model.set_active_layers(new_layer_id)
        self._add_layer_to_optimizer(model, new_layer_id)
        self._restart_learning_rate(state)
        
        self.in_warmup = True
        self.warmup_start_step = state.global_step 
        
        logger.info("🧹 Cleaning memory after growth...")
        try:
            if current_frontier in self.layer_histories:
                del self.layer_histories[current_frontier]
            
            self.layer_histories = {} # Clear everything to guarantee
            
            # CRITICAL: Synchronize before cache cleanup to prevent deadlock
            if torch.cuda.is_available():
                logger.info("   🔄 Synchronizing CUDA before cache cleanup...")
                torch.cuda.synchronize()
            
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("   ✅ GPU cache cleared")
            
            logger.info("✅ Memory cleanup completed")
        except Exception as e:
            logger.warning(f"⚠️ Memory cleanup had issues (non-critical): {e}")
        
        logger.info(f"\n{'='*70}")
        logger.info("✅ GROWTH PROCESS COMPLETED")
        logger.info(f"{'='*70}\n")

    def _add_layer_to_optimizer(self, model, layer_id):
        if not self.trainer or not self.trainer.optimizer: 
            return
        
        optimizer = self.trainer.optimizer
        new_layer = model.transformer.h[layer_id]
        
        # 1. Map what already exists in optimizer to avoid duplication (Idempotency)
        existing_param_ids = set()
        for group in optimizer.param_groups:
            for p in group['params']:
                existing_param_ids.add(id(p))
        
        decay_params = []
        no_decay_params = []
        
        # 2. Iterate over new layer
        for name, param in new_layer.named_parameters():
            # Enable gradient tracking
            param.requires_grad = True 
            
            # If parameter already in optimizer, skip (only ensure requires_grad above)
            if id(param) in existing_param_ids:
                continue
            
            # If new, categorize
            if 'bias' in name or 'ln' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        # 3. Add to optimizer groups (only new ones)
        if decay_params:
            optimizer.param_groups[0]['params'].extend(decay_params)
            logger.info(f"   ➕ Added {len(decay_params)} params (decay) from layer {layer_id} to optimizer.")
            
        if no_decay_params:
            optimizer.param_groups[1]['params'].extend(no_decay_params)
            logger.info(f"   ➕ Added {len(no_decay_params)} params (no_decay) from layer {layer_id} to optimizer.")
        
        # 4. Initialize Adam state buffers (avoids runtime crash)
        # Important: Do this for all new parameters
        new_params = decay_params + no_decay_params
        for p in new_params:
            if p not in optimizer.state:
                optimizer.state[p] = {
                    'step': torch.tensor(0.0, device=p.device),
                    'exp_avg': torch.zeros_like(p.data),
                    'exp_avg_sq': torch.zeros_like(p.data)
                }


    def unfreeze_all_layers(self, model):
        # LOG: Explicit unfreezing notice
        logger.info("🔓 End of Warmup. Verifying and releasing all active layers...")
        
        # Ensure everything active is trainable and in optimizer
        for layer_id in model.active_layers:
            for p in model.transformer.h[layer_id].parameters():
                p.requires_grad = True
            self._add_layer_to_optimizer(model, layer_id)
            
        logger.info("🧠 Full training resumed: All active layers are unfrozen.")


    def _restart_learning_rate(self, state):
        """
        Restarts LR warmup with self.warmup_steps steps,
        without changing num_training_steps (total schedule horizon).
        """
        if self.trainer is None or self.trainer.lr_scheduler is None:
            logger.warning("⚠️ Scheduler not available for LR restart")
            return

        optimizer = self.trainer.optimizer
        scheduler = self.trainer.lr_scheduler

        old_lr = optimizer.param_groups[0]["lr"]
        initial_lr = self.trainer.args.learning_rate

        logger.info(f"\n{'─'*70}")
        logger.info("🔄 LR RESTART (LOCAL WARMUP)")
        logger.info(f"{'─'*70}")
        logger.info(f"   LR before restart: {old_lr:.2e}")
        logger.info(f"   Initial LR (target): {initial_lr:.2e}")

        # 1) Force num_warmup_steps to desired value (e.g., 500)
        if hasattr(scheduler, "num_warmup_steps"):
            old_warmup = scheduler.num_warmup_steps
            scheduler.num_warmup_steps = self.warmup_steps
            logger.info(f"   ✅ num_warmup_steps: {old_warmup} → {scheduler.num_warmup_steps}")
        else:
            logger.warning("⚠️ Scheduler does not have num_warmup_steps attribute; "
                        "local warmup may not work as expected.")

        # 2) Reset base_lrs to Trainer's initial LR
        if hasattr(scheduler, "base_lrs"):
            scheduler.base_lrs = [initial_lr for _ in optimizer.param_groups]
            logger.info("   ✅ base_lrs reset to initial learning_rate")

        # 3) Reset scheduler internal timeline
        if hasattr(scheduler, "last_epoch"):
            last_epoch_backup = scheduler.last_epoch
            scheduler.last_epoch = -1
            scheduler.step()  # apply step=0 of new warmup
            logger.info(f"   ✅ last_epoch: {last_epoch_backup} → {scheduler.last_epoch}")
        else:
            logger.warning("⚠️ Scheduler does not have last_epoch attribute; "
                        "unable to reset internal counter.")

        new_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"   LR after restart: {new_lr:.2e}")
        logger.info(f"   Relative boost: {(new_lr / old_lr - 1) * 100:.1f}%")
        logger.info(f"{'─'*70}\n")

        # (Optional) Log to state for later analysis
        if hasattr(state, "log_history"):
            state.log_history.append({
                "event": "lr_restart_warmup",
                "step": state.global_step,
                "old_lr": float(old_lr),
                "new_lr": float(new_lr),
                "warmup_steps": int(getattr(scheduler, "num_warmup_steps", -1)),
                "num_training_steps": int(
                    getattr(scheduler, "num_training_steps",
                            getattr(scheduler, "total_steps", -1))
                ),
            })
    
    def get_training_status(self):
        """Returns current procedural training status."""
        return {
            'in_warmup': self.in_warmup,
            'warmup_start_step': self.warmup_start_step,
            'warmup_steps': self.warmup_steps,
            'stagnation_counter': self.stagnation_counter,
            'tau_trigger': getattr(self, 'tau_trigger', None),
            'kappa_trigger': getattr(self, 'kappa_trigger', None),
        }
