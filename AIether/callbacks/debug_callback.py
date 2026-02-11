from transformers.trainer_callback import TrainerCallback
from aiether.utils import logger_setup
import torch

logger = logger_setup()

class DeepDebugCallback(TrainerCallback):
    """Callback that captures and logs detailed model state information during training."""
    
    def __init__(self):
        self.last_grad_norms = {}
        self.last_log_step = -1
    
    def on_optimizer_step(self, args, state, control, **kwargs):
        """Captures gradient norms before the optimizer reset operation."""
        # Reduced frequency and added safety checks
        if state.global_step % 1000 != 0:
            return control
        
        try:
            model = kwargs.get('model')
            if model is None:
                return control
            
            # Capture gradients using block ID with timeout protection
            self.last_grad_norms = {}
            for block in model.transformer.h:
                grad_norms = []
                for name, param in block.named_parameters():
                    if param.grad is not None:
                        try:
                            norm_val = param.grad.norm().item()
                            if not torch.isnan(torch.tensor(norm_val)) and not torch.isinf(torch.tensor(norm_val)):
                                grad_norms.append(norm_val)
                        except:
                            pass  # Skip problematic gradients
                
                self.last_grad_norms[id(block)] = grad_norms
        except Exception as e:
            logger.warning(f"⚠️ Debug callback gradient capture failed: {e}")
        
        return control
    
    def on_step_end(self, args, state, control, model=None, optimizer=None, **kwargs):
        """Logs model and optimizer information after training step completion."""
        if state.global_step % 1000 != 0:
            return control
        
        # Prevent duplicate logging
        if state.global_step == self.last_log_step:
            return control
        self.last_log_step = state.global_step
        
        try:
            logger.info(f"\n{'='*70}")
            logger.info(f"🔬 DEEP DEBUG - Step {state.global_step}")
            logger.info(f"{'='*70}\n")
            
            # 1. MODEL STATE
            logger.info("\n📦 MODEL STATE:")
            for idx, block in enumerate(model.transformer.h):
                params_with_grad = []
                elements_with_grad = 0
                elements_total = 0
                
                for name, param in block.named_parameters():
                    elements_total += param.numel()
                    if param.requires_grad:
                        params_with_grad.append(name)
                        elements_with_grad += param.numel()
                
                status = "🟢" if len(params_with_grad) > 0 else "⚪"
                logger.info(f"   {status} Layer {idx}:")
                logger.info(f"      - {len(params_with_grad)}/{len(list(block.parameters()))} trainable params")
                logger.info(f"      - {elements_with_grad:,}/{elements_total:,} elements")
            
            # 2. OPTIMIZER STATE
            if optimizer is not None:
                logger.info(f"\n🔧 OPTIMIZER STATE:")
                logger.info(f"   Type: {type(optimizer).__name__}")
                
                total_params = sum(len(pg['params']) for pg in optimizer.param_groups)
                total_elements = sum(
                    sum(p.numel() for p in pg['params']) 
                    for pg in optimizer.param_groups
                )
                
                logger.info(f"   Total: {total_params} params, {total_elements:,} elements")
                logger.info(f"   Momentum states: {len(optimizer.state)}")
                logger.info(f"   LR: {optimizer.param_groups[0]['lr']:.2e}")
            
            # 3. GRADIENTS (with timing note)
            logger.info(f"\n📊 GRADIENTS (captured IN PREVIOUS STEP):")
            logger.info(f"   ⚠️  Note: Gradients show state from step {state.global_step - 1}")
            
            for idx, block in enumerate(model.transformer.h):
                block_id = id(block)
                has_grad_now = any(p.requires_grad for p in block.parameters())
                
                if block_id in self.last_grad_norms and len(self.last_grad_norms[block_id]) > 0:
                    grad_norms = self.last_grad_norms[block_id]
                    avg = sum(grad_norms) / len(grad_norms)
                    logger.info(f"   ✅ Layer {idx}: avg={avg:.6f}, max={max(grad_norms):.6f}")
                else:
                    status = "🆕" if has_grad_now else "⚪"
                    logger.info(f"   {status} Layer {idx}: no gradients captured")
            
            # 4. MODEL ↔ OPTIMIZER CONNECTION
            if optimizer is not None:
                logger.info(f"\n🔗 MODEL ↔ OPTIMIZER CONNECTION:")
                
                opt_param_ids = set()
                for pg in optimizer.param_groups:
                    opt_param_ids.update(id(p) for p in pg['params'])
                
                for idx, block in enumerate(model.transformer.h):
                    params_in_model = list(block.parameters())
                    params_trainable = [p for p in params_in_model if p.requires_grad]
                    params_in_opt = [p for p in params_trainable if id(p) in opt_param_ids]
                    
                    if len(params_trainable) == 0:
                        logger.info(f"   ⚪ Layer {idx}: not trainable")
                    elif len(params_in_opt) == len(params_trainable):
                        logger.info(f"   ✅ Layer {idx}: {len(params_in_opt)}/{len(params_trainable)} in optimizer")
                    else:
                        logger.info(f"   ❌ Layer {idx}: {len(params_in_opt)}/{len(params_trainable)} in optimizer")
            
            logger.info(f"{'='*70}\n")
        except Exception as e:
            logger.error(f"❌ Debug callback failed at step {state.global_step}: {e}")
        
        return control
