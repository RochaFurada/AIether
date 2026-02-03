import torch
import math
import wandb
from pathlib import Path
from transformers import GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_from_disk
from aiether.models import GPTForHF, GPTConfig
from aiether.callbacks import ProceduralGrowthCallback, DeepDebugCallback
from aiether.utils import logger_setup, parse_args, args_to_config

logger = logger_setup()

def train_with_hf_trainer():
    """
    Executes model training using Hugging Face Trainer with command line configuration.
    """
    # Process command line arguments
    args = parse_args()
    cfg = args_to_config(args)
    
    logger.info("Setup Configurado:")
    for key, value in cfg.items():
        logger.info(f"Config {key}: {value}\n")
    
    train_cfg = cfg.get('training', {})
    dataset_cfg = cfg.get('dataset', {})
    model_cfg = cfg.get('model', {})
    procedural_growth_cfg = cfg.get('Procedural_growth', {})
    experiment_cfg = cfg.get('experiment', {})
    wandb_cfg = cfg.get('wandb', {})

    # Extract experimental parameters
    beta0 = experiment_cfg.get('beta0')
    gamma0 = experiment_cfg.get('gamma0')
    eta0 = experiment_cfg.get('eta0')
    k_subspace = experiment_cfg.get('k_subspace')
    ell_orth = experiment_cfg.get('ell_orth')
    tau_crit = experiment_cfg.get('tau_crit')
    kappa_crit = experiment_cfg.get('kappa_crit')
    
    print(f"Experiment parameters: beta0={beta0}, gamma0={gamma0}, eta0={eta0}, k_subspace={k_subspace}, ell_orth={ell_orth}, tau_crit={tau_crit}, kappa_crit={kappa_crit}")

    try:
        # Load GPT-2 tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        
        # Configure pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        vocab_size = tokenizer.vocab_size
        logger.info(f"✅ GPT-2 tokenizer loaded. Vocab size: {vocab_size}")
        
    except Exception as e:
        logger.error(f"❌ Error loading tokenizer: {e}")
        return
    
    # Load dataset
    try:
        raw_datasets = load_from_disk(dataset_cfg.get('path'))
        
        train_dataset = raw_datasets["train"]
        eval_dataset = raw_datasets.get("validation", None)
        
        logger.info(f"📊 Train samples: {len(train_dataset)}")
        if eval_dataset:
            logger.info(f"📊 Eval samples: {len(eval_dataset)}")
        
    except Exception as e:
        logger.error(f"❌ Error loading dataset: {e}")
        return
    
    # Analyze sequence sizes in dataset
    try:
        logger.info("🔍 Analyzing sequence sizes in dataset...")
        max_length = 0
        sample_sizes = []
        
        num_samples_check = min(1000, len(train_dataset))
        for i in range(num_samples_check):
            seq_len = len(train_dataset[i]['input_ids'])
            sample_sizes.append(seq_len)
            max_length = max(max_length, seq_len)
        
        avg_length = sum(sample_sizes) / len(sample_sizes)
        logger.info(f"📏 Average sequence size: {avg_length:.1f}")
        logger.info(f"📏 Maximum size found: {max_length}")
        
        block_size = train_cfg.get('block_size', max_length)
        
        if block_size < max_length:
            logger.warning(f"⚠️  Configured block_size ({block_size}) is smaller than dataset maximum ({max_length})")
            logger.warning(f"⚠️  Adjusting block_size to {max_length}")
            block_size = max_length
        
        logger.info(f"✅ Final block size: {block_size}")
        
    except Exception as e:
        logger.error(f"❌ Error detecting sequence sizes: {e}")
        block_size = 1024
        logger.warning(f"⚠️  Using default block_size: {block_size}")
    
    # W&B experiment configuration
    wandb_name = wandb_cfg.get('name')
    if wandb_name is None:
        wandb_name = f"AIether_{train_cfg['num_epochs']}ep_{model_cfg['n_layer']}L_b{beta0}_g{gamma0}_e{eta0}"
    
    wandb_tags = wandb_cfg.get('tags')
    if wandb_tags is None:
        wandb_tags = [
            f"{model_cfg['n_layer']}L",
            "ablc",
            f"{train_cfg['num_epochs']}ep",
            f"b{str(beta0)}",
            f"g{str(gamma0)}",
            f"e{str(eta0)}",
            f"k{str(k_subspace)}",
            f"l{str(ell_orth)}"
        ]
    
    run = wandb.init(
        project=wandb_cfg.get('project', 'AIetherResearch'),
        name=wandb_name,
        tags=wandb_tags,
        group=wandb_cfg.get('group', 'null')
    )
    
    # Data collator configuration for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    eval_interval_dynamic = None
    eval_strategy_dynamic = "no"
    
    if eval_dataset:
        eval_strategy_dynamic = "steps"
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        global_batch_size = train_cfg.get('batch_size', 8) * num_gpus
        steps_per_epoch = math.ceil(len(train_dataset) / global_batch_size)
        
        evals_per_epoch = train_cfg.get('evals_per_epoch', None)
        if evals_per_epoch is not None and evals_per_epoch > 0:
            eval_interval_dynamic = math.floor(steps_per_epoch / evals_per_epoch)
            eval_interval_dynamic = max(1, eval_interval_dynamic)
            logger.info(f"DYNAMIC CALCULATION: {steps_per_epoch} steps/epoch, {evals_per_epoch} evals/epoch -> eval_interval = {eval_interval_dynamic} steps")
        else:
            eval_interval_dynamic = train_cfg.get('eval_interval', 500)
            logger.warning(f"WARNING: Using fixed 'eval_interval' of {eval_interval_dynamic} steps. Recommend using 'evals_per_epoch' for relative evaluation.")
    
    # Instantiate GPT model
    config = GPTConfig(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=model_cfg.get('n_layer', 6),
        n_head=model_cfg.get('n_head', 6),
        n_embd=model_cfg.get('n_embd', 384),
        dropout=model_cfg.get('dropout', 0.1),
        bias=model_cfg.get('bias', True)
    )
    model = GPTForHF(config)
    
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"✅ Model created with {num_params/1e6:.2f}M parameters")
    
    # Configure training arguments
    save_interval = eval_interval_dynamic * 4 if eval_interval_dynamic else 2000
    
    training_args = TrainingArguments(
        output_dir=train_cfg.get('output_dir', './HF_model_output'),
        num_train_epochs=train_cfg.get('num_epochs', 3),
        per_device_train_batch_size=train_cfg.get('batch_size', 8),
        per_device_eval_batch_size=train_cfg.get('batch_size', 8),
        warmup_steps=train_cfg.get('warmup_steps', 500),
        weight_decay=train_cfg.get('weight_decay', 0.01),
        learning_rate=train_cfg.get('lr', 5e-4),
        lr_scheduler_type="cosine",
        logging_dir=(train_cfg.get('output_dir', './HF_model_output') + "/logs"),
        logging_steps=train_cfg.get('logging_interval', 100),
        eval_strategy=eval_strategy_dynamic,
        eval_steps=eval_interval_dynamic,
        save_steps=save_interval,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="eval_loss" if eval_dataset else None,
        report_to=["tensorboard", "wandb"]
    )
    
    # Configure procedural growth parameters
    growth_params = {
        'beta0': beta0,
        'gamma0': gamma0,
        'eta0': eta0,
        'k_subspace': k_subspace,
        'ell_orth': ell_orth
    }

    PGD_instance = ProceduralGrowthCallback(
        patience=procedural_growth_cfg.get("patience", 3),
        threshold=procedural_growth_cfg.get("threshold", 0.15),
        strategy=train_cfg.get("new_layer_strategy", "GeometricExtrapolator"),
        warmup_steps=procedural_growth_cfg.get("warmup_steps", 400),
        output_dir=training_args.output_dir,
        history_window_size=procedural_growth_cfg.get("history_window_size", 15),
        collect_interval_steps=procedural_growth_cfg.get("collect_interval_steps", 100),
        growth_params=growth_params,
        tau_crit=tau_crit,
        kappa_crit=kappa_crit
    )
    
    # Instantiate Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=[PGD_instance, DeepDebugCallback()]
    )
    PGD_instance.attach_trainer(trainer)
    
    # Execute training
    logger.info("🚀 Starting training...")
    try:
        trainer.train()
        logger.info("✅ Training completed!")
        
        # Persist final model
        final_path = Path(training_args.output_dir) / "final_model"
        trainer.save_model(final_path)
        logger.info(f"💾 Model saved at: {final_path}")
        
    except Exception as e:
        logger.error(f"❌ Error during training: {e}")
        raise
