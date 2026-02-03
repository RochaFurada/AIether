import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="GPT model training with CLI configuration")
    
    # Training arguments
    train_group = parser.add_argument_group('training', 'Training parameters')
    train_group.add_argument("--output_dir", type=str, default="./HF_model_output",
                            help="Model output directory")
    train_group.add_argument("--num_epochs", type=int, default=3,
                            help="Number of training epochs")
    train_group.add_argument("--batch_size", type=int, default=8,
                            help="Batch size per device")
    train_group.add_argument("--lr", type=float, default=5e-4,
                            help="Learning rate")
    train_group.add_argument("--warmup_steps", type=int, default=500,
                            help="Number of warmup steps")
    train_group.add_argument("--weight_decay", type=float, default=0.01,
                            help="Weight decay for regularization")
    train_group.add_argument("--block_size", type=int, default=None,
                            help="Context block size (None = auto-detect)")
    train_group.add_argument("--logging_interval", type=int, default=100,
                            help="Logging interval in steps")
    train_group.add_argument("--eval_interval", type=int, default=500,
                            help="Evaluation interval in steps (used if evals_per_epoch not specified)")
    train_group.add_argument("--evals_per_epoch", type=int, default=None,
                            help="Number of evaluations per epoch (recommended)")
    train_group.add_argument("--save_interval", type=int, default=None,
                            help="Checkpoint save interval (None = 4x eval_interval)")
    train_group.add_argument("--new_layer_strategy", type=str, default="GeometricExtrapolator",
                            help="Layer growth strategy")
    
    # Model arguments
    model_group = parser.add_argument_group('model', 'GPT model parameters')
    model_group.add_argument("--n_layer", type=int, default=6,
                            help="Number of transformer layers")
    model_group.add_argument("--n_head", type=int, default=6,
                            help="Number of attention heads")
    model_group.add_argument("--n_embd", type=int, default=384,
                            help="Embedding dimension")
    model_group.add_argument("--dropout", type=float, default=0.1,
                            help="Dropout rate")
    model_group.add_argument("--bias", type=lambda x: x.lower() == 'true', default=True,
                            help="Use bias in linear layers (true/false)")
    
    # Dataset arguments
    dataset_group = parser.add_argument_group('dataset', 'Dataset parameters')
    dataset_group.add_argument("--dataset_path", type=str, 
                              default="/kaggle/input/finewebedu-52m/FineWebEdu-Tokenized- 52M -512",
                              help="Path to tokenized dataset")
    
    # Procedural Growth arguments
    pg_group = parser.add_argument_group('procedural_growth', 'Procedural growth parameters')
    pg_group.add_argument("--pg_patience", type=int, default=3,
                         help="Patience for procedural growth")
    pg_group.add_argument("--pg_threshold", type=float, default=0.15,
                         help="Threshold for procedural growth")
    pg_group.add_argument("--pg_warmup_steps", type=int, default=400,
                         help="Warmup steps for procedural growth")
    pg_group.add_argument("--pg_history_window", type=int, default=15,
                         help="History window size")
    pg_group.add_argument("--pg_collect_interval", type=int, default=100,
                         help="Snapshot collection interval")
    
    # Experiment arguments (ablation)
    exp_group = parser.add_argument_group('experiment', 'Experiment parameters')
    exp_group.add_argument("--beta0", type=float, default=None,
                          help="Beta0 parameter for experiment")
    exp_group.add_argument("--gamma0", type=float, default=None,
                          help="Gamma0 parameter for experiment")
    exp_group.add_argument("--eta0", type=float, default=None,
                          help="Eta0 parameter for experiment")
    exp_group.add_argument("--k_subspace", type=int, default=None,
                          help="K_subspace parameter for experiment")
    exp_group.add_argument("--ell_orth", type=int, default=None,
                          help="Ell_orth parameter for experiment")
    exp_group.add_argument("--tau_crit", type=float, default=None,
                          help="Tau_crit parameter for experiment")
    exp_group.add_argument("--kappa_crit", type=float, default=None,
                          help="Kappa_crit parameter for experiment")
    
    # W&B arguments
    wandb_group = parser.add_argument_group('wandb', 'Weights & Biases parameters')
    wandb_group.add_argument("--wandb_project", type=str, default="AIetherResearch",
                            help="W&B project name")
    wandb_group.add_argument("--wandb_name", type=str, default=None,
                            help="W&B run name (None = auto-generate)")
    wandb_group.add_argument("--wandb_tags", type=str, nargs='+', default=None,
                            help="Tags for W&B run")
    wandb_group.add_argument("--wandb_group", type=str, default="ablc",
                            help="W&B run group")
    
    return parser.parse_args()


def args_to_config(args):
    """
    Converts command line arguments to configuration structure.
    """
    config = {
        'training': {
            'output_dir': args.output_dir,
            'num_epochs': args.num_epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'warmup_steps': args.warmup_steps,
            'weight_decay': args.weight_decay,
            'logging_interval': args.logging_interval,
            'eval_interval': args.eval_interval,
            'new_layer_strategy': args.new_layer_strategy,
        },
        'model': {
            'n_layer': args.n_layer,
            'n_head': args.n_head,
            'n_embd': args.n_embd,
            'dropout': args.dropout,
            'bias': args.bias
        },
        'dataset': {
            'path': args.dataset_path
        },
        'Procedural_growth': {
            'patience': args.pg_patience,
            'threshold': args.pg_threshold,
            'warmup_steps': args.pg_warmup_steps,
            'history_window_size': args.pg_history_window,
            'collect_interval_steps': args.pg_collect_interval
        },
        'experiment': {
            'beta0': args.beta0,
            'gamma0': args.gamma0,
            'eta0': args.eta0,
            'k_subspace': args.k_subspace,
            'ell_orth': args.ell_orth,
            'tau_crit': args.tau_crit,
            'kappa_crit': args.kappa_crit
        },
        'wandb': {
            'project': args.wandb_project,
            'name': args.wandb_name,
            'tags': args.wandb_tags,
            'group': args.wandb_group
        }
    }
    
    # Add optional parameters
    if args.block_size is not None:
        config['training']['block_size'] = args.block_size
    if args.evals_per_epoch is not None:
        config['training']['evals_per_epoch'] = args.evals_per_epoch
    if args.save_interval is not None:
        config['training']['save_interval'] = args.save_interval
    
    # Validate required experimental parameters
    required_experiment_params = {
        'beta0': args.beta0,
        'gamma0': args.gamma0,
        'eta0': args.eta0,
        'k_subspace': args.k_subspace,
        'ell_orth': args.ell_orth
    }
    
    missing_params = [param for param, value in required_experiment_params.items() if value is None]
    
    if missing_params:
        raise ValueError(
            f"❌ Required experimental parameters not provided: {', '.join(missing_params)}\n"
            f"Please provide all parameters: --beta0, --gamma0, --eta0, --k_subspace, --ell_orth"
        )
    
    config['experiment']['beta0'] = args.beta0
    config['experiment']['gamma0'] = args.gamma0
    config['experiment']['eta0'] = args.eta0
    config['experiment']['k_subspace'] = args.k_subspace
    config['experiment']['ell_orth'] = args.ell_orth
    
    return config
