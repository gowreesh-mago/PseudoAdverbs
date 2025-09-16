import os
import json
import yaml
import wandb

def load_wandb_config(config_path=None):
    """
    Load wandb configuration from a config file.
    
    Args:
        config_path (str, optional): Path to config file. If None, looks for 
                                   'wandb_config.json' or 'wandb_config.yaml' in current directory.
    
    Returns:
        dict: Configuration dictionary for wandb initialization
    """
    config = {}
    
    # Default config file paths to check
    if config_path is None:
        possible_paths = [
            'wandb_config.json',
            'wandb_config.yaml',
            'wandb_config.yml',
            os.path.expanduser('~/.wandb_config.json'),
            os.path.expanduser('~/.wandb_config.yaml')
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break
    
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                if config_path.endswith('.json'):
                    config = json.load(f)
                elif config_path.endswith(('.yaml', '.yml')):
                    config = yaml.safe_load(f)
            print(f"Loaded wandb config from {config_path}")
        except Exception as e:
            print(f"Warning: Failed to load config from {config_path}: {e}")
    
    return config

def init_wandb_with_config(project_name, run_name, model_config, checkpoint_dir, config_path=None, disable_wandb=False):
    """
    Initialize wandb with configuration from file.
    
    Args:
        project_name (str): WandB project name
        run_name (str): WandB run name
        model_config (dict): Model configuration to log
        checkpoint_dir (str): Directory for checkpoints
        config_path (str, optional): Path to wandb config file
        disable_wandb (bool): If True, disable wandb logging
    """
    if disable_wandb:
        print("WandB logging disabled")
        return
        
    wandb_config = load_wandb_config(config_path)
    
    # Set API key if provided in config
    if 'api_key' in wandb_config:
        os.environ['WANDB_API_KEY'] = wandb_config['api_key']
    
    # Prepare wandb init arguments
    init_args = {
        'project': wandb_config.get('project', project_name),
        'name': wandb_config.get('name', run_name),
        'config': model_config,
        'dir': checkpoint_dir
    }
    
    # Add optional wandb settings from config
    optional_keys = ['entity', 'tags', 'notes', 'mode', 'group']
    for key in optional_keys:
        if key in wandb_config:
            init_args[key] = wandb_config[key]
    
    wandb.init(**init_args)
    print(f"Initialized wandb with project: {init_args['project']}")