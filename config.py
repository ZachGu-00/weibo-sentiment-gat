"""
Configuration file for Weibo Sentiment Analysis and GAT Training Pipeline
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class SentimentAnalysisConfig:
    """Configuration for sentiment analysis."""
    
    # Model configurations
    models: List[Dict] = None
    
    # Processing settings
    batch_size: int = 32
    max_length: int = 128
    device: Optional[str] = None  # Auto-detect if None
    
    def __post_init__(self):
        if self.models is None:
            self.models = [
                {
                    'name': 'IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment',
                    'weight': 0.5,
                    'type': 'sentiment'
                },
                {
                    'name': 'techthiyanes/chinese_sentiment',
                    'weight': 0.3,
                    'type': 'sentiment'
                },
                {
                    'name': 'uer/roberta-base-finetuned-chinanews-chinese',
                    'weight': 0.2,
                    'type': 'news'
                }
            ]


@dataclass
class DataPreprocessingConfig:
    """Configuration for data preprocessing."""
    
    # Data split ratios
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Feature extraction
    pca_components: int = 128
    embedding_dim: int = 768
    
    # Negative sampling configurations
    sampling_configs: Dict = None
    
    # Random seed
    seed: int = 42
    
    def __post_init__(self):
        if self.sampling_configs is None:
            self.sampling_configs = {
                'baseline': {
                    'train_ratios': [1.0, 0.0, 0.0],  # [easy, medium, hard]
                    'val_test_ratios': [1.0, 0.0, 0.0],
                    'description': 'Pure random sampling'
                },
                'medium': {
                    'train_ratios': [0.5, 0.3, 0.2],
                    'val_test_ratios': [0.4, 0.3, 0.3],
                    'description': 'Medium mixed sampling'
                },
                'advanced': {
                    'train_ratios': [0.2, 0.3, 0.5],
                    'val_test_ratios': [0.2, 0.3, 0.5],
                    'description': 'Advanced mixed sampling'
                }
            }


@dataclass
class GATModelConfig:
    """Configuration for GAT model."""
    
    # Model architecture
    hidden_channels: int = 64
    out_channels: int = 32
    heads: int = 4
    dropout: float = 0.3
    
    # Edge predictor architecture
    edge_hidden_dim: int = 32
    
    # Random seed
    seed: int = 42


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    # Stage-specific configurations
    stage_configs: Dict = None
    
    # General training settings
    early_stopping_patience: int = 25
    gradient_clip_norm: float = 1.0
    
    # Device settings
    device: Optional[str] = None  # Auto-detect if None
    
    def __post_init__(self):
        if self.stage_configs is None:
            self.stage_configs = {
                'baseline': {
                    'epochs': 120,
                    'patience': 25,
                    'lr': 0.003,
                    'weight_decay': 1e-4,
                    'edge_batch_size': 4096,
                    'scheduler_step': 30,
                    'scheduler_gamma': 0.9,
                    'seed': 42
                },
                'medium': {
                    'epochs': 150,
                    'patience': 30,
                    'lr': 0.002,
                    'weight_decay': 1e-4,
                    'edge_batch_size': 4096,
                    'scheduler_step': 40,
                    'scheduler_gamma': 0.9,
                    'seed': 42
                },
                'advanced': {
                    'epochs': 180,
                    'patience': 35,
                    'lr': 0.001,
                    'weight_decay': 5e-5,
                    'edge_batch_size': 4096,
                    'scheduler_step': 45,
                    'scheduler_gamma': 0.95,
                    'seed': 42
                }
            }


@dataclass
class PathConfig:
    """Configuration for file paths."""
    
    # Input paths
    raw_network_path: str = ""
    
    # Output directories
    output_dir: str = "./output"
    sentiment_output_dir: str = "./output/sentiment"
    preprocessing_output_dir: str = "./output/preprocessing"
    training_output_dir: str = "./output/training"
    
    # Model paths
    pretrained_model_path: Optional[str] = None
    
    def __post_init__(self):
        # Create output directories
        for dir_path in [
            self.output_dir,
            self.sentiment_output_dir,
            self.preprocessing_output_dir,
            self.training_output_dir
        ]:
            os.makedirs(dir_path, exist_ok=True)


@dataclass
class Config:
    """Main configuration class."""
    
    sentiment: SentimentAnalysisConfig = None
    preprocessing: DataPreprocessingConfig = None
    model: GATModelConfig = None
    training: TrainingConfig = None
    paths: PathConfig = None
    
    def __post_init__(self):
        if self.sentiment is None:
            self.sentiment = SentimentAnalysisConfig()
        if self.preprocessing is None:
            self.preprocessing = DataPreprocessingConfig()
        if self.model is None:
            self.model = GATModelConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.paths is None:
            self.paths = PathConfig()
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'Config':
        """Create config from dictionary."""
        return cls(
            sentiment=SentimentAnalysisConfig(**config_dict.get('sentiment', {})),
            preprocessing=DataPreprocessingConfig(**config_dict.get('preprocessing', {})),
            model=GATModelConfig(**config_dict.get('model', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            paths=PathConfig(**config_dict.get('paths', {}))
        )
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {
            'sentiment': self.sentiment.__dict__,
            'preprocessing': self.preprocessing.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'paths': self.paths.__dict__
        }


# Default configuration instance
DEFAULT_CONFIG = Config()


# Example custom configuration
def create_custom_config(
    raw_network_path: str,
    output_dir: str = "./output",
    epochs_baseline: int = 120,
    epochs_medium: int = 150,
    epochs_advanced: int = 180,
    learning_rate: float = 0.003,
    batch_size: int = 4096
) -> Config:
    """Create a custom configuration with common parameters."""
    
    config = Config()
    
    # Update paths
    config.paths.raw_network_path = raw_network_path
    config.paths.output_dir = output_dir
    config.paths.sentiment_output_dir = os.path.join(output_dir, "sentiment")
    config.paths.preprocessing_output_dir = os.path.join(output_dir, "preprocessing")
    config.paths.training_output_dir = os.path.join(output_dir, "training")
    
    # Update training configuration
    config.training.stage_configs['baseline']['epochs'] = epochs_baseline
    config.training.stage_configs['medium']['epochs'] = epochs_medium
    config.training.stage_configs['advanced']['epochs'] = epochs_advanced
    config.training.stage_configs['baseline']['lr'] = learning_rate
    config.training.stage_configs['baseline']['edge_batch_size'] = batch_size
    config.training.stage_configs['medium']['edge_batch_size'] = batch_size
    config.training.stage_configs['advanced']['edge_batch_size'] = batch_size
    
    # Create output directories
    for dir_path in [
        config.paths.output_dir,
        config.paths.sentiment_output_dir,
        config.paths.preprocessing_output_dir,
        config.paths.training_output_dir
    ]:
        os.makedirs(dir_path, exist_ok=True)
    
    return config


# Configuration validation
def validate_config(config: Config) -> List[str]:
    """Validate configuration and return list of errors."""
    errors = []
    
    # Check required paths
    if not config.paths.raw_network_path:
        errors.append("raw_network_path is required")
    
    # Check data split ratios
    total_ratio = (config.preprocessing.train_ratio + 
                  config.preprocessing.val_ratio + 
                  config.preprocessing.test_ratio)
    if abs(total_ratio - 1.0) > 1e-6:
        errors.append(f"Data split ratios must sum to 1.0, got {total_ratio}")
    
    # Check model parameters
    if config.model.heads <= 0:
        errors.append("Number of attention heads must be positive")
    
    if config.model.dropout < 0 or config.model.dropout >= 1:
        errors.append("Dropout must be in range [0, 1)")
    
    # Check training parameters
    for stage_name, stage_config in config.training.stage_configs.items():
        if stage_config['epochs'] <= 0:
            errors.append(f"Epochs for stage {stage_name} must be positive")
        if stage_config['lr'] <= 0:
            errors.append(f"Learning rate for stage {stage_name} must be positive")
    
    return errors