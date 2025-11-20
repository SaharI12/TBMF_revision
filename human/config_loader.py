"""
Configuration loader for TBMF project.

This module provides utilities to load and manage configuration from YAML files.
"""

import os
import yaml
from typing import Dict, Any
from pathlib import Path


class ConfigLoader:
    """Load and manage project configuration from YAML files."""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration loader.

        Args:
            config_path (str): Path to configuration YAML file.
                              Defaults to "config.yaml" in current directory.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            yaml.YAMLError: If YAML parsing fails.
        """
        self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}\n"
                f"Please copy config.example.yaml to config.yaml and update with your paths:\n"
                f"  cp config.example.yaml config.yaml"
            )

        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        if self.config is None:
            raise ValueError(f"Configuration file is empty: {config_path}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key (str): Configuration key using dot notation (e.g., 'data.root_dir')
            default (Any): Default value if key not found

        Returns:
            Any: Configuration value or default

        Example:
            >>> config = ConfigLoader()
            >>> data_dir = config.get('data.root_dir')
            >>> batch_size = config.get('training.batch_size', 16)
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration."""
        return self.config.get('data', {})

    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.config.get('model', {})

    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self.config.get('training', {})

    def get_validation_config(self) -> Dict[str, Any]:
        """Get validation configuration."""
        return self.config.get('validation', {})

    def get_analysis_config(self) -> Dict[str, Any]:
        """Get analysis configuration."""
        return self.config.get('analysis', {})

    def get_normalization_config(self) -> Dict[str, Any]:
        """Get normalization configuration."""
        return self.config.get('normalization', {})

    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        dirs_to_create = [
            self.get('data.root_dir'),
            self.get('model.checkpoint_dir'),
            self.get('analysis.predictions_dir'),
            self.get('logging.log_dir'),
        ]

        for dir_path in dirs_to_create:
            if dir_path:
                Path(dir_path).mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"ConfigLoader(path={self.config_path})"


def load_config(config_path: str = "config.yaml") -> ConfigLoader:
    """
    Convenience function to load configuration.

    Args:
        config_path (str): Path to configuration file

    Returns:
        ConfigLoader: Configuration loader instance
    """
    return ConfigLoader(config_path)


# Example usage
if __name__ == "__main__":
    try:
        config = load_config()
        print(f"Configuration loaded from: {config.config_path}")
        print(f"Data directory: {config.get('data.root_dir')}")
        print(f"Model checkpoint: {config.get('model.model1_path')}")
        print(f"Batch size: {config.get('training.batch_size')}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
