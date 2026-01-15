#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration loading utilities for the multimodal medical imaging analysis project.

@author: mumuaktar, dscarmo
"""
# Standard library imports
import os
import argparse
import sys
from pathlib import Path

# Third-party library imports
import yaml


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_config_args(description: str, example_usage: str, default_config: str = None) -> dict:
    """
    Parse command line arguments and load configuration from YAML file.
    
    Args:
        description: Description for the argument parser
        example_usage: Example usage string for the epilog
        default_config: Default config path for Jupyter notebooks
        
    Returns:
        Dictionary containing configuration parameters with 'debug' and 'config_name' added
    """
    parser = argparse.ArgumentParser(description=description, epilog=example_usage)
    
    parser.add_argument(
        "config",
        type=str,
        help="Path to YAML configuration file (required)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )

    # Detect if running in Jupyter
    if "ipykernel" in sys.modules:
        # Jupyter: ignore sys.argv to prevent conflicts
        if default_config is None:
            raise ValueError("default_config must be provided for Jupyter notebooks")
        parsed_args = parser.parse_args(args=[default_config])
    else:
        # Standard script: parse normally
        parsed_args = parser.parse_args()

    # Load configuration from YAML file
    config = load_config(parsed_args.config)

    # Check if data_dir and output_base_dir are set
    if config['data_dir'] is None or config['output_base_dir'] is None:
        # Try to get from environment variables
        config['data_dir'] = os.getenv('DATA_DIR', None)
        config['output_base_dir'] = os.getenv('OUTPUT_BASE_DIR', None)
    
    # Add debug flag and config metadata to the dictionary
    config['debug'] = parsed_args.debug
    config['config_path'] = parsed_args.config
    config['config_name'] = Path(parsed_args.config).stem  # Get filename without extension
    
    # Pretty print configuration
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("--------------------------------")

    return config
