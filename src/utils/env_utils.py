"""
This module contains utility functions for working with environment variables.
"""

import os
from pathlib import Path

def get_env_variable(variable_name: str, persist: bool = True) -> str:
    """
    Get the value of an environment variable. If not found, prompt the user for input.
    
    Args:
        variable_name: Name of the environment variable
        persist: Whether to save the variable to .env file for future use
        
    Returns:
        str: The value of the environment variable
    """
    value = os.getenv(variable_name)
    
    if not value:
        value = input(f"Please enter value for {variable_name}: ")
        os.environ[variable_name] = value  # Set for current session
        
        if persist:
            env_file = Path('.env')
            
            # Read existing contents
            existing_vars = {}
            if env_file.exists():
                with open(env_file, 'r') as f:
                    for line in f:
                        if '=' in line:
                            key, val = line.strip().split('=', 1)
                            existing_vars[key] = val
            
            # Update with new variable
            existing_vars[variable_name] = value
            
            # Write back to file
            with open(env_file, 'w') as f:
                for key, val in existing_vars.items():
                    f.write(f"{key}={val}\n")
    
    return value