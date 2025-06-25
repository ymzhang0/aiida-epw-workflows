"""
Configuration file for SuperCon controllers.

This file contains predefined configurations for different environments
and use cases.
"""

from typing import Dict, Any
from aiida_quantumespresso.common.types import SpinType, ElectronicType


# Base configurations for different environments
LUMI_CONFIG = {
    "max_wallclock_seconds": 7200,
    "account": "project_465000106",
    "queue_name": "debug",
    "num_machines": 1,
    "num_mpiprocs_per_machine": 16,
    "num_cores_per_machine": 16,
}

EIGER_CONFIG = {
    "max_wallclock_seconds": 3600,
    "account": "project_465000106",
    "queue_name": "debug",
    "num_machines": 1,
    "num_mpiprocs_per_machine": 8,
    "num_cores_per_machine": 8,
}

LOCAL_CONFIG = {
    "max_wallclock_seconds": 1800,
    "account": None,
    "queue_name": None,
    "num_machines": 1,
    "num_mpiprocs_per_machine": 4,
    "num_cores_per_machine": 4,
}

# Code configurations for different environments
LUMI_CODES = {
    "pw_code": "pw@lumi",
    "ph_code": "ph@lumi",
    "q2r_code": "q2r@lumi",
    "matdyn_code": "matdyn@lumi",
}

EIGER_CODES = {
    "pw_code": "pw@eiger",
    "ph_code": "ph@eiger",
    "q2r_code": "q2r@eiger",
    "matdyn_code": "matdyn@eiger",
}

LOCAL_CODES = {
    "pw_code": "pw@localhost",
    "ph_code": "ph@localhost",
    "q2r_code": "q2r@localhost",
    "matdyn_code": "matdyn@localhost",
}

# Protocol configurations
PROTOCOLS = {
    "fast": {
        "description": "Fast calculations with lower accuracy",
        "kpoints_distance": 0.3,
        "ecutwfc": 30,
        "conv_thr": 1e-6,
    },
    "moderate": {
        "description": "Balanced accuracy and speed",
        "kpoints_distance": 0.2,
        "ecutwfc": 40,
        "conv_thr": 1e-8,
    },
    "precise": {
        "description": "High accuracy calculations",
        "kpoints_distance": 0.1,
        "ecutwfc": 50,
        "conv_thr": 1e-10,
    },
}

# Predefined controller configurations
CONTROLLER_CONFIGS = {
    "lumi_debug": {
        "environment": "lumi",
        "max_concurrent": 2,
        "protocol": "moderate",
        "description": "LUMI debug configuration for testing",
        **LUMI_CONFIG,
        **LUMI_CODES,
    },
    "lumi_production": {
        "environment": "lumi",
        "max_concurrent": 10,
        "protocol": "moderate",
        "description": "LUMI production configuration",
        **LUMI_CONFIG,
        **LUMI_CODES,
    },
    "eiger_debug": {
        "environment": "eiger",
        "max_concurrent": 3,
        "protocol": "moderate",
        "description": "Eiger debug configuration",
        **EIGER_CONFIG,
        **EIGER_CODES,
    },
    "local_test": {
        "environment": "local",
        "max_concurrent": 1,
        "protocol": "fast",
        "description": "Local testing configuration",
        **LOCAL_CONFIG,
        **LOCAL_CODES,
    },
}


def get_controller_config(config_name: str) -> Dict[str, Any]:
    """Get a predefined controller configuration."""
    if config_name not in CONTROLLER_CONFIGS:
        raise ValueError(f"Unknown configuration: {config_name}. "
                        f"Available: {list(CONTROLLER_CONFIGS.keys())}")
    
    return CONTROLLER_CONFIGS[config_name].copy()


def create_overrides_from_protocol(protocol: str, custom_overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create overrides dictionary from protocol settings."""
    if protocol not in PROTOCOLS:
        raise ValueError(f"Unknown protocol: {protocol}. "
                        f"Available: {list(PROTOCOLS.keys())}")
    
    protocol_config = PROTOCOLS[protocol]
    
    # Base overrides structure
    overrides = {
        "b2w": {
            "scf": {
                "pw": {
                    "parameters": {
                        "SYSTEM": {
                            "ecutwfc": protocol_config["ecutwfc"],
                        },
                        "ELECTRONS": {
                            "conv_thr": protocol_config["conv_thr"],
                        }
                    }
                }
            },
            "nscf": {
                "pw": {
                    "parameters": {
                        "SYSTEM": {
                            "ecutwfc": protocol_config["ecutwfc"],
                        }
                    }
                }
            }
        },
        "a2f": {
            "ph": {
                "parameters": {
                    "INPUTPH": {
                        "epsil": True,
                    }
                }
            }
        }
    }
    
    # Apply custom overrides if provided
    if custom_overrides:
        overrides.update(custom_overrides)
    
    return overrides


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate a controller configuration."""
    required_fields = [
        "max_concurrent", "protocol", "pw_code", "ph_code", 
        "q2r_code", "matdyn_code"
    ]
    
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field: {field}")
    
    if config["max_concurrent"] <= 0:
        raise ValueError("max_concurrent must be positive")
    
    if config["protocol"] not in PROTOCOLS:
        raise ValueError(f"Invalid protocol: {config['protocol']}")
    
    return True


# Example usage functions
def get_lumi_debug_controller(parent_group: str, workchain_group: str):
    """Get a LUMI debug controller configuration."""
    from .supercon_complete import create_supercon_controller
    
    config = get_controller_config("lumi_debug")
    overrides = create_overrides_from_protocol(config["protocol"])
    
    return create_supercon_controller(
        parent_group_label=parent_group,
        group_label=workchain_group,
        overrides=overrides,
        **config
    )


def get_production_controller(parent_group: str, workchain_group: str):
    """Get a production controller configuration."""
    from .supercon_complete import create_supercon_controller
    
    config = get_controller_config("lumi_production")
    overrides = create_overrides_from_protocol(config["protocol"])
    
    return create_supercon_controller(
        parent_group_label=parent_group,
        group_label=workchain_group,
        overrides=overrides,
        **config
    ) 