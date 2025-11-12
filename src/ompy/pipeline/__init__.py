"""
Pipeline infrastructure for OMpy.

This module provides:
- Stage: Enum for pipeline stages
- Result: Base class for pipeline results with stage tracking
- lift: Decorator for automatic dispatch on collections/ensembles
- template_strategy: Ensemble strategy for automatic template extraction
- Helper functions: unwrap(), get_stage()
"""
from .stage import Stage
from .result import Result, ResultMeta, Settings
from .lifting import lift, unwrap, get_stage, template_strategy

__all__ = [
    'Stage',
    'Result', 
    'ResultMeta',
    'lift',
    'unwrap',
    'get_stage',
    'Settings',
    'template_strategy',
]

