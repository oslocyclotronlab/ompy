from __future__ import annotations
from abc import ABC, ABCMeta, abstractmethod
from dataclasses import dataclass, asdict
from typing import ClassVar, Type, Any
from .stubs import VectorizedFunction
import numpy as np

class ParametersBase:
    _subclasses: ClassVar[dict[str, type['ParametersBase']]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        ParametersBase._subclasses[cls.__name__] = cls

    @classmethod
    def get_type(cls, name: str) -> Type['ParametersBase']:
        if name in cls._subclasses:
            return cls._subclasses[name]
        raise ValueError(f"No subclass with name {name} found")

    @abstractmethod
    def parameter_descriptions(self) -> list[tuple[str, Any, str]]:
        """ Return a list of tuples with (parameter_name, value, description) """
        pass
    
    def _repr_html_(self) -> str:
        return make_html_table(self.parameter_descriptions())


@dataclass(slots=True, frozen=True, kw_only=True)
class Parameters(ParametersBase):
    pass


@dataclass(slots=True, frozen=True, kw_only=True)
class ModelParameters(Parameters):
    pass


class ModelMeta(ABCMeta, type):
    def __new__(mcs, name, bases, namespace, **kwargs):
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)
        if 'parameters' in namespace.get('__annotations__', {}):
            param_type_name = namespace['__annotations__']['parameters']
            if isinstance(param_type_name, str):
                param_type = Parameters.get_type(param_type_name)
            else:
                param_type = param_type_name
            setattr(cls, 'parameter_type', param_type)
        return cls


@dataclass(slots=True, frozen=True)
class Model(ABC, metaclass=ModelMeta):
    """ Base class for all *complete* models

    A complete model is one that has both parameters and a callback.
    This differs from a model *specification* which would be a model without
    parameters specified (TODO).
    """
    parameters: ModelParameters
    callback: VectorizedFunction

    def __call__(self, e: float | np.ndarray) -> float | np.ndarray:
        return self.callback(e, **asdict(self.parameters))

    @classmethod
    def from_kwargs(cls, **kwargs):
        if not hasattr(cls, 'parameter_type'):
            raise ValueError("No parameters field found in model")

        parsed_params = cls.parameter_type(**kwargs)
        return cls(parsed_params)

    @classmethod
    def from_params(cls, parameters):
        if not hasattr(cls, 'parameter_type'):
            raise ValueError("No parameters field found in model")

        if not isinstance(parameters, cls.parameter_type):
            raise TypeError(f"Expected {cls.parameter_type.__name__}, got {type(parameters).__name__}")

        return cls(parameters) 

    def _repr_html_(self) -> str:
        return make_html_table(self.parameters.parameter_descriptions(), title=self.__class__.__name__)

        


def make_html_table(params: list[tuple[str, Any, str]], title: str = None, subtitle: str = None) -> str:
    """
    Create a nicely styled HTML table with optional title and subtitle.
    
    Args:
        params: List of tuples with (parameter_name, value, description)
        title: Optional title for the table
        subtitle: Optional subtitle for the table
        
    Returns:
        HTML string for the table
    """
    html = """
    <div style="font-family: Arial, sans-serif; max-width: 800px; margin: 20px auto;">
    """
    
    # Add title if provided
    if title:
        html += f"""
        <h2 style="color: #2c3e50; margin-bottom: 5px;">{title}</h2>
        """
        
    # Add subtitle if provided
    if subtitle:
        html += f"""
        <p style="color: #7f8c8d; margin-top: 0; margin-bottom: 15px; font-style: italic;">{subtitle}</p>
        """
    
    html += f"""
    <table style="width: 100%; border-collapse: collapse; border-radius: 8px; overflow: hidden; box-shadow: 0 0 20px rgba(0, 0, 0, 0.1);">
        <thead>
            <tr style="background-color: #3498db; color: white;">
                <th style="padding: 12px 15px; text-align: left; border-bottom: 1px solid #ddd;">Parameter</th>
                <th style="padding: 12px 15px; text-align: left; border-bottom: 1px solid #ddd;">Value</th>
                <th style="padding: 12px 15px; text-align: left; border-bottom: 1px solid #ddd;">Description</th>
            </tr>
        </thead>
        <tbody>
    """
    
    for i, (param, value, desc) in enumerate(params):
        # Alternate row colors for better readability
        bg_color = "#f2f2f2" if i % 2 == 0 else "white"
        
        html += f"""
        <tr style="background-color: {bg_color};">
            <td style="padding: 12px 15px; border-bottom: 1px solid #ddd; font-weight: bold;">{param}</td>
            <td style="padding: 12px 15px; border-bottom: 1px solid #ddd; font-family: monospace;">{value}</td>
            <td style="padding: 12px 15px; border-bottom: 1px solid #ddd;">{desc}</td>
        </tr>
        """
    
    html += """
        </tbody>
    </table>
    </div>
    """
    
    return html   