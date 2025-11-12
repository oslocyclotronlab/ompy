from typing import Dict
from pathlib import Path
from . import Matrix

# Project examples from data/projects/
PROJECT_EXAMPLES = {
    'cd111': {
        'matrix': "../../data/projects/1_Cd111/ex_eg_111Cd.m",
        'counting': "../../data/projects/1_Cd111/counting.dat",
        'description': "111Cd excitation-gamma matrix"
    },
    'cd112': {
        'matrix': "../../data/projects/2_Cd112/ex_eg_112Cd.m",
        'counting': "../../data/projects/2_Cd112/counting.dat",
        'description': "112Cd excitation-gamma matrix"
    },
    'th233': {
        'matrix': "../../data/projects/3_Th233/ex_eg_233Th.m",
        'counting': "../../data/projects/3_Th233/counting.dat",
        'description': "233Th excitation-gamma matrix"
    },
    'np238': {
        'matrix': "../../data/projects/4_Np238/ex_eg_238Np.m",
        'counting': "../../data/projects/4_Np238/counting.dat",
        'description': "238Np excitation-gamma matrix"
    },
    'ge74': {
        'matrix': "../../data/projects/5_Ge74/ex_eg_74Ge.m",
        'counting': "../../data/projects/5_Ge74/counting.dat",
        'description': "74Ge excitation-gamma matrix"
    },
    'sn117': {
        'matrix': "../../data/projects/6_Sn117/ex_eg_117Sn.m",
        'counting': "../../data/projects/6_Sn117/counting.dat",
        'description': "117Sn excitation-gamma matrix"
    },
    'dy164': {
        'matrix': "../../data/projects/7_Dy164/ex_eg_164Dy.m",
        'counting': "../../data/projects/7_Dy164/counting.dat",
        'description': "164Dy excitation-gamma matrix"
    }
}

_MODULE_PATH = Path(__file__).resolve()


def get_path(path: str) -> Path:
    """Convert relative path to absolute path based on this module's location"""
    return (_MODULE_PATH.parent / path).resolve()


def list_projects() -> list[str]:
    """List project example names"""
    return list(PROJECT_EXAMPLES.keys())




def load_project(name: str) -> Matrix:
    """Load a project example matrix by name
    
    Args:
        name: Project name (e.g., 'cd111', 'ge74', 'dy164').
              Use `list_projects()` to see all available projects
        
    Returns:
        Matrix object loaded from the project file
        
    Example:
        >>> mat = load_project('cd111')
        >>> mat = load_project('ge74')
    """
    name = name.lower()
    if name not in PROJECT_EXAMPLES:
        available = ', '.join(list_projects())
        raise ValueError(f"Unknown project '{name}'. Available projects: {available}")
    
    path = get_path(PROJECT_EXAMPLES[name]['matrix'])
    return Matrix.from_path(path)


def get_project_info(name: str) -> Dict[str, str]:
    """Get information about a project example
    
    Args:
        name: Project name
        
    Returns:
        Dictionary containing project information (paths, description)
        
    Example:
        >>> info = get_project_info('cd111')
        >>> print(info['description'])
    """
    name = name.lower()
    if name not in PROJECT_EXAMPLES:
        available = ', '.join(list_projects())
        raise ValueError(f"Unknown project '{name}'. Available projects: {available}")
    
    info = PROJECT_EXAMPLES[name].copy()
    # Convert relative paths to absolute paths
    info['matrix_path'] = get_path(info['matrix'])
    info['counting_path'] = get_path(info['counting'])
    return info


def get_project_counting_path(name: str) -> Path:
    """Get the path to the counting.dat file for a project
    
    Args:
        name: Project name
        
    Returns:
        Absolute path to the counting.dat file
        
    Example:
        >>> path = get_project_counting_path('cd111')
    """
    name = name.lower()
    if name not in PROJECT_EXAMPLES:
        available = ', '.join(list_projects())
        raise ValueError(f"Unknown project '{name}'. Available projects: {available}")
    
    return get_path(PROJECT_EXAMPLES[name]['counting'])


