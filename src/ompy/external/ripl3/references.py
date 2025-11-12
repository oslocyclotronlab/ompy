from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import re

from .stubs import RIPL3_RESONANCE_SPACING_PATH


@dataclass
class Reference:
    """Reference information parsed from RIPL3 documentation."""
    code: str  # Reference code (e.g., "81M", "06M")
    authors: str  # Author names
    title: str  # Title of the work
    journal: str  # Journal or publication venue
    year: str  # Publication year
    volume: str  # Volume number (if applicable)
    pages: str  # Page numbers (if applicable)
    publisher: str  # Publisher (if applicable)
    location: str  # Publication location (if applicable)
    doi: str  # DOI (if available)
    url: str  # URL (if available)
    
    def __post_init__(self):
        # Trim all strings
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if isinstance(value, str):
                setattr(self, field, value.strip())
            if getattr(self, field) == '':
                setattr(self, field, None)

    def _repr_html_(self):
        """HTML representation for Jupyter notebook display."""
        html = f"""
        <div style="font-family: Arial, sans-serif; max-width: 800px; margin: 10px; padding: 15px; border: 1px solid #ddd; border-radius: 5px; background-color: #f9f9f9;">
            <h3 style="color: #333; border-bottom: 1px solid #ddd; padding-bottom: 5px;">Reference: {self.code}</h3>
            
            <div style="margin-top: 10px;">
                <p style="margin: 5px 0;"><strong>Citation:</strong> {self.title}</p>
            </div>
        </div>
        """
        return html


# Reference parsing functions
_REFERENCE_CACHE: Dict[str, Reference] = {}


def parse_references_from_readme() -> Dict[str, Reference]:
    """Parse references from the resonance_spacing_readme.txt file.
    
    Returns
    -------
    Dict[str, Reference]
        Dictionary mapping reference codes to Reference objects
    """
    if _REFERENCE_CACHE:
        return _REFERENCE_CACHE
    
    readme_path = RIPL3_RESONANCE_SPACING_PATH / "resonance_spacing_readme.txt"
    references = {}
    
    with readme_path.open() as f:
        content = f.read()
    
    # Find the References section
    ref_section_start = content.find("References")
    if ref_section_start == -1:
        return references
    
    ref_section = content[ref_section_start:]
    
    # Parse each reference
    lines = ref_section.split('\n')
    current_ref = None
    current_text = ""
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith("References") or line.startswith("----------"):
            continue
            
        # Check if this line starts a new reference (pattern: XXY where XX is year, Y is letter)
        ref_match = re.match(r'^(\d{2}[A-Z])\s+(.+)', line)
        if ref_match:
            # Save previous reference if exists
            if current_ref and current_text:
                _parse_reference_text(current_text, current_ref)
                references[current_ref['code']] = Reference(**current_ref)
            
            # Start new reference
            code = ref_match.group(1)
            rest = ref_match.group(2)
            current_ref = {
                'code': code,
                'authors': '',
                'title': '',
                'journal': '',
                'year': '',
                'volume': '',
                'pages': '',
                'publisher': '',
                'location': '',
                'doi': '',
                'url': ''
            }
            current_text = rest
        elif current_ref and line:
            # Continue the current reference
            current_text += " " + line
    
    # Don't forget the last reference
    if current_ref and current_text:
        _parse_reference_text(current_text, current_ref)
        references[current_ref['code']] = Reference(**current_ref)
    
    # Cache the results
    _REFERENCE_CACHE.update(references)
    return references


def _parse_reference_text(text: str, ref_dict: Dict[str, str]) -> None:
    """Parse the complete reference text and update the reference dictionary."""
    # Clean up the text
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Just store the entire citation as the title
    ref_dict['title'] = text


def get_reference_by_code(code: str) -> Optional[Reference]:
    """Get a reference by its code.
    
    Parameters
    ----------
    code : str
        The reference code (e.g., "81M", "06M", "4I", "5I")
        
    Returns
    -------
    Optional[Reference]
        The reference information, or None if not found
    """
    if not _REFERENCE_CACHE:
        parse_references_from_readme()
    
    # First try the code as-is
    ref = _REFERENCE_CACHE.get(code)
    if ref:
        return ref
    
    # If not found and it's a 2-digit code, try with leading zero
    if len(code) == 2 and code[0].isdigit() and code[1].isalpha():
        padded_code = f"0{code}"
        return _REFERENCE_CACHE.get(padded_code)
    
    # If not found and it's a 4-digit code, try without leading zero
    if len(code) == 3 and code[0] == '0' and code[1].isdigit() and code[2].isalpha():
        unpadded_code = code[1:]
        return _REFERENCE_CACHE.get(unpadded_code)
    
    return None


def get_all_references() -> Dict[str, Reference]:
    """Get all parsed references.
    
    Returns
    -------
    Dict[str, Reference]
        Dictionary mapping reference codes to Reference objects
    """
    if not _REFERENCE_CACHE:
        parse_references_from_readme()
    
    return _REFERENCE_CACHE.copy()
