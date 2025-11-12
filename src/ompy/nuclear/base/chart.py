from .nuclide import Nuclide
from IPython.display import HTML
import json
import colorsys
from typing import Callable, Dict, List, Optional, Union

def default_tooltip(element: Nuclide) -> str:
    """Default tooltip function that returns simple HTML with element information."""
    return f"""
        <strong>{element.symbol}</strong><br>
        Z: {element.Z}<br>
        N: {element.N}<br>
        A: {element.A}
    """

def get_default_colors(group_names: List[str]) -> Dict[str, str]:
    """Generate a set of distinct colors for groups."""
    default_colors = {
        'base': '#d9d9d9',  # Light gray
    }
    
    # Predefined colors for first few groups
    predefined = [
        '#ff9999',  # Light red
        '#99ccff',  # Light blue
        '#99ff99',  # Light green
        '#ffcc99',  # Light orange
        '#cc99ff',  # Light purple
        '#ffff99',  # Light yellow
        '#ff99cc',  # Light pink
        '#99ffff',  # Light cyan
    ]
    
    for i, group in enumerate(group_names):
        if group == 'base':
            continue
        
        if i < len(predefined):
            default_colors[group] = predefined[i]
        else:
            # Generate colors using HSV color space for remaining groups
            h = (i * 0.618033988749895) % 1  # Golden ratio conjugate
            s = 0.5
            v = 0.95
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            default_colors[group] = f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
    
    return default_colors

def draw_chart(base: list[Nuclide],
               groups: dict[str, list[Nuclide]],
               colors: dict[str, str] = None,
               tooltip: Callable[[Nuclide], str] = None):
    """
    Draw a chart of nuclides using HTML tables with multiple grouping.
    
    Parameters:
    -----------
    base : List[Element]
        List of all base nuclides to display
    groups : Dict[str, List[Element]]
        Dictionary with group names as keys and lists of elements as values
    colors : Dict[str, str], optional
        Dictionary mapping group names to color codes (hex, rgb, etc.)
    tooltip : Callable[[Element], str], optional
        Function that takes an Element and returns HTML string for tooltip
        
    Returns:
    --------
    IPython.display.HTML
        HTML display object for the Jupyter notebook
    """
    # Set default tooltip function if none provided
    if tooltip is None:
        tooltip = default_tooltip
    
    # Initialize groups for grouping elements
    group_names = list(groups.keys())
    
    # Set default colors if none provided
    if colors is None:
        colors = get_default_colors(group_names)
    else:
        # Ensure all groups have a color, use defaults for missing ones
        default_colors = get_default_colors(group_names)
        for group in group_names:
            if group not in colors:
                colors[group] = default_colors.get(group, '#cccccc')
    
    # Ensure 'base' has a color
    if 'base' not in colors:
        colors['base'] = '#d9d9d9'  # Light gray
        
    # Process all elements
    all_elements = {}
    
    # First add base elements
    for elem in base:
        key = f"{elem.Z}-{elem.N}"
        all_elements[key] = {
            'element': elem,
            'group': 'base'
        }
    
    # Then add elements from each group
    for group_name, elements in groups.items():
        for elem in elements:
            key = f"{elem.Z}-{elem.N}"
            all_elements[key] = {
                'element': elem,
                'group': group_name
            }
    
    # Find min and max values for Z and N across all elements
    all_z = [e['element'].Z for e in all_elements.values()]
    all_n = [e['element'].N for e in all_elements.values()]
    
    min_z = min(all_z) if all_z else 0
    max_z = max(all_z) if all_z else 10
    min_n = min(all_n) if all_n else 0
    max_n = max(all_n) if all_n else 10
    
    # Add padding
    min_z = max(0, min_z - 1)
    min_n = max(0, min_n - 1)
    max_z += 1
    max_n += 1
    
    # Generate the HTML table
    html = """
    <style>
        .nuclide-chart {
            font-family: Arial, sans-serif;
            border-collapse: collapse;
            margin: 20px 0;
            overflow: auto;
        }
        .nuclide-chart th {
            background-color: #f2f2f2;
            border: 1px solid #ddd;
            padding: 8px;
            text-align: center;
            position: sticky;
            top: 0;
            z-index: 10;
        }
        .nuclide-chart td {
            border: 1px solid #ddd;
            width: 60px;
            height: 60px;
            text-align: center;
            font-size: 12px;
            padding: 0;
            position: relative;
        }
        .element-symbol {
            font-weight: bold;
            font-size: 14px;
        }
        .mass-number {
            position: absolute;
            font-size: 10px;
            left: 50%;
            top: 6px;
            transform: translateX(-50%);
        }
        .atomic-number {
            position: absolute;
            font-size: 10px;
            left: 30%;
            bottom: 6px;
        }
        .neutron-number {
            position: absolute;
            font-size: 10px;
            right: 30%;
            bottom: 6px;
        }
        .nuclide-tooltip {
            visibility: hidden;
            width: 200px;
            background-color: #555;
            color: #fff;
            text-align: left;
            border-radius: 6px;
            padding: 8px;
            position: absolute;
            z-index: 1000;
            bottom: 125%;
            left: 50%;
            margin-left: -100px;
            opacity: 0;
            transition: opacity 0.3s;
            pointer-events: none;
        }
        .nuclide-cell:hover .nuclide-tooltip {
            visibility: visible;
            opacity: 1;
        }
        .z-label {
            font-weight: bold;
            background-color: #f2f2f2;
            position: sticky;
            left: 0;
            z-index: 5;
        }
        .chart-container {
            max-height: 600px;
            overflow: auto;
            border: 1px solid #ccc;
        }
        .legend {
            margin-top: 10px;
            margin-bottom: 20px;
            display: flex;
            flex-wrap: wrap;
        }
        .legend-item {
            display: flex;
            align-items: center;
            margin-right: 20px;
            margin-bottom: 5px;
        }
        .legend-color {
            display: inline-block;
            width: 20px;
            height: 20px;
            margin-right: 5px;
            border: 1px solid #999;
        }
        .chart-title {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
            text-align: center;
        }
        .axis-label {
            font-weight: bold;
            text-align: center;
            margin: 10px 0;
        }
    </style>
    
    <div class="chart-title">Chart of Nuclides</div>
    
    <div class="axis-label">Neutrons (N) →</div>
    
    <div class="chart-container">
        <table class="nuclide-chart">
            <tr>
                <th>Z\\N</th>
    """
    
    # Add N (neutron) headers
    for n in range(min_n, max_n + 1):
        html += f"<th>{n}</th>"
    
    html += "</tr>"
    
    # Add rows for each Z (proton)
    for z in range(min_z, max_z + 1):
        html += f"""
            <tr>
                <td class="z-label">{z}</td>
        """
        
        # Add cells for each N (neutron)
        for n in range(min_n, max_n + 1):
            key = f"{z}-{n}"
            if key in all_elements:
                elem_data = all_elements[key]
                elem = elem_data['element']
                group = elem_data['group']
                bg_color = colors.get(group, colors['base'])
                
                # Get tooltip HTML from the tooltip function
                tooltip_html = tooltip(elem)
                
                html += f"""
                <td class="nuclide-cell" style="background-color: {bg_color};">
                    <span class="mass-number">{elem.A}</span>
                    <span class="atomic-number">{elem.Z}</span>
                    <span class="neutron-number">{elem.N}</span>
                    <span class="element-symbol">{elem.symbol}</span>
                    <div class="nuclide-tooltip">
                        {tooltip_html}
                    </div>
                </td>
                """
            else:
                html += "<td></td>"
        
        html += "</tr>"
    
    html += """
        </table>
    </div>
    
    <div class="axis-label">← Protons (Z)</div>
    
    <div class="legend">
    """
    
    # Add legend items
    for group, color in colors.items():
        html += f"""
        <div class="legend-item">
            <div class="legend-color" style="background-color: {color};"></div>
            <div>{group}</div>
        </div>
        """
    
    html += """
    </div>
    
    <p><small>Note: Hover over a nuclide to see detailed information.</small></p>
    """
    
    return html