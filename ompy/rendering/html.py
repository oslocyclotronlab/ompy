import numpy as np 
def _create_array_header(
    array: np.ndarray,
    name: str,
    description: str,
    unique_id: str,
    max_preview_items: int,
    precision: int,
    initially_expanded: bool
) -> str:
    """
    Create the header portion of a collapsible array display.
    
    Parameters:
    -----------
    array : np.ndarray
        The array to display
    name : str
        The name/label for the array
    description : str
        Description of what the array represents
    unique_id : str
        Unique identifier for this array display
    max_preview_items : int
        Number of items to show in the preview when collapsed
    precision : int
        Number of decimal places to display for floating point numbers
    initially_expanded : bool
        Whether the array content should be expanded by default
        
    Returns:
    --------
    str
        HTML string for the header
    """
    # Format a preview of the array
    if array.size <= max_preview_items:
        preview = str(array)
    else:
        if array.ndim == 1:
            # For 1D arrays, show first few and last item
            first_items = [f"{x:.{precision}f}" if isinstance(x, float) else str(x) 
                          for x in array[:max_preview_items-1]]
            preview = "[" + ", ".join(first_items) + ", ..., "
            preview += f"{array[-1]:.{precision}f}" if isinstance(array[-1], float) else str(array[-1])
            preview += "]"
        else:
            # For multi-dimensional arrays, just show the shape
            preview = f"Array with shape {array.shape}"
    
    # Initial display state
    toggle_symbol = "▼" if initially_expanded else "▶"
    
    # Create the header HTML
    header = f"""
    <div class="array-header" style="display: flex; align-items: center; cursor: pointer; padding: 8px; background-color: #f2f2f2; border-top-left-radius: 4px; border-top-right-radius: 4px;"
         onclick="toggleArray_{unique_id}()">
        <span class="toggle-btn" id="toggle-btn-{unique_id}" style="margin-right: 8px; font-size: 12px; transition: transform 0.2s; display: inline-block; width: 12px;">{toggle_symbol}</span>
        <span style="font-weight: bold;">{name}</span>
        <span style="margin-left: 8px; color: #0366d6; font-size: 0.9em;">[{array.size} elements, shape: {array.shape}]</span>
        <span style="margin-left: 10px; font-family: monospace; font-size: 0.9em; color: #6c757d;">{preview}</span>
        {f'<span style="margin-left: 12px; color: #6c757d; font-size: 0.9em;">{description}</span>' if description else ''}
    </div>
    """
    
    return header


def _format_1d_array_html(
    array: np.ndarray,
    precision: int,
    max_rows: int = 100
) -> str:
    """
    Format a 1D numpy array as an HTML table with row limit.
    
    Parameters:
    -----------
    array : np.ndarray
        1D array to display
    precision : int
        Number of decimal places for floating point numbers
    max_rows : int, default=100
        Maximum number of rows to display
        
    Returns:
    --------
    str
        HTML table representing the 1D array (limited to max_rows)
    """
    html = '<table style="width: 100%; border-collapse: collapse;">'
    html += '<thead><tr>'
    html += '<th style="text-align: left; padding: 4px; border-bottom: 1px solid #e0e0e0; background-color: #f8f9fa; width: 1%; white-space: nowrap;">Index</th>'
    html += '<th style="text-align: left; padding: 4px; border-bottom: 1px solid #e0e0e0; background-color: #f8f9fa; width: 99%;">Value</th>'
    html += '</tr></thead>'
    html += '<tbody>'
    
    # Determine if we need to show start/end sections with ellipsis in middle
    total_rows = array.size
    if total_rows <= max_rows:
        # Show all rows
        indices = range(total_rows)
    else:
        # Show first and last rows with ellipsis in the middle
        start_count = max_rows // 2
        end_count = max_rows - start_count - 1  # -1 for the ellipsis row
        indices = list(range(start_count)) + list(range(total_rows - end_count, total_rows))
    
    for idx, i in enumerate(indices):
        bg_color = '#f8f9fa' if idx % 2 == 0 else '#ffffff'
        value = array[i]
        
        if isinstance(value, float):
            formatted_value = f"{value:.{precision}f}"
        else:
            formatted_value = str(value)
            
        html += f'<tr style="background-color: {bg_color};">'
        html += f'<td style="padding: 3px; border-bottom: 1px solid #e0e0e0; color: #6c757d;">{i}</td>'
        html += f'<td style="padding: 3px; text-align: left; border-bottom: 1px solid #e0e0e0;">{formatted_value}</td>'
        html += '</tr>'
        
        # Add ellipsis row if we're at the end of the first section
        if total_rows > max_rows and idx == start_count - 1:
            html += f'<tr style="background-color: #f2f2f2;">'
            html += f'<td colspan="2" style="text-align: center; padding: 5px; border-bottom: 1px solid #e0e0e0; color: #6c757d;">... {total_rows - max_rows} more rows ...</td>'
            html += '</tr>'
    
    html += '</tbody></table>'
    return html


def _format_2d_array_html(
    array: np.ndarray,
    precision: int,
    max_rows: int = 50,
    max_cols: int = 20
) -> str:
    """
    Format a 2D numpy array as an HTML table with row/column limits.
    
    Parameters:
    -----------
    array : np.ndarray
        2D array to display
    precision : int
        Number of decimal places for floating point numbers
    max_rows : int, default=50
        Maximum number of rows to display
    max_cols : int, default=20
        Maximum number of columns to display
        
    Returns:
    --------
    str
        HTML table representing the 2D array (limited to max_rows and max_cols)
    """
    rows, cols = array.shape
    
    # Determine if we need to limit columns
    if cols <= max_cols:
        col_indices = range(cols)
        show_col_ellipsis = False
    else:
        # Show first and last columns with ellipsis in between
        start_cols = max_cols // 2
        end_cols = max_cols - start_cols - 1  # -1 for ellipsis column
        col_indices = list(range(start_cols)) + list(range(cols - end_cols, cols))
        show_col_ellipsis = True
    
    # Determine if we need to limit rows
    if rows <= max_rows:
        row_indices = range(rows)
        show_row_ellipsis = False
    else:
        # Show first and last rows with ellipsis in between
        start_rows = max_rows // 2
        end_rows = max_rows - start_rows - 1  # -1 for ellipsis row
        row_indices = list(range(start_rows)) + list(range(rows - end_rows, rows))
        show_row_ellipsis = True
    
    # Start the table
    html = '<table style="border-collapse: collapse;">'
    
    # Add column headers (indices)
    html += '<thead><tr><th style="padding: 4px; background-color: #f8f9fa; border: 1px solid #e0e0e0;"></th>'
    
    for col_idx, j in enumerate(col_indices):
        html += f'<th style="padding: 4px; background-color: #f8f9fa; border: 1px solid #e0e0e0;">{j}</th>'
        
        # Add ellipsis column if needed
        if show_col_ellipsis and col_idx == start_cols - 1:
            html += f'<th style="padding: 4px; background-color: #f8f9fa; border: 1px solid #e0e0e0;">...</th>'
    
    html += '</tr></thead><tbody>'
    
    # Add rows
    for row_idx, i in enumerate(row_indices):
        bg_color = '#f8f9fa' if row_idx % 2 == 0 else '#ffffff'
        html += f'<tr style="background-color: {bg_color};">'
        html += f'<td style="padding: 3px; border: 1px solid #e0e0e0; font-weight: bold;">{i}</td>'
        
        for col_idx, j in enumerate(col_indices):
            value = array[i, j]
            if isinstance(value, float):
                formatted_value = f"{value:.{precision}f}"
            else:
                formatted_value = str(value)
            html += f'<td style="padding: 3px; text-align: right; border: 1px solid #e0e0e0;">{formatted_value}</td>'
            
            # Add ellipsis column if needed
            if show_col_ellipsis and col_idx == start_cols - 1:
                html += f'<td style="padding: 3px; text-align: center; border: 1px solid #e0e0e0;">...</td>'
        
        html += '</tr>'
        
        # Add ellipsis row if needed
        if show_row_ellipsis and row_idx == start_rows - 1:
            html += f'<tr style="background-color: #f2f2f2;">'
            html += f'<td style="padding: 3px; border: 1px solid #e0e0e0; font-weight: bold;">...</td>'
            
            # Add cells for all columns (including ellipsis column if present)
            col_count = len(col_indices) + (1 if show_col_ellipsis else 0)
            html += f'<td colspan="{col_count}" style="text-align: center; padding: 5px; border: 1px solid #e0e0e0; color: #6c757d;">... {rows - max_rows} more rows ...</td>'
            html += '</tr>'
    
    html += '</tbody></table>'
    return html


def _format_nd_array_html(
    array: np.ndarray,
    precision: int,
    max_items: int = 50
) -> str:
    """
    Format a higher-dimensional numpy array (3D+) as HTML with item limit.
    
    Parameters:
    -----------
    array : np.ndarray
        ND array to display
    precision : int
        Number of decimal places for floating point numbers
    max_items : int, default=50
        Maximum number of items to show in the flattened view
        
    Returns:
    --------
    str
        HTML representation of the ND array
    """
    total_size = array.size
    
    # Array metadata section
    html = f'<div style="padding: 5px; margin-bottom: 10px; background-color: #f8f9fa;">'
    html += f'<strong>Array information:</strong><br>'
    html += f'Dimensions: {array.ndim}<br>'
    html += f'Shape: {array.shape}<br>'
    html += f'Size: {total_size:,} elements<br>'  # Format with commas for readability
    html += f'Data type: {array.dtype}<br>'
    
    # Add some statistical information for numerical arrays
    if np.issubdtype(array.dtype, np.number):
        try:
            html += f'<br><strong>Statistics:</strong><br>'
            html += f'Min: {np.min(array):.{precision}f}<br>'
            html += f'Max: {np.max(array):.{precision}f}<br>'
            html += f'Mean: {np.mean(array):.{precision}f}<br>'
            html += f'Std Dev: {np.std(array):.{precision}f}'
        except:
            # In case of overflow or other computational issues
            pass
    
    html += '</div>'
    
    # Flattened view section
    html += '<div style="padding: 5px; margin-bottom: 10px;">'
    html += f'<strong>Flattened view (showing {min(max_items, total_size)} of {total_size:,} elements):</strong>'
    html += '</div>'
    
    flat_array = array.flatten()
    
    # Determine if we need to show start/end sections with ellipsis in middle
    if total_size <= max_items:
        # Show all items
        indices = range(total_size)
        show_ellipsis = False
    else:
        # Show first and last items with ellipsis in the middle
        start_count = max_items // 2
        end_count = max_items - start_count
        indices = list(range(start_count)) + list(range(total_size - end_count, total_size))
        show_ellipsis = True
    
    # Create the table for the flattened view
    html += '<table style="width: 100%; border-collapse: collapse;">'
    html += '<thead><tr><th style="text-align: left; padding: 4px; border-bottom: 1px solid #e0e0e0; background-color: #f8f9fa;">Index</th>'
    html += '<th style="text-align: right; padding: 4px; border-bottom: 1px solid #e0e0e0; background-color: #f8f9fa;">Value</th></tr></thead>'
    html += '<tbody>'
    
    for idx, i in enumerate(indices):
        bg_color = '#f8f9fa' if idx % 2 == 0 else '#ffffff'
        value = flat_array[i]
        
        if isinstance(value, float):
            formatted_value = f"{value:.{precision}f}"
        else:
            formatted_value = str(value)
            
        html += f'<tr style="background-color: {bg_color};">'
        html += f'<td style="padding: 3px; border-bottom: 1px solid #e0e0e0; color: #6c757d;">{i}</td>'
        html += f'<td style="padding: 3px; text-align: right; border-bottom: 1px solid #e0e0e0;">{formatted_value}</td>'
        html += '</tr>'
        
        # Add ellipsis row if we're at the end of the first section
        if show_ellipsis and idx == start_count - 1:
            html += f'<tr style="background-color: #f2f2f2;">'
            html += f'<td colspan="2" style="text-align: center; padding: 5px; border-bottom: 1px solid #e0e0e0; color: #6c757d;">... {total_size - max_items:,} more elements ...</td>'
            html += '</tr>'
    
    html += '</tbody></table>'
    return html


def _create_toggle_script(unique_id: str) -> str:
    """
    Create the JavaScript toggle function for a collapsible array.
    
    Parameters:
    -----------
    unique_id : str
        Unique identifier for this array display
        
    Returns:
    --------
    str
        JavaScript code for the toggle function
    """
    script = f"""
    <script>
    function toggleArray_{unique_id}() {{
        const content = document.getElementById('array-content-{unique_id}');
        const toggleBtn = document.getElementById('toggle-btn-{unique_id}');
        
        if (content.style.display === 'none') {{
            content.style.display = 'block';
            toggleBtn.textContent = '▼';
        }} else {{
            content.style.display = 'none';
            toggleBtn.textContent = '▶';
        }}
    }}
    </script>
    """
    return script


def collapsible(
    array: np.ndarray,
    name: str,
    description: str = None,
    initially_expanded: bool = False,
    max_preview_items: int = 3,
    precision: int = 4,
    max_height: str = "300px",
    max_rows_1d: int = 100,
    max_rows_2d: int = 50,
    max_cols_2d: int = 20,
    max_items_nd: int = 50
) -> str:
    """
    Generate HTML for a collapsible display of a numpy array, with limits to handle large arrays efficiently.
    
    Parameters:
    -----------
    array : np.ndarray
        The array to display
    name : str
        The name/label for the array
    description : str, optional
        Description of what the array represents
    initially_expanded : bool, default=False
        Whether the array content should be expanded by default
    max_preview_items : int, default=3
        Number of items to show in the preview when collapsed
    precision : int, default=4
        Number of decimal places to display for floating point numbers
    max_height : str, default="300px"
        Maximum height of the expanded content (CSS value)
    max_rows_1d : int, default=100
        Maximum number of rows to display for 1D arrays
    max_rows_2d : int, default=50
        Maximum number of rows to display for 2D arrays
    max_cols_2d : int, default=20
        Maximum number of columns to display for 2D arrays
    max_items_nd : int, default=50
        Maximum number of items to display for higher-dimensional arrays
        
    Returns:
    --------
    str
        HTML string for the collapsible array display
    """
    import uuid
    import numpy as np
    
    # Generate a unique ID for this instance
    unique_id = str(uuid.uuid4()).replace('-', '')
    
    # Initial display state
    display_style = "block" if initially_expanded else "none"
    
    # Create container and header
    html = f"""
    <div class="array-container" style="margin: 8px 0; border: 1px solid #e0e0e0; border-radius: 4px; background-color: #ffffff;">
        {_create_array_header(array, name, description, unique_id, max_preview_items, precision, initially_expanded)}
        
        <div id="array-content-{unique_id}" class="array-content" style="display: {display_style}; padding: 10px; max-height: {max_height}; overflow-y: auto; font-family: monospace;">
    """
    
    # Add appropriate content based on array dimensions
    if array.ndim == 1:
        html += _format_1d_array_html(array, precision, max_rows_1d)
    elif array.ndim == 2:
        html += _format_2d_array_html(array, precision, max_rows_2d, max_cols_2d)
    else:
        html += _format_nd_array_html(array, precision, max_items_nd)
    
    # Close the content div and add the toggle script
    html += """
        </div>
    </div>
    """
    
    html += _create_toggle_script(unique_id)
    
    return html

    
def table(data: list[tuple[str, str]], color="#f5f5f5", alignments=None, styles=None) -> str:
    """
    Generate an HTML table representation from a list of tuples with key-value pairs.
    
    Parameters:
    -----------
    data : list of tuples
        List of (key, value) tuples to display in the table
    color : str, optional
        Background color for the table header. Default is "#f5f5f5" (light gray)
    alignments : tuple, optional
        Text alignments for key and value columns ('left', 'center', 'right').
        Default is ('left', 'left')
    styles : tuple, optional
        Custom CSS styles for key and value columns
        
    Returns:
    --------
    str
        HTML representation of the table
    """
    if not data:
        return "<p>Empty table</p>"
    
    # Set default alignments if not provided
    if alignments is None:
        alignments = ('left', 'left')
    
    # Set default styles if not provided
    if styles is None:
        styles = ('', '')
    
    # Build the HTML
    html = '<table style="border-collapse: collapse; width: 100%;">\n'
    html += '  <thead>\n'
    html += f'    <tr style="background-color: {color};">\n'
    html += f'      <th style="text-align: {alignments[0]}; padding: 8px; {styles[0]}">Key</th>\n'
    html += f'      <th style="text-align: {alignments[1]}; padding: 8px; {styles[1]}">Value</th>\n'
    html += '    </tr>\n'
    html += '  </thead>\n'
    html += '  <tbody>\n'
    
    # Add data rows
    for i, (key, value) in enumerate(data):
        bg_color = '#f9f9f9' if i % 2 == 0 else 'white'
        html += f'    <tr style="background-color: {bg_color};">\n'
        html += f'      <td style="text-align: {alignments[0]}; padding: 8px; {styles[0]}">{key}</td>\n'
        html += f'      <td style="text-align: {alignments[1]}; padding: 8px; {styles[1]}">{value}</td>\n'
        html += '    </tr>\n'
    
    html += '  </tbody>\n'
    html += '</table>'
    
    return html


def collapse(content: str, title: str) -> str:
    """
    Create a collapsible HTML section that is hidden by default.
    
    Parameters:
    -----------
    content : str
        HTML content to be placed inside the collapsible section
    title : str
        Title to display in the header of the collapsible section
        
    Returns:
    --------
    str
        HTML string with collapsible section
    """
    html = f"""
    <details>
        <summary style="cursor: pointer; font-weight: bold;">{title}</summary>
        <div style="margin-top: 8px; margin-left: 15px;">
            {content}
        </div>
    </details>
    """
    return html
# Example usage:
# html_table([("Range", "X = 4.004"), ("Label", "Excitation Energy")])