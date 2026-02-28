#!/usr/bin/env python3
"""
Parse Jupyter Notebook HTML export and extract cell content (input/output).
Stores the extracted content in a clean markdown format.
Also extracts plots/images from cell outputs.
"""

from bs4 import BeautifulSoup
import base64
import os
import sys
from pathlib import Path


def parse_jupyter_html(html_file_path, output_file_path=None, extract_images=True):
    """
    Parse a Jupyter HTML export file and extract cell content.

    Args:
        html_file_path: Path to the HTML file
        output_file_path: Optional path to save the extracted content.
                         If None, uses the same name as input with .md extension
        extract_images: Whether to extract and save images from outputs

    Returns:
        str: The formatted content of all cells
    """
    html_path = Path(html_file_path)

    if not html_path.exists():
        raise FileNotFoundError(f"HTML file not found: {html_file_path}")

    # Read the HTML content
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    soup = BeautifulSoup(html_content, 'html.parser')
    notebook_content = []

    # Create output directory for images
    if extract_images:
        images_dir = html_path.parent / f"{html_path.stem}_images"
        images_dir.mkdir(exist_ok=True)
        print(f"Images will be saved to: {images_dir}")

    # Try different cell selectors for various Jupyter HTML formats
    cells = []

    # Format 1: Newer JupyterLab format (jp-Cell)
    jp_cells = soup.find_all('div', class_='jp-Cell')
    if jp_cells:
        cells = jp_cells
        format_type = "jupyterlab"
    else:
        # Format 2: Classic nbconvert format (div.cell)
        classic_cells = soup.find_all('div', class_='cell')
        if classic_cells:
            cells = classic_cells
            format_type = "nbconvert"
        else:
            # Format 3: Try finding by other patterns
            code_cells = soup.find_all('div', class_='code_cell')
            text_cells = soup.find_all('div', class_='text_cell')
            if code_cells or text_cells:
                # Combine and sort by position
                all_cells = code_cells + text_cells
                all_cells.sort(key=lambda x: x.sourceline if hasattr(x, 'sourceline') else 0)
                cells = all_cells
                format_type = "mixed"

    if not cells:
        raise ValueError("No cells found in the HTML file. The format may not be recognized.")

    print(f"Found {len(cells)} cells in the notebook (format: {format_type}).")

    # Track images across all cells
    image_counter = 0
    all_images = []

    for i, cell in enumerate(cells, 1):
        cell_data = {"cell_number": i, "input": "", "output": "", "cell_type": "code", "images": []}

        # Determine cell type
        if 'text_cell' in cell.get('class', []) or cell.find('div', class_='text_cell'):
            cell_data["cell_type"] = "markdown"
        elif 'jp-MarkdownCell' in cell.get('class', []):
            cell_data["cell_type"] = "markdown"
        else:
            cell_data["cell_type"] = "code"

        # Extract Input based on format
        input_text = ""

        if format_type == "jupyterlab":
            # JupyterLab format
            input_area = cell.find('div', class_='jp-InputArea')
            if input_area:
                pre_tags = input_area.find_all('pre')
                if pre_tags:
                    input_text = "\n".join([pre.get_text(strip=False) for pre in pre_tags])
                else:
                    input_text = input_area.get_text(strip=False)
        else:
            # Classic nbconvert format
            # Input is in div.input_area > pre or directly in the cell
            input_area = cell.find('div', class_='input_area')
            if input_area:
                pre = input_area.find('pre')
                if pre:
                    input_text = pre.get_text(strip=False)
            else:
                # For text cells, look for text_cell_render
                text_render = cell.find('div', class_='text_cell_render')
                if text_render:
                    input_text = text_render.get_text(strip=False)

        cell_data["input"] = input_text

        # Extract Output based on format
        output_text = ""
        cell_images = []

        if format_type == "jupyterlab":
            output_area = cell.find('div', class_='jp-OutputArea')
            if output_area:
                outputs = []
                output_pre = output_area.find_all('pre')
                for pre in output_pre:
                    outputs.append(pre.get_text(strip=False))

                # Extract images from output
                if extract_images:
                    img_tags = output_area.find_all('img')
                    for img in img_tags:
                        img_info = process_image(img, images_dir, image_counter, html_path.parent)
                        if img_info:
                            cell_images.append(img_info)
                            image_counter += 1

                output_text = "\n".join(outputs)
        else:
            # Classic nbconvert format
            # Output is in div.output_area or div.output
            output_area = cell.find('div', class_='output_area')
            if not output_area:
                output_area = cell.find('div', class_='output')

            if output_area:
                outputs = []

                # Text outputs
                output_pre = output_area.find_all('pre')
                for pre in output_pre:
                    outputs.append(pre.get_text(strip=False))

                # Stream outputs
                output_stream = output_area.find_all('div', class_='stream')
                for stream in output_stream:
                    outputs.append(stream.get_text(strip=False))

                # Error outputs
                output_error = output_area.find_all('div', class_='error')
                for error in output_error:
                    outputs.append(error.get_text(strip=False))

                # Extract images from output
                if extract_images:
                    img_tags = output_area.find_all('img')
                    for img in img_tags:
                        img_info = process_image(img, images_dir, image_counter, html_path.parent)
                        if img_info:
                            cell_images.append(img_info)
                            image_counter += 1

                output_text = "\n".join(outputs)

        cell_data["output"] = output_text
        cell_data["images"] = cell_images
        all_images.extend(cell_images)

        # Format for LLM - clean markdown format
        formatted_string = f"### Cell {i} [Type: {cell_data['cell_type'].capitalize()}]\n\n"

        if cell_data['input'].strip():
            formatted_string += f"**Input:**\n```\n{cell_data['input'].strip()}\n```\n"

        if cell_data['output'].strip():
            formatted_string += f"\n**Output:**\n```\n{cell_data['output'].strip()}\n```\n"

        # Add image references
        if cell_images:
            formatted_string += f"\n**Plots/Images ({len(cell_images)}):**\n"
            for img_info in cell_images:
                formatted_string += f"- `{img_info['filename']}` ({img_info['format']})\n"
                # For markdown files, we can include the image
                rel_path = img_info['relative_path']
                formatted_string += f"  ![{img_info['filename']}]({rel_path})\n"

        notebook_content.append(formatted_string)

    # Join all cells with separator
    final_content = "\n---\n".join(notebook_content)

    # Add summary at the top
    summary = f"""# Notebook Analysis Summary

- **Total Cells:** {len(cells)}
- **Images/Plots Extracted:** {len(all_images)}
- **Image Directory:** `{images_dir.name if extract_images else 'N/A'}`

---

"""

    final_content = summary + final_content

    # Determine output path
    if output_file_path is None:
        output_path = html_path.with_suffix('.md')
    else:
        output_path = Path(output_file_path)

    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(final_content)

    print(f"\nExtracted content saved to: {output_path}")
    print(f"Total cells processed: {len(cells)}")
    print(f"Total images extracted: {len(all_images)}")

    return final_content


def process_image(img_tag, images_dir, counter, base_path):
    """
    Process an image tag and save the image to disk.

    Args:
        img_tag: BeautifulSoup img tag
        images_dir: Directory to save images
        counter: Image counter for naming
        base_path: Base path for calculating relative paths

    Returns:
        dict with image info or None if failed
    """
    src = img_tag.get('src', '')

    if not src:
        return None

    try:
        # Determine image format and data
        if src.startswith('data:image'):
            # Base64 encoded image
            # Format: data:image/png;base64,iVBORw0KGgo...
            header, data = src.split(',', 1)
            img_format = header.split('/')[1].split(';')[0] if '/' in header else 'png'
            img_data = base64.b64decode(data)
        elif src.startswith('http'):
            # External URL - skip or download
            return None
        else:
            # Relative path - skip for now
            return None

        # Create filename
        filename = f"plot_{counter:03d}.{img_format}"
        filepath = images_dir / filename

        # Save image
        with open(filepath, 'wb') as f:
            f.write(img_data)

        # Calculate relative path for markdown
        rel_path = filepath.relative_to(base_path)

        return {
            'filename': filename,
            'filepath': str(filepath),
            'relative_path': str(rel_path),
            'format': img_format
        }

    except Exception as e:
        print(f"Warning: Failed to process image {counter}: {e}")
        return None


def main():
    """Main entry point."""
    # Default HTML file in current directory
    default_html = "/Users/sarveshmani/demo/Another copy of AIML_ML_Project_Full_Code_Notebook.html"

    # Check if file path provided as argument
    if len(sys.argv) > 1:
        html_file = sys.argv[1]
    else:
        html_file = default_html

    # Optional output path
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        content = parse_jupyter_html(html_file, output_file, extract_images=True)
        print("\n" + "="*50)
        print("Preview of first 3000 characters:")
        print("="*50)
        print(content[:3000])
        print("\n... [truncated for display] ...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
