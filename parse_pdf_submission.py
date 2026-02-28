#!/usr/bin/env python3
"""
Parse PDF submissions (business reports, slide decks, executive summaries) into
layout-aware Markdown format.

Supports both text extraction (for text-only LLMs) and image conversion (for vision models).
Also extracts embedded images from PDF pages.
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import io


def parse_pdf_submission(pdf_path: str, output_path: Optional[str] = None,
                         convert_to_images: bool = False, dpi: int = 150,
                         extract_embedded_images: bool = True) -> str:
    """
    Parse a PDF submission into structured Markdown format.

    Args:
        pdf_path: Path to the PDF file
        output_path: Optional path to save the extracted content
        convert_to_images: If True, also convert pages to images (for vision models)
        dpi: DPI for image conversion (higher = better quality but larger files)
        extract_embedded_images: If True, extract embedded images from PDF

    Returns:
        str: The formatted content of all pages
    """
    try:
        import pdfplumber
    except ImportError:
        print("Error: pdfplumber not installed. Installing...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "pdfplumber", "-q"])
        import pdfplumber

    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    # Create output directories
    if convert_to_images:
        page_images_dir = pdf_file.parent / f"{pdf_file.stem}_pdf_pages"
        page_images_dir.mkdir(exist_ok=True)
        print(f"Page images will be saved to: {page_images_dir}")
    else:
        page_images_dir = None

    if extract_embedded_images:
        embedded_images_dir = pdf_file.parent / f"{pdf_file.stem}_images"
        embedded_images_dir.mkdir(exist_ok=True)
        print(f"Embedded images will be saved to: {embedded_images_dir}")
    else:
        embedded_images_dir = None

    formatted_content = []
    page_images_list = []
    all_embedded_images = []
    total_tables = 0

    print(f"Processing PDF: {pdf_file.name}")

    with pdfplumber.open(pdf_file) as pdf:
        print(f"Total pages: {len(pdf.pages)}")

        for i, page in enumerate(pdf.pages, 1):
            print(f"  Processing page {i}/{len(pdf.pages)}...", end="\r")

            # 1. Extract Text (preserving layout with tolerance settings)
            text = page.extract_text(x_tolerance=2, y_tolerance=2) or "[No text content]"

            # 2. Extract Tables (crucial for business reports)
            tables = page.extract_tables()
            table_str = ""
            if tables:
                total_tables += len(tables)
                table_str = "\n**Tables Found:**\n\n"
                for table_idx, table in enumerate(tables, 1):
                    table_str += f"*Table {table_idx}:*\n\n"
                    if table:
                        max_cols = max(len(row) for row in table if row)
                        for row_idx, row in enumerate(table):
                            if not row:
                                continue
                            cleaned_row = [str(cell).strip() if cell else "" for cell in row]
                            while len(cleaned_row) < max_cols:
                                cleaned_row.append("")
                            table_str += "| " + " | ".join(cleaned_row) + " |\n"
                            if row_idx == 0:
                                separator = "|" + "|".join([" --- " for _ in range(max_cols)]) + "|"
                                table_str += separator + "\n"
                    table_str += "\n"

            # 3. Extract embedded images from the page
            page_images_extracted = []
            if extract_embedded_images:
                page_images_extracted = extract_images_from_page(
                    pdf_file, i, embedded_images_dir, all_embedded_images
                )
                all_embedded_images.extend(page_images_extracted)

            # Create image note
            image_note = ""
            if page.images:
                image_note = f"\n**Visual Elements:**\n[Detected {len(page.images)} image(s)/chart(s) on this page]\n"

            # Add extracted images info
            extracted_img_note = ""
            if page_images_extracted:
                extracted_img_note = f"\n**Extracted Images ({len(page_images_extracted)}):**\n"
                for img_info in page_images_extracted:
                    rel_path = img_info['relative_path']
                    extracted_img_note += f"- `{img_info['filename']}` ({img_info['width']}x{img_info['height']})\n"
                    extracted_img_note += f"  ![{img_info['filename']}]({rel_path})\n"

            # 4. Convert page to image if requested
            img_info = ""
            if convert_to_images:
                img_filename = convert_page_to_image_plumber(page, page_images_dir, i, dpi)
                if img_filename:
                    img_rel_path = f"{page_images_dir.name}/{img_filename}"
                    page_images_list.append(img_rel_path)
                    img_info = f"\n**Page Image:** `![Page {i}]({img_rel_path})`\n"

            # 5. Format page content for LLM
            page_content = f"## Page {i}\n\n"
            page_content += f"**Text Content:**\n```\n{text}\n```\n"

            if table_str:
                page_content += f"{table_str}\n"

            if image_note:
                page_content += f"{image_note}\n"

            if extracted_img_note:
                page_content += f"{extracted_img_note}\n"

            if img_info:
                page_content += f"{img_info}\n"

            formatted_content.append(page_content)

    print(f"\nProcessing complete!")
    print(f"  Total tables extracted: {total_tables}")
    print(f"  Total embedded images extracted: {len(all_embedded_images)}")

    # Add summary at the top
    summary = f"""# PDF Submission Analysis

## Document Summary
- **File:** {pdf_file.name}
- **Total Pages:** {len(pdf.pages)}
- **Tables Found:** {total_tables}
- **Page Images:** {len(page_images_list) if convert_to_images else 'N/A (not generated)'}
- **Embedded Images Extracted:** {len(all_embedded_images)}

---

"""

    final_content = summary + "\n---\n".join(formatted_content)

    # Save to output file
    if output_path is None:
        output_file = pdf_file.with_suffix('.md')
    else:
        output_file = Path(output_path)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(final_content)

    print(f"\nExtracted content saved to: {output_file}")

    return final_content


def extract_images_from_page(pdf_path: Path, page_num: int, images_dir: Path, existing_images: list) -> List[Dict]:
    """
    Extract embedded images from a PDF page using PyMuPDF (fitz).

    Args:
        pdf_path: Path to PDF file
        page_num: Page number (1-indexed)
        images_dir: Directory to save images
        existing_images: List of already extracted images to avoid duplicates

    Returns:
        List of dicts with image info
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("\nWarning: PyMuPDF not installed. Cannot extract embedded images.")
        return []

    extracted = []

    try:
        doc = fitz.open(pdf_path)
        page = doc[page_num - 1]  # 0-indexed

        # Get images from page
        image_list = page.get_images(full=True)

        for img_index, img in enumerate(image_list, start=1):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            # Generate unique filename
            img_filename = f"page{page_num:03d}_img{img_index:03d}.{image_ext}"
            img_path = images_dir / img_filename

            # Save image
            with open(img_path, "wb") as f:
                f.write(image_bytes)

            # Get image info
            try:
                from PIL import Image
                pil_img = Image.open(io.BytesIO(image_bytes))
                width, height = pil_img.size
            except:
                width, height = 0, 0

            rel_path = f"{images_dir.name}/{img_filename}"

            extracted.append({
                'filename': img_filename,
                'filepath': str(img_path),
                'relative_path': rel_path,
                'format': image_ext,
                'page': page_num,
                'width': width,
                'height': height
            })

        doc.close()

    except Exception as e:
        print(f"\nWarning: Error extracting images from page {page_num}: {e}")

    return extracted


def convert_page_to_image_plumber(page, images_dir: Path, page_num: int, dpi: int = 150) -> Optional[str]:
    """
    Convert a PDF page to an image using pdfplumber's built-in method.

    Args:
        page: pdfplumber page object
        images_dir: Directory to save the image
        page_num: Page number for naming
        dpi: Resolution for the image

    Returns:
        str: Filename of the saved image, or None if failed
    """
    try:
        from PIL import Image

        # Use pdfplumber's to_image method
        im = page.to_image(resolution=dpi)

        img_filename = f"page_{page_num:03d}.png"
        img_path = images_dir / img_filename

        # Save using PIL
        im.save(img_path)

        return img_filename

    except Exception as e:
        print(f"\nWarning: Could not convert page {page_num} to image: {e}")
        return None


def batch_process_pdfs(directory: str, convert_to_images: bool = False,
                       extract_embedded_images: bool = True) -> List[str]:
    """
    Process all PDF files in a directory.

    Args:
        directory: Directory containing PDF files
        convert_to_images: Whether to convert pages to images
        extract_embedded_images: Whether to extract embedded images

    Returns:
        List of output file paths
    """
    dir_path = Path(directory)
    pdf_files = list(dir_path.glob("*.pdf"))

    print(f"Found {len(pdf_files)} PDF files in {directory}")

    output_files = []
    for pdf_file in pdf_files:
        try:
            output = parse_pdf_submission(
                str(pdf_file),
                convert_to_images=convert_to_images,
                extract_embedded_images=extract_embedded_images
            )
            output_files.append(str(pdf_file.with_suffix('.md')))
        except Exception as e:
            print(f"Error processing {pdf_file.name}: {e}")

    return output_files


def extract_text_only(pdf_path: str) -> str:
    """Quick extraction of all text from PDF without formatting."""
    try:
        import pdfplumber
    except ImportError:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "pdfplumber", "-q"])
        import pdfplumber

    all_text = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            all_text.append(text)

    return "\n\n".join(all_text)


def extract_tables_only(pdf_path: str) -> List[Dict]:
    """Extract only tables from PDF."""
    try:
        import pdfplumber
    except ImportError:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "pdfplumber", "-q"])
        import pdfplumber

    all_tables = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            tables = page.extract_tables()
            for table in tables:
                all_tables.append({
                    'page': i,
                    'data': table
                })

    return all_tables


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Parse PDF submissions into structured Markdown format"
    )
    parser.add_argument("pdf_path", help="Path to the PDF file or directory")
    parser.add_argument("-o", "--output", help="Output file path (default: same name with .md extension)")
    parser.add_argument("-i", "--images", action="store_true",
                        help="Convert pages to images (for vision models)")
    parser.add_argument("--dpi", type=int, default=150,
                        help="DPI for image conversion (default: 150)")
    parser.add_argument("-b", "--batch", action="store_true",
                        help="Process all PDFs in directory")
    parser.add_argument("--text-only", action="store_true",
                        help="Extract text only (no formatting)")
    parser.add_argument("--tables-only", action="store_true",
                        help="Extract tables only")
    parser.add_argument("--no-extract-images", action="store_true",
                        help="Skip extracting embedded images from PDF")

    args = parser.parse_args()

    try:
        if args.batch:
            output_files = batch_process_pdfs(
                args.pdf_path,
                convert_to_images=args.images,
                extract_embedded_images=not args.no_extract_images
            )
            print(f"\nProcessed {len(output_files)} files:")
            for f in output_files:
                print(f"  - {f}")

        elif args.text_only:
            text = extract_text_only(args.pdf_path)
            print(text[:3000])

        elif args.tables_only:
            tables = extract_tables_only(args.pdf_path)
            print(f"Found {len(tables)} tables:")
            for t in tables[:5]:
                print(f"\nPage {t['page']}:")
                for row in t['data'][:5]:
                    print(f"  {row}")

        else:
            content = parse_pdf_submission(
                args.pdf_path,
                output_path=args.output,
                convert_to_images=args.images,
                dpi=args.dpi,
                extract_embedded_images=not args.no_extract_images
            )
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
