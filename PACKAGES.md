# Python Scripts Package Dependencies

## Overview
This document lists all the Python scripts created and their package dependencies.

---

## Scripts Created

### 1. parse_jupyter_html.py
**Purpose:** Parse Jupyter Notebook HTML exports and extract cell content (input/output) along with embedded plots/images.

**Features:**
- Extracts code cells and markdown cells
- Captures cell outputs (stdout, errors)
- Extracts base64-encoded images/plots
- Saves images to separate files
- Generates structured Markdown output

**External Dependencies:**
| Package | Purpose | Installation |
|---------|---------|--------------|
| beautifulsoup4 | Parse HTML content | `pip install beautifulsoup4` |
| lxml | Optional faster HTML parser | `pip install lxml` |

**Built-in Modules Used:**
- `base64` - Decode base64 images
- `os` - File operations
- `sys` - System arguments
- `pathlib` - Path handling
- `typing` - Type hints

---

### 2. parse_pdf_submission.py
**Purpose:** Parse PDF submissions (business reports, slide decks, executive summaries) into layout-aware Markdown format.

**Features:**
- Page-level text extraction
- Table extraction (converts to Markdown tables)
- Embedded image extraction (charts, graphs)
- Optional page-to-image conversion
- Structured output with page references

**External Dependencies:**
| Package | Purpose | Installation |
|---------|---------|--------------|
| pdfplumber | Extract text and tables from PDF | `pip install pdfplumber` |
| PyMuPDF | Extract embedded images (fitz) | `pip install PyMuPDF` |
| Pillow | Image processing (PIL) | `pip install Pillow` |

**Built-in Modules Used:**
- `io` - Byte stream handling
- `sys` - System arguments
- `pathlib` - Path handling
- `typing` - Type hints
- `argparse` - Command-line arguments

---

## Installation

### Option 1: Install all packages at once
```bash
pip install -r requirements.txt
```

### Option 2: Install individually

**For Jupyter HTML parsing:**
```bash
pip install beautifulsoup4 lxml
```

**For PDF parsing:**
```bash
pip install pdfplumber PyMuPDF Pillow
```

### Option 3: Install everything
```bash
pip install beautifulsoup4 lxml pdfplumber PyMuPDF Pillow
```

---

## Package Descriptions

### beautifulsoup4
- **Version:** >=4.12.0
- **Description:** Library for parsing HTML and XML documents
- **Used in:** parse_jupyter_html.py
- **Purpose:** Parse Jupyter HTML exports to find cells, inputs, outputs, and images

### lxml
- **Version:** >=4.9.0
- **Description:** High-performance XML and HTML parser
- **Used in:** parse_jupyter_html.py (optional but recommended)
- **Purpose:** Faster alternative to Python's built-in html.parser

### pdfplumber
- **Version:** >=0.10.0
- **Description:** Extracts tables and text from PDFs
- **Used in:** parse_pdf_submission.py
- **Purpose:** Extract text while preserving layout, extract tables as structured data

### PyMuPDF
- **Version:** >=1.23.0
- **Description:** Also known as 'fitz' - comprehensive PDF manipulation library
- **Used in:** parse_pdf_submission.py
- **Purpose:** Extract embedded images (charts, graphs) from PDF pages

### Pillow
- **Version:** >=10.0.0
- **Description:** Python Imaging Library (fork)
- **Used in:** parse_pdf_submission.py
- **Purpose:** Process and save extracted images, get image dimensions

---

## Quick Reference: Import Statements

```python
# parse_jupyter_html.py
from bs4 import BeautifulSoup  # beautifulsoup4
import base64                    # built-in
import os                        # built-in
import sys                       # built-in
from pathlib import Path         # built-in

# parse_pdf_submission.py
import pdfplumber                # pdfplumber
import fitz                      # PyMuPDF
from PIL import Image            # Pillow
import io                        # built-in
import sys                       # built-in
from pathlib import Path         # built-in
from typing import List, Dict, Optional, Tuple  # built-in
import argparse                  # built-in
```

---

## Verification

To verify all packages are installed correctly:

```bash
python3 -c "from bs4 import BeautifulSoup; print('✓ beautifulsoup4')"
python3 -c "import lxml; print('✓ lxml')"
python3 -c "import pdfplumber; print('✓ pdfplumber')"
python3 -c "import fitz; print('✓ PyMuPDF')"
python3 -c "from PIL import Image; print('✓ Pillow')"
```

Or run the test script:
```bash
python3 -c "
from bs4 import BeautifulSoup
import pdfplumber
import fitz
from PIL import Image
print('All packages installed successfully!')
"
```

---

## File Locations

```
/Users/sarveshmani/demo/
├── parse_jupyter_html.py          # Jupyter HTML parser
├── parse_pdf_submission.py        # PDF parser
├── requirements.txt               # Package dependencies
└── PACKAGES.md                    # This file
```

---

## Usage Examples

### Jupyter HTML Parsing
```bash
python3 parse_jupyter_html.py notebook.html
```

### PDF Parsing
```bash
# Basic text and table extraction
python3 parse_pdf_submission.py report.pdf

# With embedded image extraction (default)
python3 parse_pdf_submission.py report.pdf

# With page images (for vision models)
python3 parse_pdf_submission.py report.pdf --images
```

---

## Notes

1. **lxml** is optional for parse_jupyter_html.py but recommended for better performance
2. **PyMuPDF** requires compilation tools on some systems; pre-built wheels are available for most platforms
3. **Pillow** may require system libraries for image format support (libjpeg, zlib, etc.)
4. All scripts include automatic package installation fallback if imports fail
