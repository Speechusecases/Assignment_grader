#!/usr/bin/env python3
"""
Convert Rubric PDF to JSON

This script converts a rubric PDF file to a structured JSON format
that can be used with the Assignment Evaluator agent.

Uses Claude via Azure Foundry to intelligently parse and structure the rubric.

Usage:
    python convert_rubric_pdf_to_json.py <rubric.pdf> [output.json]

Example:
    python convert_rubric_pdf_to_json.py my_rubric.pdf
    python convert_rubric_pdf_to_json.py my_rubric.pdf custom_output.json
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

# PDF parsing
import pdfplumber

# LLM for intelligent parsing
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file."""
    text_content = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text_content.append(f"--- Page {i + 1} ---\n{page_text}")

                # Also try to extract tables
                tables = page.extract_tables()
                for table in tables:
                    if table:
                        table_text = "\n".join([
                            " | ".join([str(cell) if cell else "" for cell in row])
                            for row in table
                        ])
                        text_content.append(f"\n[Table]\n{table_text}\n")

    except Exception as e:
        print(f"Error extracting PDF text: {e}")
        raise

    return "\n\n".join(text_content)


def get_llm():
    """Initialize the LLM."""
    azure_endpoint = os.environ.get("AZURE_FOUNDRY_ENDPOINT", "")
    api_key = os.environ.get("AZURE_FOUNDRY_API_KEY", "")
    model = os.environ.get("AZURE_FOUNDRY_MODEL", "claude-3-5-sonnet-20241022")

    if not azure_endpoint or not api_key:
        print("Error: Azure Foundry credentials not configured.")
        print("Please set AZURE_FOUNDRY_ENDPOINT and AZURE_FOUNDRY_API_KEY in your .env file")
        sys.exit(1)

    return ChatAnthropic(
        model=model,
        base_url=azure_endpoint,
        api_key=api_key,
        temperature=0.1,  # Low temperature for structured output
        max_tokens=8000
    )


def parse_rubric_with_llm(rubric_text: str, llm) -> dict:
    """Use LLM to parse rubric text into structured JSON."""

    system_prompt = """You are an expert at parsing educational rubrics into structured JSON format.

Your task is to extract rubric information from the provided text and convert it to a specific JSON structure.

OUTPUT FORMAT (you must output ONLY valid JSON, no other text):
{
  "rubric_name": "Name of the rubric/assignment",
  "total_points": <total maximum points as integer>,
  "sections": [
    {
      "section": "Section/Category Name",
      "points": <max points for this section as integer>,
      "weightage": <percentage weight as number, e.g., 10.5>,
      "description": [
        "Criterion 1",
        "Criterion 2",
        "Criterion 3"
      ],
      "levels": {
        "80-100": [
          "What earns high marks",
          "Excellence criteria"
        ],
        "60-80": [
          "What earns medium marks",
          "Satisfactory criteria"
        ],
        "<60": [
          "What earns low marks",
          "Below expectations"
        ]
      }
    }
  ]
}

INSTRUCTIONS:
1. Extract ALL sections/categories from the rubric
2. For each section, identify:
   - The section name
   - Maximum points possible
   - Weightage/percentage (calculate if not given: points/total_points * 100)
   - Description criteria (what is being evaluated)
   - Scoring levels (what distinguishes excellent/good/poor work)
3. If scoring levels are not explicitly given, infer them from the criteria
4. Ensure all points add up correctly
5. Output ONLY the JSON, no explanations or markdown

IMPORTANT: Return ONLY valid JSON. No markdown code blocks, no explanations."""

    user_prompt = f"""Parse this rubric into the JSON format specified:

{rubric_text}

Remember: Output ONLY valid JSON, nothing else."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    print("Parsing rubric with AI...")
    response = llm.invoke(messages)
    response_text = response.content.strip()

    # Clean up response - remove markdown code blocks if present
    if response_text.startswith("```"):
        # Remove ```json and ``` markers
        lines = response_text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines[-1].strip() == "```":
            lines = lines[:-1]
        response_text = "\n".join(lines)

    # Parse JSON
    try:
        rubric_json = json.loads(response_text)
        return rubric_json
    except json.JSONDecodeError as e:
        print(f"Error parsing LLM response as JSON: {e}")
        print(f"Response was:\n{response_text[:500]}...")
        raise


def validate_rubric_json(rubric: dict) -> bool:
    """Validate the rubric JSON structure."""
    required_keys = ["rubric_name", "total_points", "sections"]

    for key in required_keys:
        if key not in rubric:
            print(f"Missing required key: {key}")
            return False

    if not isinstance(rubric["sections"], list) or len(rubric["sections"]) == 0:
        print("Sections must be a non-empty list")
        return False

    for i, section in enumerate(rubric["sections"]):
        if "section" not in section:
            print(f"Section {i} missing 'section' name")
            return False
        if "points" not in section:
            print(f"Section '{section.get('section', i)}' missing 'points'")
            return False

    return True


def convert_rubric_pdf_to_json(pdf_path: str, output_path: str = None, save_file: bool = True) -> dict:
    """
    Convert a rubric PDF to JSON format.

    Args:
        pdf_path: Path to the rubric PDF file
        output_path: Optional path for output JSON (default: same name as PDF with .json)
        save_file: Whether to save the JSON to a file (default: True, set False for API usage)

    Returns:
        The parsed rubric dictionary
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if not pdf_path.suffix.lower() == ".pdf":
        raise ValueError("Input file must be a PDF")

    print(f"Converting: {pdf_path}")

    # Step 1: Extract text from PDF
    print("Step 1: Extracting text from PDF...")
    rubric_text = extract_text_from_pdf(str(pdf_path))
    print(f"  Extracted {len(rubric_text)} characters")

    # Step 2: Parse with LLM
    print("Step 2: Parsing rubric with AI...")
    llm = get_llm()
    rubric_json = parse_rubric_with_llm(rubric_text, llm)

    # Step 3: Validate
    print("Step 3: Validating JSON structure...")
    if not validate_rubric_json(rubric_json):
        print("Warning: Rubric JSON validation failed, but continuing...")

    # Step 4: Save to file (optional)
    if save_file:
        if output_path is None:
            output_path = pdf_path.with_suffix(".json")
        else:
            output_path = Path(output_path)

        print("Step 4: Saving JSON file...")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(rubric_json, f, indent=2, ensure_ascii=False)

        print("-" * 50)
        print(f"Success! Rubric saved to: {output_path}")

    print(f"  Rubric Name: {rubric_json.get('rubric_name', 'N/A')}")
    print(f"  Total Points: {rubric_json.get('total_points', 'N/A')}")
    print(f"  Sections: {len(rubric_json.get('sections', []))}")

    # Print section summary
    print("\nSections:")
    for section in rubric_json.get("sections", []):
        print(f"  - {section.get('section', 'Unknown')}: {section.get('points', '?')} points")

    return rubric_json


def main():
    parser = argparse.ArgumentParser(
        description="Convert rubric PDF to JSON format for the Assignment Evaluator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python convert_rubric_pdf_to_json.py rubric.pdf
    python convert_rubric_pdf_to_json.py rubric.pdf output.json
    python convert_rubric_pdf_to_json.py "My Rubric.pdf" "my_rubric.json"

The script uses Claude AI to intelligently parse the rubric structure.
Make sure your .env file has Azure Foundry credentials configured.
        """
    )

    parser.add_argument(
        "pdf_file",
        help="Path to the rubric PDF file"
    )

    parser.add_argument(
        "output_file",
        nargs="?",
        default=None,
        help="Output JSON file path (default: same name as PDF with .json extension)"
    )

    parser.add_argument(
        "--preview",
        action="store_true",
        help="Only extract and show PDF text without converting"
    )

    args = parser.parse_args()

    if args.preview:
        # Preview mode - just show extracted text
        print(f"Extracting text from: {args.pdf_file}")
        print("=" * 50)
        text = extract_text_from_pdf(args.pdf_file)
        print(text)
        print("=" * 50)
        print(f"Total characters: {len(text)}")
    else:
        # Convert mode
        try:
            convert_rubric_pdf_to_json(args.pdf_file, args.output_file)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
