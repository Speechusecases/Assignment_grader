#!/usr/bin/env python3
"""
LangGraph Agent for Evaluating Student Submissions

This agent:
1. Takes HTML or PDF file paths as input
2. Extracts content using parse_jupyter_html.py or parse_pdf_submission.py
3. Uses Anthropic Claude models via Azure Foundry with LangGraph workflow
4. Evaluates submissions based on rubrics and prompt guidelines
5. Outputs structured feedback with marks

Usage:
    python langraph_evaluator_agent.py <submission_file> [--type html|pdf] [--model claude-3-5-sonnet]

Environment Variables:
    AZURE_FOUNDRY_ENDPOINT: Azure Foundry endpoint URL
    AZURE_FOUNDRY_API_KEY: Azure Foundry API key
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, TypedDict, Optional
from dataclasses import dataclass

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

# LangGraph imports
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic

# Import existing parsers
sys.path.insert(0, str(Path(__file__).parent))
from parse_jupyter_html import parse_jupyter_html
from parse_pdf_submission import parse_pdf_submission


# =============================================================================
# STATE DEFINITIONS
# =============================================================================

class EvaluatorState(TypedDict):
    """State maintained throughout the evaluation workflow."""
    file_path: str
    file_type: str
    extracted_content: str
    rubric_criteria: Dict[str, Any]
    prompt_guidelines: str
    evaluations: Dict[str, Any]
    total_marks: int
    max_marks: int
    final_feedback: str
    errors: List[str]


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class EvaluatorConfig:
    """Configuration for the evaluator agent."""
    # Azure Foundry settings for Anthropic Claude
    model: str = None  # Claude model (loaded from .env if not provided)
    azure_endpoint: str = None  # Azure Foundry endpoint URL
    api_key: str = None  # Azure Foundry API key
    temperature: float = None
    max_tokens: int = None
    extract_images: bool = True
    custom_rubric: Dict[str, Any] = None  # Custom rubric data (if provided)

    def __post_init__(self):
        """Load Azure Foundry configuration from .env file if not provided."""
        # Load endpoint and API key
        if self.azure_endpoint is None:
            self.azure_endpoint = os.environ.get("AZURE_FOUNDRY_ENDPOINT", "")
        if self.api_key is None:
            self.api_key = os.environ.get("AZURE_FOUNDRY_API_KEY", "")

        # Load model with default fallback
        if self.model is None:
            self.model = os.environ.get("AZURE_FOUNDRY_MODEL", "claude-3-5-sonnet-20241022")

        # Load temperature with default fallback
        if self.temperature is None:
            self.temperature = float(os.environ.get("AZURE_FOUNDRY_TEMPERATURE", "0.3"))

        # Load max_tokens with default fallback
        if self.max_tokens is None:
            self.max_tokens = int(os.environ.get("AZURE_FOUNDRY_MAX_TOKENS", "4096"))


# =============================================================================
# NODE FUNCTIONS
# =============================================================================

def detect_file_type(state: EvaluatorState, config: EvaluatorConfig) -> EvaluatorState:
    """Node: Detect and validate file type."""
    file_path = Path(state["file_path"])

    if not file_path.exists():
        state["errors"].append(f"File not found: {file_path}")
        return state

    extension = file_path.suffix.lower()

    if extension == ".html":
        state["file_type"] = "html"
    elif extension == ".pdf":
        state["file_type"] = "pdf"
    else:
        state["errors"].append(f"Unsupported file type: {extension}. Only .html and .pdf are supported.")

    return state


def extract_content(state: EvaluatorState, config: EvaluatorConfig) -> EvaluatorState:
    """Node: Extract content from submission file."""
    if state["errors"]:
        return state

    try:
        file_path = state["file_path"]
        file_type = state["file_type"]

        if file_type == "html":
            # Parse Jupyter HTML export
            content = parse_jupyter_html(
                html_file_path=file_path,
                output_file_path=None,  # Don't save intermediate file
                extract_images=config.extract_images
            )
        elif file_type == "pdf":
            # Parse PDF submission
            content = parse_pdf_submission(
                pdf_path=file_path,
                output_path=None,  # Don't save intermediate file
                convert_to_images=False,
                extract_embedded_images=config.extract_images
            )
        else:
            state["errors"].append(f"Unknown file type: {file_type}")
            return state

        state["extracted_content"] = content
        print(f"✓ Extracted {len(content)} characters from submission")

    except Exception as e:
        state["errors"].append(f"Error extracting content: {str(e)}")

    return state


def load_rubric(state: EvaluatorState, config: EvaluatorConfig) -> EvaluatorState:
    """Node: Load rubric criteria from JSON files or use custom rubric."""

    # If custom rubric is provided in config, use it
    if config.custom_rubric:
        state["rubric_criteria"] = config.custom_rubric
        rubric_name = config.custom_rubric.get("rubric_name", "Custom Rubric")
        print(f"✓ Using custom rubric: {rubric_name}")
        return state

    # Otherwise, load from default rubric files
    rubric_files = [
        "personal_loan_rubrics.json",
        "model_deployment_rubric.json"
    ]

    rubric_data = {}

    for rubric_file in rubric_files:
        rubric_path = Path(__file__).parent / rubric_file
        if rubric_path.exists():
            try:
                with open(rubric_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    rubric_data[rubric_file.replace('.json', '')] = data
                    print(f"✓ Loaded rubric: {rubric_file}")
            except json.JSONDecodeError:
                # File exists but is empty or invalid
                pass
            except Exception as e:
                state["errors"].append(f"Error loading {rubric_file}: {str(e)}")

    # If no rubrics loaded, use default criteria
    if not rubric_data:
        rubric_data = get_default_rubric()
        print("✓ Using default rubric")

    state["rubric_criteria"] = rubric_data
    return state


def load_prompt_guidelines(state: EvaluatorState, config: EvaluatorConfig) -> EvaluatorState:
    """Node: Load prompt guidelines from prompt.txt."""
    prompt_path = Path(__file__).parent / "prompt.txt"

    if prompt_path.exists():
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                guidelines = f.read()
                if guidelines.strip():
                    state["prompt_guidelines"] = guidelines
                    print(f"✓ Loaded prompt guidelines ({len(guidelines)} characters)")
                else:
                    state["prompt_guidelines"] = get_default_prompt()
                    print("✓ Using default prompt guidelines (prompt.txt was empty)")
        except Exception as e:
            state["errors"].append(f"Error loading prompt.txt: {str(e)}")
            state["prompt_guidelines"] = get_default_prompt()
    else:
        state["prompt_guidelines"] = get_default_prompt()
        print("✓ Using default prompt guidelines (prompt.txt not found)")

    return state


def evaluate_with_llm(state: EvaluatorState, config: EvaluatorConfig) -> EvaluatorState:
    """Node: Use LLM to evaluate the submission."""
    if state["errors"]:
        return state

    try:
        # Validate Azure Foundry configuration
        if not config.azure_endpoint or not config.api_key:
            state["errors"].append(
                "Azure Foundry credentials not configured. "
                "Set AZURE_FOUNDRY_ENDPOINT and AZURE_FOUNDRY_API_KEY in .env file."
            )
            return state

        # Initialize Anthropic Claude model via Azure Foundry
        llm = ChatAnthropic(
            model=config.model,
            base_url=config.azure_endpoint,
            api_key=config.api_key,
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )

        # Prepare system prompt with guidelines
        system_prompt = build_system_prompt(state["prompt_guidelines"])

        # Prepare user prompt with submission content
        user_prompt = build_evaluation_prompt(
            content=state["extracted_content"],
            rubric=state["rubric_criteria"]
        )

        print(f"\n🤖 Evaluating with {config.model} via Azure Foundry...")

        # Call LLM
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        response = llm.invoke(messages)
        evaluation_text = response.content

        # Debug: Print first 500 chars of response
        print(f"\n📝 LLM Response Preview:\n{evaluation_text[:500]}...\n")

        # Parse the evaluation response
        parsed_eval = parse_evaluation_response(evaluation_text)
        state["evaluations"] = parsed_eval

        # Get marks from parsed evaluation, with fallback to rubric total
        state["total_marks"] = parsed_eval.get("total_marks", 0)

        # Calculate max_marks from rubric if not parsed
        if parsed_eval.get("max_marks", 0) > 0:
            state["max_marks"] = parsed_eval.get("max_marks")
        else:
            # Calculate from rubric sections
            rubric = state.get("rubric_criteria", {})
            sections = rubric.get("sections", [])
            total_from_rubric = sum(
                s.get("points", s.get("max_marks", 0))
                for s in sections
            )
            state["max_marks"] = rubric.get("total_points", total_from_rubric) or 100

        state["final_feedback"] = evaluation_text

        print(f"✓ Evaluation complete: {state['total_marks']}/{state['max_marks']} marks")
        print(f"  Sections parsed: {len(parsed_eval.get('sections', []))}")

    except Exception as e:
        state["errors"].append(f"Error during LLM evaluation: {str(e)}")
        import traceback
        traceback.print_exc()

    return state


def generate_report(state: EvaluatorState, config: EvaluatorConfig) -> EvaluatorState:
    """Node: Generate final evaluation report."""
    if state["errors"]:
        error_report = "\n".join([f"❌ {error}" for error in state["errors"]])
        state["final_feedback"] = f"Evaluation Failed:\n{error_report}"
        return state

    # Format the final report
    report = format_evaluation_report(state)
    state["final_feedback"] = report

    return state


def save_results(state: EvaluatorState, config: EvaluatorConfig) -> EvaluatorState:
    """Node: Save evaluation results to file."""
    if state["errors"]:
        return state

    try:
        input_path = Path(state["file_path"])
        output_path = input_path.with_suffix('.evaluation.json')

        results = {
            "file": input_path.name,
            "file_type": state["file_type"],
            "model": config.model,
            "total_marks": state["total_marks"],
            "max_marks": state["max_marks"],
            "percentage": round((state["total_marks"] / state["max_marks"]) * 100, 2) if state["max_marks"] > 0 else 0,
            "evaluations": state["evaluations"],
            "feedback": state["final_feedback"]
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Results saved to: {output_path}")

        # Also save markdown report
        md_path = input_path.with_suffix('.evaluation.md')
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(state["final_feedback"])

        print(f"💾 Report saved to: {md_path}")

    except Exception as e:
        state["errors"].append(f"Error saving results: {str(e)}")

    return state


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_default_rubric() -> Dict[str, Any]:
    """Return default rubric criteria if no rubric files are found."""
    return {
        "sections": [
            {
                "name": "Business Understanding & Problem Definition",
                "max_marks": 10,
                "criteria": [
                    "Clear problem statement",
                    "Business context explained",
                    "Objectives are well-defined"
                ]
            },
            {
                "name": "Data Understanding & Exploration",
                "max_marks": 15,
                "criteria": [
                    "Data loading and initial inspection",
                    "Exploratory data analysis (EDA)",
                    "Data quality assessment",
                    "Visualizations of key patterns"
                ]
            },
            {
                "name": "Data Preparation & Preprocessing",
                "max_marks": 15,
                "criteria": [
                    "Handling missing values",
                    "Feature engineering",
                    "Data transformation",
                    "Train-test split strategy"
                ]
            },
            {
                "name": "Modeling",
                "max_marks": 25,
                "criteria": [
                    "Multiple models attempted",
                    "Hyperparameter tuning",
                    "Cross-validation",
                    "Model comparison and selection"
                ]
            },
            {
                "name": "Evaluation",
                "max_marks": 15,
                "criteria": [
                    "Appropriate metrics used",
                    "Confusion matrix analysis",
                    "ROC/AUC analysis",
                    "Model performance interpretation"
                ]
            },
            {
                "name": "Deployment & Conclusions",
                "max_marks": 10,
                "criteria": [
                    "Final model selection justification",
                    "Business recommendations",
                    "Deployment considerations"
                ]
            },
            {
                "name": "Code Quality & Documentation",
                "max_marks": 10,
                "criteria": [
                    "Code is well-organized",
                    "Comments and explanations",
                    "Notebook structure",
                    "Professional presentation"
                ]
            }
        ]
    }


def get_default_prompt() -> str:
    """Return default prompt guidelines if prompt.txt is empty."""
    return """Role: You are an expert mentor reviewer for Data Science / ML / AI assignments. Use a supportive, technical, and actionable tone. Treat the instructor's evaluation guide as internal scoring rules only.

Input Types:
- You will typically receive a notebook and/or report as the submission.
- Sometimes you will receive image screenshots of a working application for model deployment assignments.

Formatting & Structure:

Section Headers:
Use the exact section names listed in the evaluation guide, in the same order. Do not add, merge, rename, or omit sections.

Section Paragraphs:
For each section, write a single paragraph of exactly 4–5 sentences.

One sentence must be the marks sentence in the exact format:
"Marks: X/Y" (where Y is the section max).

The remaining sentences must:
- State what was done well tied to that section's criteria
- Explain what to improve tied to the criteria
- Identify the specific technical gaps or omissions found in the submission that led to a lower score

Constraint:
Do not provide a numerical justification or a granular point-by-point breakdown for how many marks were deducted for each specific error or step.

Missing/NA Sections:
If a section is missing, write: "Not addressed."
Award 0/Y, and state the technical requirement that was not met.

Tone & Voice:
- Write like a senior mentor giving helpful, constructive feedback to a junior data scientist.
- Be specific about code/approach issues (e.g., "Consider using stratified sampling" rather than "Sampling was bad").
- Keep language professional but encouraging.
- Avoid generic praise or criticism; tie everything to observable evidence in the submission.

Example Output Format:

## 1. Business Understanding & Problem Definition
The submission clearly defines the business problem of predicting loan defaults, outlining how this impacts the bank's risk management strategy. Marks: 8/10
However, the objectives could be more specific regarding the target metric (precision vs. recall trade-off). The business context would benefit from quantifying the cost of false positives versus false negatives. Consider adding stakeholder requirements and success criteria for the model deployment.

## 2. Data Understanding & Exploration
Excellent exploratory analysis with comprehensive univariate and bivariate visualizations that reveal key patterns in the data. Marks: 14/15
The correlation analysis is thorough, but the submission lacks an analysis of outliers and their potential impact on model performance. Consider adding data profiling reports and documenting data quality issues more explicitly. The class imbalance observation is good but needs a discussion on handling strategies.
"""


def build_system_prompt(prompt_guidelines: str) -> str:
    """Build the system prompt for the LLM."""
    return f"""{prompt_guidelines}

Your task is to evaluate the following student submission and provide structured feedback following the guidelines above.

IMPORTANT RULES:
1. Evaluate ONLY based on what is present in the submission - do not assume missing content exists
2. Provide specific, actionable feedback tied to the evaluation criteria
3. Use the exact marks format: "Marks: X/Y" in each section
4. Be constructive but honest about gaps and areas for improvement
5. Focus on technical accuracy, methodology, and completeness
6. For code submissions, evaluate code quality, documentation, and reproducibility
7. For model deployment submissions, evaluate functional correctness and completeness
"""


def build_evaluation_prompt(content: str, rubric: Dict[str, Any]) -> str:
    """Build the evaluation prompt with content and rubric."""

    # Format rubric for the prompt - handle different rubric formats
    rubric_lines = []
    total_max_marks = 0

    # Get sections from rubric (handle both direct and nested formats)
    sections = rubric.get('sections', [])

    for section in sections:
        # Handle different key names for section name
        section_name = section.get('section', section.get('name', 'Section'))
        # Handle different key names for points
        max_marks = section.get('points', section.get('max_marks', 0))
        total_max_marks += max_marks

        # Handle different key names for criteria
        criteria = section.get('description', section.get('criteria', []))
        if isinstance(criteria, list):
            criteria_str = ', '.join(criteria)
        else:
            criteria_str = str(criteria)

        rubric_lines.append(f"- **{section_name}**: Max {max_marks} marks")
        if criteria_str:
            rubric_lines.append(f"  Criteria: {criteria_str}")

    rubric_text = "\n".join(rubric_lines)
    rubric_name = rubric.get('rubric_name', 'Evaluation Rubric')
    total_points = rubric.get('total_points', total_max_marks)

    # Truncate content if too long (to fit in context window)
    max_content_length = 15000  # Adjust based on model context window
    if len(content) > max_content_length:
        content = content[:max_content_length] + "\n\n[Content truncated due to length...]"

    return f"""Please evaluate the following student submission.

## RUBRIC: {rubric_name}
Total Maximum Points: {total_points}

### Sections to Evaluate:
{rubric_text}

## SUBMISSION CONTENT
{content}

## INSTRUCTIONS
1. Evaluate EACH section listed in the rubric above
2. For EACH section, you MUST include marks in this EXACT format: "Marks: X/Y" (where Y is the max marks for that section)
3. Provide specific, actionable feedback for each section
4. At the end, include: "Total Marks: X/{total_points}"
5. Provide an overall summary with key strengths and areas for improvement

## REQUIRED OUTPUT FORMAT

For each section in the rubric, write:

## [Section Name]
[3-5 sentences of specific feedback about this section]
Marks: X/Y

Then at the end:

## Overall Summary
[2-3 paragraphs summarizing overall performance]

## Total Score
Total Marks: X/{total_points}

## Key Recommendations
- [Specific recommendation 1]
- [Specific recommendation 2]
- [Specific recommendation 3]

IMPORTANT: You must evaluate ALL sections and provide marks for EACH one using "Marks: X/Y" format.
"""


def parse_evaluation_response(response: str) -> Dict[str, Any]:
    """Parse the LLM evaluation response to extract structured data."""
    import re

    evaluations = {
        "sections": [],
        "total_marks": 0,
        "max_marks": 0,
        "summary": "",
        "recommendations": []
    }

    # Try multiple patterns to extract marks

    # Pattern 1: ## Section Name ... Marks: X/Y
    section_pattern1 = r'##\s*\d*\.?\s*([^\n]+)\n([\s\S]*?)Marks:\s*(\d+)\s*/\s*(\d+)'
    section_matches = re.findall(section_pattern1, response, re.IGNORECASE)

    # Pattern 2: **Section Name** ... Marks: X/Y
    if not section_matches:
        section_pattern2 = r'\*\*([^\*\n]+)\*\*[:\s]*([\s\S]*?)Marks:\s*(\d+)\s*/\s*(\d+)'
        section_matches = re.findall(section_pattern2, response, re.IGNORECASE)

    # Pattern 3: Section Name: ... Marks: X/Y (numbered list)
    if not section_matches:
        section_pattern3 = r'\d+\.\s*\*?\*?([^\n\*]+)\*?\*?[:\s]*([\s\S]*?)Marks:\s*(\d+)\s*/\s*(\d+)'
        section_matches = re.findall(section_pattern3, response, re.IGNORECASE)

    for match in section_matches:
        section_name = match[0].strip().strip('*').strip(':').strip()
        feedback = match[1].strip()
        marks = int(match[2])
        max_marks = int(match[3])

        # Skip if this looks like "Total" section
        if 'total' in section_name.lower() and 'score' in section_name.lower():
            continue

        evaluations["sections"].append({
            "name": section_name,
            "marks": marks,
            "max_marks": max_marks,
            "feedback": feedback
        })

        evaluations["total_marks"] += marks
        evaluations["max_marks"] += max_marks

    # Also try to find individual "Marks: X/Y" patterns if no sections found
    if not evaluations["sections"]:
        marks_pattern = r'Marks:\s*(\d+)\s*/\s*(\d+)'
        marks_matches = re.findall(marks_pattern, response, re.IGNORECASE)
        for match in marks_matches:
            marks = int(match[0])
            max_marks = int(match[1])
            evaluations["total_marks"] += marks
            evaluations["max_marks"] += max_marks

    # Try to extract explicit total marks (this takes precedence if found)
    total_patterns = [
        r'Total\s*Marks?:\s*(\d+)\s*/\s*(\d+)',
        r'Total\s*Score:\s*(\d+)\s*/\s*(\d+)',
        r'Overall\s*Score:\s*(\d+)\s*/\s*(\d+)',
        r'Final\s*Score:\s*(\d+)\s*/\s*(\d+)',
        r'\*\*Total[:\s]*(\d+)\s*/\s*(\d+)\*\*',
    ]

    for pattern in total_patterns:
        total_match = re.search(pattern, response, re.IGNORECASE)
        if total_match:
            evaluations["total_marks"] = int(total_match.group(1))
            evaluations["max_marks"] = int(total_match.group(2))
            break

    # Extract recommendations
    rec_section = re.search(r'(?:Key\s*)?Recommendations?:?\s*\n([\s\S]+?)(?:\n##|\n\*\*|\Z)', response, re.IGNORECASE)
    if rec_section:
        rec_text = rec_section.group(1)
        recommendations = re.findall(r'^[\s]*[-*\d.]+\s*(.+)$', rec_text, re.MULTILINE)
        evaluations["recommendations"] = [r.strip() for r in recommendations if r.strip()]

    # Store full response as summary
    evaluations["summary"] = response[:500] if len(response) > 500 else response

    return evaluations


def format_evaluation_report(state: EvaluatorState) -> str:
    """Format the final evaluation report."""

    # Calculate percentage safely
    if state['max_marks'] > 0:
        percentage = round((state['total_marks'] / state['max_marks']) * 100, 2)
    else:
        percentage = 0

    report = f"""# Student Submission Evaluation Report

## Summary
- **File:** {state['file_path']}
- **Type:** {state['file_type'].upper()}
- **Model Used:** {EvaluatorConfig().model} (via Azure Foundry)
- **Total Score:** {state['total_marks']}/{state['max_marks']} ({percentage}%)

---

## Detailed Evaluation

{state['final_feedback']}
"""

    return report


# =============================================================================
# GRAPH BUILDER
# =============================================================================

def build_evaluation_graph(config: EvaluatorConfig):
    """Build and compile the LangGraph evaluation workflow."""

    # Create state graph
    workflow = StateGraph(EvaluatorState)

    # Add nodes
    workflow.add_node("detect_file_type", lambda state: detect_file_type(state, config))
    workflow.add_node("extract_content", lambda state: extract_content(state, config))
    workflow.add_node("load_rubric", lambda state: load_rubric(state, config))
    workflow.add_node("load_prompt", lambda state: load_prompt_guidelines(state, config))
    workflow.add_node("evaluate", lambda state: evaluate_with_llm(state, config))
    workflow.add_node("generate_report", lambda state: generate_report(state, config))
    workflow.add_node("save_results", lambda state: save_results(state, config))

    # Define edges
    workflow.set_entry_point("detect_file_type")

    # Conditional edges based on errors
    def check_errors(state):
        return "error" if state["errors"] else "continue"

    workflow.add_conditional_edges(
        "detect_file_type",
        check_errors,
        {
            "continue": "extract_content",
            "error": "generate_report"
        }
    )

    workflow.add_conditional_edges(
        "extract_content",
        check_errors,
        {
            "continue": "load_rubric",
            "error": "generate_report"
        }
    )

    # Parallel loading of rubric and prompt
    workflow.add_edge("load_rubric", "load_prompt")
    workflow.add_edge("load_prompt", "evaluate")

    workflow.add_conditional_edges(
        "evaluate",
        check_errors,
        {
            "continue": "generate_report",
            "error": "generate_report"
        }
    )

    workflow.add_edge("generate_report", "save_results")
    workflow.add_edge("save_results", END)

    return workflow.compile()


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def run_evaluation(
    file_path: str,
    file_type: Optional[str] = None,
    model: str = None,
    temperature: float = None,
    extract_images: bool = True,
    azure_endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
    custom_rubric: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Run the evaluation workflow on a submission file.

    Args:
        file_path: Path to the submission file (HTML or PDF)
        file_type: Override file type detection ('html' or 'pdf')
        model: Claude model to use (defaults to AZURE_FOUNDRY_MODEL from .env)
        temperature: LLM temperature (defaults to AZURE_FOUNDRY_TEMPERATURE from .env)
        extract_images: Whether to extract images from submission
        azure_endpoint: Azure Foundry endpoint (defaults to AZURE_FOUNDRY_ENDPOINT from .env)
        api_key: Azure Foundry API key (defaults to AZURE_FOUNDRY_API_KEY from .env)
        custom_rubric: Custom rubric dict to use instead of default rubric files

    Returns:
        Dict containing evaluation results
    """
    # Initialize configuration (loads from .env if not provided)
    config = EvaluatorConfig(
        model=model,
        temperature=temperature,
        extract_images=extract_images,
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        custom_rubric=custom_rubric
    )

    # Initialize state
    initial_state: EvaluatorState = {
        "file_path": file_path,
        "file_type": file_type or "",
        "extracted_content": "",
        "rubric_criteria": {},
        "prompt_guidelines": "",
        "evaluations": {},
        "total_marks": 0,
        "max_marks": 0,
        "final_feedback": "",
        "errors": []
    }

    # Build and run graph
    graph = build_evaluation_graph(config)

    print(f"\n{'='*60}")
    print(f"LangGraph Evaluator Agent")
    print(f"{'='*60}")
    print(f"File: {file_path}")
    print(f"Model: {model}")
    print(f"{'='*60}\n")

    # Run the workflow
    final_state = graph.invoke(initial_state)

    print(f"\n{'='*60}")
    print("Evaluation Complete!")
    print(f"{'='*60}")

    if final_state["errors"]:
        print("\n⚠️  Errors encountered:")
        for error in final_state["errors"]:
            print(f"   - {error}")

    return {
        "file_path": final_state["file_path"],
        "file_type": final_state["file_type"],
        "total_marks": final_state["total_marks"],
        "max_marks": final_state["max_marks"],
        "percentage": round((final_state["total_marks"] / final_state["max_marks"]) * 100, 2) if final_state["max_marks"] > 0 else 0,
        "feedback": final_state["final_feedback"],
        "evaluations": final_state["evaluations"],
        "errors": final_state["errors"]
    }


def main():
    """Main entry point with command-line arguments."""
    parser = argparse.ArgumentParser(
        description="LangGraph Agent for evaluating student submissions using Anthropic Claude via Azure Foundry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Evaluate HTML notebook submission (uses settings from .env)
    python langraph_evaluator_agent.py submission.html

    # Evaluate PDF report with specific Claude model
    python langraph_evaluator_agent.py report.pdf --model claude-3-5-sonnet-20241022

    # Evaluate with custom temperature
    python langraph_evaluator_agent.py submission.html --model claude-3-opus-20240229 --temperature 0.5

Available Claude Models (via Azure Foundry):
    - claude-3-5-sonnet-20241022  (Recommended - balanced performance)
    - claude-3-opus-20240229      (Most capable)
    - claude-3-sonnet-20240229    (Balanced)
    - claude-3-haiku-20240307     (Fast and efficient)
    - claude-opus-4-5             (Latest Opus)

Environment Variables (set in .env file):
    AZURE_FOUNDRY_ENDPOINT  - Azure Foundry endpoint URL
    AZURE_FOUNDRY_API_KEY   - Azure Foundry API key
    AZURE_FOUNDRY_MODEL     - Default model to use
    AZURE_FOUNDRY_TEMPERATURE - Default temperature
        """
    )

    parser.add_argument(
        "file_path",
        help="Path to the submission file (HTML or PDF)"
    )

    parser.add_argument(
        "--type",
        choices=["html", "pdf"],
        help="File type (auto-detected if not specified)"
    )

    parser.add_argument(
        "--model",
        default=None,
        help="Claude model to use (default: from .env or claude-3-5-sonnet-20241022)"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="LLM temperature - lower is more deterministic (default: from .env or 0.3)"
    )

    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Skip extracting images from submission"
    )

    parser.add_argument(
        "--output",
        "-o",
        help="Custom output file path for the report"
    )

    args = parser.parse_args()

    # Check Azure Foundry configuration
    endpoint = os.environ.get("AZURE_FOUNDRY_ENDPOINT")
    api_key = os.environ.get("AZURE_FOUNDRY_API_KEY")

    if not endpoint or not api_key:
        print("⚠️  Warning: Azure Foundry credentials not configured in .env file.")
        print("   Please set the following in your .env file:")
        print("     AZURE_FOUNDRY_ENDPOINT='https://your-endpoint.azure.com'")
        print("     AZURE_FOUNDRY_API_KEY='your-api-key'")
        print()

    # Run evaluation
    try:
        results = run_evaluation(
            file_path=args.file_path,
            file_type=args.type,
            model=args.model,
            temperature=args.temperature,
            extract_images=not args.no_images
        )

        # Print summary
        print(f"\n📊 FINAL SCORE: {results['total_marks']}/{results['max_marks']} ({results['percentage']}%)")

        # Save to custom output if specified
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(results['feedback'])
            print(f"\n💾 Custom output saved to: {args.output}")

        # Exit with error code if there were errors
        if results['errors']:
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
