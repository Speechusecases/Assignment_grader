# LangGraph Evaluator Agent

An AI-powered agent built with LangGraph and Ollama for automated evaluation of student submissions (Jupyter notebooks as HTML exports and PDF reports).

## Overview

This agent:
1. **Parses** student submissions from HTML (Jupyter notebooks) or PDF files
2. **Extracts** content using existing parser scripts
3. **Evaluates** submissions using Ollama models (kimi, llama3, mistral, etc.)
4. **Generates** structured feedback following your custom evaluation guidelines

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Setup Ollama

Install Ollama from [ollama.com](https://ollama.com) and start the server:

```bash
ollama serve
```

Pull a model:

```bash
ollama pull kimi
# or
ollama pull llama3
```

### 3. Run Evaluation

```bash
python langraph_evaluator_agent.py submission.html
```

Or with options:

```bash
python langraph_evaluator_agent.py submission.pdf --model kimi --temperature 0.3
```

## Usage Options

### Command Line

```bash
# Evaluate HTML notebook
python langraph_evaluator_agent.py notebook.html

# Evaluate PDF report
python langraph_evaluator_agent.py report.pdf

# Use specific model
python langraph_evaluator_agent.py submission.html --model llama3

# Custom temperature (0.0 = deterministic, 1.0 = creative)
python langraph_evaluator_agent.py submission.html --temperature 0.5

# Skip image extraction
python langraph_evaluator_agent.py submission.html --no-images

# Custom output file
python langraph_evaluator_agent.py submission.html -o my_report.md
```

### Python API

```python
from langraph_evaluator_agent import run_evaluation

# Evaluate a submission
results = run_evaluation(
    file_path="submission.html",
    model="kimi",
    temperature=0.3,
    extract_images=True
)

# Access results
print(f"Score: {results['total_marks']}/{results['max_marks']}")
print(f"Percentage: {results['percentage']}%")

# Access section breakdown
for section in results['evaluations']['sections']:
    print(f"{section['name']}: {section['marks']}/{section['max_marks']}")
```

### Quickstart Script

```bash
# Setup everything
python quickstart.py setup

# Check Ollama status
python quickstart.py check

# Pull a model
python quickstart.py pull kimi

# Test evaluation
python quickstart.py test submission.html

# Batch process directory
python quickstart.py batch ./submissions
```

## Customization

### Edit Evaluation Prompt

Modify `prompt.txt` to customize evaluation guidelines:

- Section headers and order
- Evaluation criteria per section
- Marks distribution
- Tone and voice instructions

### Define Custom Rubrics

Create JSON rubric files:

**personal_loan_rubrics.json:**
```json
{
  "sections": [
    {
      "name": "Business Understanding",
      "max_marks": 10,
      "criteria": ["Problem statement", "Business context"]
    },
    {
      "name": "Data Understanding",
      "max_marks": 15,
      "criteria": ["EDA", "Data quality"]
    }
  ]
}
```

## Output Files

After evaluation, three files are generated:

1. **`.evaluation.json`** - Structured results with marks breakdown
2. **`.evaluation.md`** - Human-readable evaluation report
3. **`-o custom_output.md`** (optional) - Custom report location

### Example Output Structure

```markdown
# Student Submission Evaluation Report

## Summary
- **File:** submission.html
- **Type:** HTML
- **Model Used:** kimi
- **Total Score:** 82/100 (82.0%)

---

## Detailed Feedback

# Section-wise Evaluation

## 1. Business Understanding & Problem Definition
[Feedback paragraph]
Marks: 8/10

## 2. Data Understanding & Exploration
[Feedback paragraph]
Marks: 13/15

...

# Overall Summary
[Summary of performance]

# Key Recommendations
- Specific recommendation 1
- Specific recommendation 2
```

## Available Models

The agent works with any Ollama model:

| Model | Description | Best For |
|-------|-------------|----------|
| `kimi` | Moonshot AI model | Chinese/English evaluation |
| `llama3` | Meta's Llama 3 | General evaluation |
| `mistral` | Mistral AI | Fast evaluation |
| `qwen` | Alibaba Qwen | Multilingual evaluation |
| `gemma` | Google Gemma | Lightweight evaluation |

## Architecture

The agent uses LangGraph for orchestration:

```
[detect_file_type] → [extract_content] → [load_rubric] → [load_prompt]
                                                           ↓
[save_results] ← [generate_report] ← [evaluate_with_llm]
```

Each node is a function that:
1. Receives the current state
2. Performs an operation
3. Returns updated state

## Troubleshooting

### Ollama Connection Issues

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama
ollama serve

# Pull a model
ollama pull kimi
```

### Missing Dependencies

```bash
pip install -r requirements.txt
```

### File Not Found

Ensure the submission file path is correct:

```bash
# Relative path
python langraph_evaluator_agent.py ./submissions/student1.html

# Absolute path
python langraph_evaluator_agent.py /path/to/submission.pdf
```

## File Structure

```
.
├── langraph_evaluator_agent.py  # Main agent
├── quickstart.py                # Quick start helper
├── example_usage.py             # Usage examples
├── parse_jupyter_html.py        # HTML parser
├── parse_pdf_submission.py      # PDF parser
├── prompt.txt                   # Evaluation guidelines
├── requirements.txt             # Dependencies
├── personal_loan_rubrics.json   # Rubric (optional)
└── model_deployment_rubric.json # Rubric (optional)
```

## License

MIT License - Feel free to use and modify for your evaluation needs.
