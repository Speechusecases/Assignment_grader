# LangGraph Evaluator Agent

An AI-powered evaluation system for grading Data Science/ML/AI student submissions using LangGraph workflow and Ollama local models.

## Features

- 📄 **Multi-format Support**: Evaluates both Jupyter HTML exports and PDF submissions
- 🤖 **Local LLM Integration**: Uses Ollama models (kimi, llama3, mistral, etc.)
- 🔄 **LangGraph Workflow**: Structured evaluation pipeline with state management
- 📊 **Structured Feedback**: Section-wise evaluation with marks and actionable feedback
- 🖼️ **Image Extraction**: Extracts and includes plots/visualizations from submissions
- 💾 **Multiple Output Formats**: JSON and Markdown reports

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Or use the quickstart script:

```bash
python quickstart.py setup
```

### 2. Install and Start Ollama

Download Ollama from [https://ollama.com/download](https://ollama.com/download)

```bash
# Start Ollama server
ollama serve

# Pull models (in another terminal)
ollama pull kimi
ollama pull llama3
```

## Usage

### Command Line

```bash
# Evaluate an HTML notebook submission
python langraph_evaluator_agent.py submission.html

# Evaluate a PDF report
python langraph_evaluator_agent.py report.pdf

# Use a specific model
python langraph_evaluator_agent.py submission.html --model kimi

# Custom temperature (creativity vs determinism)
python langraph_evaluator_agent.py submission.html --temperature 0.5

# Skip image extraction (faster)
python langraph_evaluator_agent.py submission.html --no-images

# Custom output file
python langraph_evaluator_agent.py submission.html -o my_report.md
```

### Quick Start Script

```bash
# Setup everything
python quickstart.py setup

# Check Ollama status
python quickstart.py check

# Test with a file
python quickstart.py test submission.html

# Batch process directory
python quickstart.py batch ./submissions
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

print(f"Score: {results['total_marks']}/{results['max_marks']}")
print(f"Feedback: {results['feedback']}")
```

## Workflow Architecture

```
┌─────────────────┐
│  detect_file    │  Detect file type (HTML/PDF)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ extract_content │  Parse using existing scripts
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  load_rubric    │  Load evaluation criteria
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  load_prompt    │  Load prompt guidelines
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    evaluate     │  LLM evaluation with Ollama
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ generate_report │  Format evaluation results
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  save_results   │  Save JSON and Markdown files
└─────────────────┘
```

## Output Files

For each submission `submission.html` or `submission.pdf`, the agent generates:

1. **`submission.evaluation.json`** - Structured evaluation data
2. **`submission.evaluation.md`** - Human-readable evaluation report

### Example Output Structure

```json
{
  "file": "submission.html",
  "file_type": "html",
  "model": "kimi",
  "total_marks": 75,
  "max_marks": 100,
  "percentage": 75.0,
  "evaluations": {
    "sections": [
      {
        "name": "Business Understanding",
        "marks": 8,
        "max_marks": 10,
        "feedback": "..."
      }
    ],
    "recommendations": [...]
  }
}
```

## Configuration

### Rubric Files

The agent looks for rubric files in the same directory:
- `personal_loan_rubrics.json` - For personal loan prediction assignments
- `model_deployment_rubric.json` - For model deployment assignments

If these files are empty or missing, a default rubric is used.

### Prompt Guidelines

The agent reads `prompt.txt` for evaluation guidelines. If empty, default guidelines are used.

### Default Evaluation Criteria

1. **Business Understanding & Problem Definition** (10 marks)
2. **Data Understanding & Exploration** (15 marks)
3. **Data Preparation & Preprocessing** (15 marks)
4. **Modeling** (25 marks)
5. **Evaluation** (15 marks)
6. **Deployment & Conclusions** (10 marks)
7. **Code Quality & Documentation** (10 marks)

## Available Models

| Model | Description | Recommended For |
|-------|-------------|-----------------|
| `kimi` | Moonshot AI model | Chinese/English evaluation |
| `llama3` | Meta Llama 3 | General evaluation |
| `mistral` | Mistral AI | Fast inference |
| `qwen` | Alibaba Qwen | Code evaluation |
| `gemma` | Google Gemma | Lightweight |

## Integration with Existing Scripts

The agent reuses the existing parsing scripts:

- **`parse_jupyter_html.py`** - Extracts cells, outputs, and images from HTML exports
- **`parse_pdf_submission.py`** - Extracts text, tables, and images from PDFs

## Error Handling

The agent includes comprehensive error handling:
- File not found errors
- Unsupported file types
- Ollama connection issues
- Parsing errors
- LLM evaluation failures

Errors are logged and included in the output report.

## Requirements

- Python 3.9+
- Ollama installed and running
- Required Python packages (see `requirements.txt`)

## License

MIT License - See LICENSE file for details

## Contributing

Pull requests welcome! Please ensure:
1. Code follows existing style
2. Tests pass
3. Documentation is updated
