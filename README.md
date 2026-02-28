# Assignment Evaluator Agent

An AI-powered automated evaluation system using **LangGraph** and **Anthropic Claude** via **Azure Foundry**, with a modern ChatGPT-like web interface.

## Features

- **AI-Powered Evaluation**: Uses Claude (Opus/Sonnet) via Azure Foundry for intelligent grading
- **ChatGPT-like Web UI**: Modern chat interface built with FastAPI
- **Multiple Input Formats**: Supports HTML (Jupyter notebooks) and PDF submissions
- **Flexible Rubrics**: Upload rubrics as JSON or PDF (auto-converted)
- **Structured Feedback**: Section-wise marks with detailed comments
- **LangGraph Workflow**: Robust multi-step evaluation pipeline

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Azure Foundry

Create a `.env` file in the project root:

```bash
# Copy the example file
cp .env.example .env

# Edit with your credentials
nano .env
```

Required environment variables:

```env
AZURE_FOUNDRY_ENDPOINT='https://your-endpoint.services.ai.azure.com/anthropic/'
AZURE_FOUNDRY_API_KEY='your-api-key-here'
AZURE_FOUNDRY_MODEL='claude-opus-4-5'
AZURE_FOUNDRY_TEMPERATURE=1
```

### 3. Start the Web UI

```bash
uvicorn app:app --reload
```

Open your browser to: **http://localhost:8000**

---

## Web Interface Usage

### Step 1: Upload Rubric

- Click "Upload Rubric" button
- Select a **JSON** or **PDF** file containing your rubric
- PDF rubrics are automatically converted to JSON format

### Step 2: Upload Student Submission

- Click "Upload Submission" button
- Select an **HTML** (Jupyter notebook export) or **PDF** file

### Step 3: Evaluate

- Type in the chat: "Evaluate this submission" or "Grade the assignment"
- The AI will analyze the submission against the rubric
- Receive detailed feedback with section-wise marks

---

## Command Line Usage

### Evaluate a Single Submission

```bash
python langraph_evaluator_agent.py submission.html
```

### With Custom Rubric

```bash
python langraph_evaluator_agent.py submission.pdf --rubric my_rubric.json
```

### Convert PDF Rubric to JSON

```bash
python convert_rubric_pdf_to_json.py rubric.pdf

# Preview PDF text only
python convert_rubric_pdf_to_json.py rubric.pdf --preview

# Custom output path
python convert_rubric_pdf_to_json.py rubric.pdf output.json
```

---

## API Endpoints

The FastAPI backend provides these endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI interface |
| `/api/session` | POST | Create new evaluation session |
| `/api/upload/rubric` | POST | Upload rubric file (JSON/PDF) |
| `/api/upload/submission` | POST | Upload submission file (HTML/PDF) |
| `/api/chat` | POST | Send evaluation query |

### API Example

```python
import requests

# Create session
session = requests.post("http://localhost:8000/api/session").json()
session_id = session["session_id"]

# Upload rubric
with open("rubric.json", "rb") as f:
    requests.post(
        "http://localhost:8000/api/upload/rubric",
        files={"file": f},
        data={"session_id": session_id}
    )

# Upload submission
with open("submission.html", "rb") as f:
    requests.post(
        "http://localhost:8000/api/upload/submission",
        files={"file": f},
        data={"session_id": session_id}
    )

# Evaluate
response = requests.post(
    "http://localhost:8000/api/chat",
    json={"session_id": session_id, "message": "Evaluate this submission"}
)
print(response.json()["response"])
```

---

## Rubric JSON Format

Create rubrics in this structure:

```json
{
  "rubric_name": "Data Science Project Rubric",
  "total_points": 100,
  "sections": [
    {
      "section": "Data Preprocessing",
      "points": 20,
      "weightage": 20,
      "description": [
        "Data cleaning performed",
        "Missing values handled",
        "Feature engineering applied"
      ],
      "levels": {
        "80-100": ["Excellent preprocessing", "All edge cases handled"],
        "60-80": ["Good preprocessing", "Most issues addressed"],
        "<60": ["Minimal preprocessing", "Major issues remain"]
      }
    },
    {
      "section": "Model Building",
      "points": 30,
      "weightage": 30,
      "description": [
        "Appropriate model selected",
        "Hyperparameter tuning performed",
        "Model comparison done"
      ]
    }
  ]
}
```

---

## Project Structure

```
.
├── app.py                          # FastAPI web UI (main entry point)
├── langraph_evaluator_agent.py     # LangGraph evaluation workflow
├── convert_rubric_pdf_to_json.py   # PDF rubric converter
├── parse_jupyter_html.py           # Jupyter HTML parser
├── parse_pdf_submission.py         # PDF submission parser
├── requirements.txt                # Python dependencies
├── .env                            # Azure Foundry credentials
├── .env.example                    # Example environment file
└── prompt.txt                      # Evaluation prompt template
```

---

## Configuration Options

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AZURE_FOUNDRY_ENDPOINT` | Azure Foundry API endpoint | Required |
| `AZURE_FOUNDRY_API_KEY` | Azure Foundry API key | Required |
| `AZURE_FOUNDRY_MODEL` | Claude model to use | `claude-3-5-sonnet-20241022` |
| `AZURE_FOUNDRY_TEMPERATURE` | Response creativity (0-1) | `1` |

### Supported Models

- `claude-opus-4-5` - Most capable, best for complex evaluations
- `claude-3-5-sonnet-20241022` - Balanced performance and cost
- `claude-3-5-haiku-20241022` - Fastest, good for simple evaluations

---

## Troubleshooting

### Connection Errors

```
Error: Azure Foundry credentials not configured
```

**Solution**: Ensure `.env` file exists with valid credentials:

```bash
# Check if .env exists
cat .env

# Verify endpoint is correct (should end with /anthropic/)
```

### Import Errors

```
ModuleNotFoundError: No module named 'langchain_anthropic'
```

**Solution**: Install all dependencies:

```bash
pip install -r requirements.txt
```

### PDF Parsing Errors

```
Error extracting PDF text
```

**Solution**: Install PDF libraries:

```bash
pip install pdfplumber PyMuPDF Pillow
```

### 0/0 Marks in Evaluation

**Solution**: Ensure your rubric JSON has the correct structure. Each section needs:
- `section` or `name`: Section name
- `points` or `max_marks`: Maximum points

---

## Parser Scripts

### parse_jupyter_html.py

Parses Jupyter Notebook HTML exports:

```bash
python parse_jupyter_html.py notebook.html
```

Extracts:
- Code cells with content
- Markdown cells with text
- Cell outputs (stdout, errors)
- Plots and charts (as PNG files)

### parse_pdf_submission.py

Parses PDF submissions:

```bash
python parse_pdf_submission.py report.pdf

# With page images for vision models
python parse_pdf_submission.py report.pdf --images --dpi 200
```

Extracts:
- Text content (layout-aware)
- Tables (converted to Markdown)
- Embedded images/charts
- Page images (optional)

---

## Development

### Running in Development Mode

```bash
# Start with auto-reload
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Running Tests

```bash
# Test the evaluator agent directly
python langraph_evaluator_agent.py test_submission.html

# Test PDF converter
python convert_rubric_pdf_to_json.py test_rubric.pdf --preview
```

---

## License

This project is provided for educational evaluation purposes.
