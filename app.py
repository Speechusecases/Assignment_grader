#!/usr/bin/env python3
"""
FastAPI Backend with ChatGPT-like UI for Assignment Evaluator

Features:
- Modern chat interface similar to ChatGPT
- File upload for rubrics and submissions
- Real-time evaluation using LangGraph agent
- Chat history and follow-up questions

Usage:
    uvicorn app:app --reload --port 8000
    Then open: http://localhost:8000
"""

import os
import sys
import json
import tempfile
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Load environment variables
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

# Import the evaluator agent
sys.path.insert(0, str(Path(__file__).parent))
from langraph_evaluator_agent import run_evaluation, EvaluatorConfig

# Import PDF to JSON converter
from convert_rubric_pdf_to_json import convert_rubric_pdf_to_json

# For chat functionality
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import AzureChatOpenAI

# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="Assignment Evaluator",
    description="AI-powered assignment evaluation using Claude via Azure Foundry",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# In-memory storage
# =============================================================================

sessions: Dict[str, Dict[str, Any]] = {}

# =============================================================================
# Pydantic Models
# =============================================================================

class ChatMessage(BaseModel):
    session_id: str
    message: str

class SessionData(BaseModel):
    session_id: str
    has_rubric: bool
    has_submission: bool
    rubric_name: Optional[str] = None
    submission_name: Optional[str] = None
    evaluation_done: bool
    total_marks: Optional[int] = None
    max_marks: Optional[int] = None

# =============================================================================
# Helper Functions
# =============================================================================

def get_llm():
    """Initialize the LLM, handling o-series reasoning models correctly."""
    from langraph_evaluator_agent import is_reasoning_model
    config = EvaluatorConfig()
    if not config.azure_endpoint or not config.api_key:
        return None

    if config.thinking_mode:
        # Reasoning model — omit temperature, use max_completion_tokens
        return AzureChatOpenAI(
            azure_endpoint=config.azure_endpoint,
            api_key=config.api_key,
            api_version=config.api_version,
            azure_deployment=config.azure_deployment,
            max_completion_tokens=config.max_tokens,
        )
    else:
        return AzureChatOpenAI(
            azure_endpoint=config.azure_endpoint,
            api_key=config.api_key,
            api_version=config.api_version,
            azure_deployment=config.azure_deployment,
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )

def get_or_create_session(session_id: str = None) -> str:
    """Get existing session or create new one."""
    if session_id and session_id in sessions:
        return session_id

    new_id = str(uuid.uuid4())
    sessions[new_id] = {
        "rubric_data": None,
        "rubric_name": None,
        "submission_path": None,
        "submission_name": None,
        "evaluation_result": None,
        "chat_history": [],
        "created_at": datetime.now().isoformat()
    }
    return new_id

# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the ChatGPT-like UI."""
    return get_chat_ui_html()

@app.post("/api/session")
async def create_session():
    """Create a new session."""
    session_id = get_or_create_session()
    return {"session_id": session_id}

@app.post("/api/session/{session_id}/reset")
async def reset_session(session_id: str):
    """Reset session for new evaluation — keeps rubric, clears submission and results."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]
    # Preserve rubric data
    rubric_data = session["rubric_data"]
    rubric_name = session["rubric_name"]

    # Clear submission and evaluation
    session["submission_path"] = None
    session["submission_name"] = None
    session["evaluation_result"] = None
    session["chat_history"] = []

    return {
        "session_id": session_id,
        "rubric_preserved": rubric_data is not None,
        "rubric_name": rubric_name
    }

@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    """Get session status."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]
    return SessionData(
        session_id=session_id,
        has_rubric=session["rubric_data"] is not None,
        has_submission=session["submission_path"] is not None,
        rubric_name=session["rubric_name"],
        submission_name=session["submission_name"],
        evaluation_done=session["evaluation_result"] is not None,
        total_marks=session["evaluation_result"]["total_marks"] if session["evaluation_result"] else None,
        max_marks=session["evaluation_result"]["max_marks"] if session["evaluation_result"] else None
    )

@app.post("/api/upload/rubric")
async def upload_rubric(
    file: UploadFile = File(...),
    session_id: str = Form(...)
):
    """Upload rubric file (JSON or PDF)."""
    if session_id not in sessions:
        session_id = get_or_create_session(session_id)

    file_ext = file.filename.split(".")[-1].lower()

    if file_ext not in ["json", "pdf"]:
        raise HTTPException(status_code=400, detail="Rubric must be a JSON or PDF file")

    try:
        content = await file.read()

        if file_ext == "json":
            # Parse JSON directly
            rubric_data = json.loads(content.decode("utf-8"))
            message = f"Rubric '{file.filename}' uploaded successfully"

        elif file_ext == "pdf":
            # Save PDF temporarily and convert to JSON
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, f"rubric_{session_id}.pdf")

            with open(temp_path, "wb") as f:
                f.write(content)

            try:
                print(f"Converting PDF rubric: {file.filename}")
                # Use the imported converter script (save_file=False for API usage)
                rubric_data = convert_rubric_pdf_to_json(temp_path, save_file=False)
                message = f"PDF rubric '{file.filename}' converted and loaded successfully"
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_path)
                except:
                    pass

        sessions[session_id]["rubric_data"] = rubric_data
        sessions[session_id]["rubric_name"] = file.filename

        return {
            "success": True,
            "session_id": session_id,
            "rubric_name": file.filename,
            "rubric_data": rubric_data,  # Include converted data for PDF
            "message": message
        }

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing rubric: {str(e)}")

@app.post("/api/upload/submission")
async def upload_submission(
    file: UploadFile = File(...),
    session_id: str = Form(...)
):
    """Upload submission file (HTML or PDF)."""
    if session_id not in sessions:
        session_id = get_or_create_session(session_id)

    file_ext = file.filename.split(".")[-1].lower()
    if file_ext not in ["html", "pdf"]:
        raise HTTPException(status_code=400, detail="Submission must be HTML or PDF")

    # Save to temp file
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, f"submission_{session_id}.{file_ext}")

    content = await file.read()
    with open(temp_path, "wb") as f:
        f.write(content)

    sessions[session_id]["submission_path"] = temp_path
    sessions[session_id]["submission_name"] = file.filename

    return {
        "success": True,
        "session_id": session_id,
        "submission_name": file.filename,
        "message": f"Submission '{file.filename}' uploaded successfully"
    }

@app.post("/api/chat")
async def chat(request: ChatMessage):
    """Handle chat messages and evaluation requests."""
    session_id = request.session_id
    message = request.message.strip()

    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions[session_id]

    # Add user message to history
    session["chat_history"].append({
        "role": "user",
        "content": message,
        "timestamp": datetime.now().isoformat()
    })

    # Check if this is an evaluation request
    is_eval_request = any(keyword in message.lower() for keyword in [
        "evaluate", "grade", "assess", "review", "score", "mark", "check", "analyze"
    ])

    # Handle different scenarios
    if not session["rubric_data"]:
        response = "Please upload a **rubric JSON file** first using the upload button above."

    elif not session["submission_path"]:
        response = "Please upload a **student submission** (HTML or PDF) using the upload button above."

    elif is_eval_request and not session["evaluation_result"]:
        # Run evaluation
        try:
            print(f"\n{'='*60}")
            print(f"Running evaluation for session: {session_id}")
            print(f"Submission: {session['submission_name']}")
            print(f"Rubric: {session['rubric_name']}")
            print(f"{'='*60}\n")

            results = run_evaluation(
                file_path=session["submission_path"],
                custom_rubric=session["rubric_data"]
            )

            session["evaluation_result"] = results

            if results.get("errors"):
                response = f"**Evaluation completed with errors:**\n\n" + "\n".join(results["errors"])
            else:
                response = f"""**Evaluation Complete!**

**Score: {results['total_marks']}/{results['max_marks']} ({results['percentage']}%)**

---

{results['feedback']}"""

        except Exception as e:
            response = f"**Error during evaluation:** {str(e)}"

    elif session["evaluation_result"]:
        # Handle follow-up questions
        llm = get_llm()
        if llm is None:
            response = "Error: LLM not configured. Please check Azure OpenAI credentials in your .env file."
        else:
            try:
                results = session["evaluation_result"]
                context = f"""
Evaluation Results:
- Submission: {session['submission_name']}
- Score: {results['total_marks']}/{results['max_marks']} ({results['percentage']}%)

Feedback:
{results['feedback'][:4000]}
"""

                messages = [
                    SystemMessage(content="""You are an expert assignment evaluator assistant.
Help answer questions about the evaluation results. Be specific and constructive.
If asked to evaluate again, refer to the existing results."""),
                    HumanMessage(content=f"Context:\n{context}"),
                ]

                # Add recent chat history
                for msg in session["chat_history"][-6:]:
                    if msg["role"] == "user":
                        messages.append(HumanMessage(content=msg["content"]))
                    else:
                        messages.append(AIMessage(content=msg["content"]))

                llm_response = llm.invoke(messages)
                response = llm_response.content

            except Exception as e:
                response = f"Error: {str(e)}"
    else:
        response = "Please type 'evaluate' or 'grade this assignment' to start the evaluation."

    # Add assistant response to history
    session["chat_history"].append({
        "role": "assistant",
        "content": response,
        "timestamp": datetime.now().isoformat()
    })

    return {
        "response": response,
        "evaluation_done": session["evaluation_result"] is not None,
        "total_marks": session["evaluation_result"]["total_marks"] if session["evaluation_result"] else None,
        "max_marks": session["evaluation_result"]["max_marks"] if session["evaluation_result"] else None
    }

@app.get("/api/history/{session_id}")
async def get_history(session_id: str):
    """Get chat history for a session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"history": sessions[session_id]["chat_history"]}

@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    if session_id in sessions:
        # Clean up temp file
        if sessions[session_id]["submission_path"]:
            try:
                os.unlink(sessions[session_id]["submission_path"])
            except:
                pass
        del sessions[session_id]
    return {"message": "Session deleted"}

# =============================================================================
# ChatGPT-like UI HTML
# =============================================================================

def get_chat_ui_html():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Assignment Evaluator - AI Grading Assistant</title>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism-tomorrow.min.css">
    <style>
        :root {
            /* GitHub Copilot / VS Code inspired colors */
            --bg-primary: #0d1117;
            --bg-secondary: #161b22;
            --bg-tertiary: #21262d;
            --bg-elevated: #30363d;
            --border-primary: #30363d;
            --border-secondary: #21262d;
            --text-primary: #e6edf3;
            --text-secondary: #8b949e;
            --text-muted: #6e7681;
            --accent-primary: #8b5cf6;
            --accent-secondary: #a78bfa;
            --accent-gradient: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%);
            --success: #3fb950;
            --success-bg: rgba(63, 185, 80, 0.15);
            --warning: #d29922;
            --warning-bg: rgba(210, 153, 34, 0.15);
            --error: #f85149;
            --error-bg: rgba(248, 81, 73, 0.15);
            --info: #58a6ff;
            --info-bg: rgba(88, 166, 255, 0.15);
            --shadow-sm: 0 1px 2px rgba(0,0,0,0.3);
            --shadow-md: 0 4px 12px rgba(0,0,0,0.4);
            --shadow-lg: 0 8px 24px rgba(0,0,0,0.5);
            --radius-sm: 6px;
            --radius-md: 8px;
            --radius-lg: 12px;
            --transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            height: 100vh;
            display: flex;
            overflow: hidden;
        }

        /* Custom Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: transparent;
        }
        ::-webkit-scrollbar-thumb {
            background: var(--bg-elevated);
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: var(--text-muted);
        }

        /* ============ Sidebar ============ */
        .sidebar {
            width: 280px;
            background-color: var(--bg-secondary);
            display: flex;
            flex-direction: column;
            border-right: 1px solid var(--border-primary);
            flex-shrink: 0;
            transition: margin-left 0.3s cubic-bezier(0.4, 0, 0.2, 1),
                        opacity 0.3s cubic-bezier(0.4, 0, 0.2, 1),
                        visibility 0.3s;
            position: relative;
            z-index: 10;
        }

        .sidebar.collapsed {
            margin-left: -280px;
            opacity: 0;
            visibility: hidden;
        }

        .sidebar-header {
            padding: 16px;
            border-bottom: 1px solid var(--border-primary);
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .logo {
            width: 36px;
            height: 36px;
            background: var(--accent-gradient);
            border-radius: var(--radius-md);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            color: white;
            box-shadow: var(--shadow-md);
        }

        .logo-text {
            font-weight: 600;
            font-size: 15px;
        }

        .logo-text span {
            color: var(--accent-secondary);
        }

        .sidebar-close-btn {
            margin-left: auto;
            width: 28px;
            height: 28px;
            border-radius: var(--radius-sm);
            background: transparent;
            border: none;
            color: var(--text-muted);
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 14px;
            transition: var(--transition);
        }

        .sidebar-close-btn:hover {
            background: var(--bg-tertiary);
            color: var(--text-primary);
        }

        .new-session-btn {
            margin: 16px;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            padding: 10px 16px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-primary);
            border-radius: var(--radius-md);
            color: var(--text-primary);
            cursor: pointer;
            font-size: 13px;
            font-weight: 500;
            transition: var(--transition);
        }

        .new-session-btn:hover {
            background: var(--bg-elevated);
            border-color: var(--accent-primary);
        }

        .new-session-btn i {
            font-size: 12px;
        }

        /* Upload Section */
        .upload-section {
            padding: 16px;
            flex: 1;
            overflow-y: auto;
        }

        .section-label {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 11px;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 12px;
        }

        .section-label i {
            font-size: 10px;
        }

        .file-upload-card {
            background: var(--bg-tertiary);
            border: 1px solid var(--border-primary);
            border-radius: var(--radius-md);
            padding: 14px;
            margin-bottom: 12px;
            transition: var(--transition);
        }

        .file-upload-card:hover {
            border-color: var(--text-muted);
        }

        .file-upload-card.uploaded {
            border-color: var(--success);
            background: var(--success-bg);
        }

        .file-upload-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 10px;
        }

        .file-type-badge {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 12px;
            font-weight: 500;
            color: var(--text-secondary);
        }

        .file-type-badge i {
            font-size: 14px;
            color: var(--accent-primary);
        }

        .file-status-icon {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 10px;
        }

        .file-status-icon.pending {
            background: var(--bg-elevated);
            color: var(--text-muted);
        }

        .file-status-icon.success {
            background: var(--success);
            color: white;
        }

        .upload-drop-zone {
            border: 1px dashed var(--border-primary);
            border-radius: var(--radius-sm);
            padding: 16px;
            text-align: center;
            cursor: pointer;
            transition: var(--transition);
        }

        .upload-drop-zone:hover {
            border-color: var(--accent-primary);
            background: rgba(139, 92, 246, 0.05);
        }

        .upload-drop-zone.dragover {
            border-color: var(--accent-primary);
            background: rgba(139, 92, 246, 0.1);
        }

        .upload-drop-zone i {
            font-size: 24px;
            color: var(--text-muted);
            margin-bottom: 8px;
        }

        .upload-drop-zone p {
            font-size: 12px;
            color: var(--text-secondary);
            margin-bottom: 4px;
        }

        .upload-drop-zone .hint {
            font-size: 11px;
            color: var(--text-muted);
        }

        .uploaded-file-info {
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px;
            background: rgba(63, 185, 80, 0.1);
            border-radius: var(--radius-sm);
        }

        .uploaded-file-info i {
            color: var(--success);
            font-size: 16px;
        }

        .uploaded-file-info .file-details {
            flex: 1;
            min-width: 0;
        }

        .uploaded-file-info .file-name {
            font-size: 12px;
            font-weight: 500;
            color: var(--text-primary);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .uploaded-file-info .file-meta {
            font-size: 10px;
            color: var(--text-muted);
        }

        .remove-file-btn {
            width: 24px;
            height: 24px;
            border-radius: var(--radius-sm);
            background: transparent;
            border: none;
            color: var(--text-muted);
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: var(--transition);
        }

        .remove-file-btn:hover {
            background: var(--error-bg);
            color: var(--error);
        }

        /* Status Card */
        .status-card {
            margin: 16px;
            padding: 14px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-primary);
            border-radius: var(--radius-md);
        }

        .status-card.ready {
            border-color: var(--success);
            background: var(--success-bg);
        }

        .status-card.evaluated {
            border-color: var(--accent-primary);
            background: rgba(139, 92, 246, 0.15);
        }

        .status-indicator {
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: var(--text-muted);
        }

        .status-dot.ready {
            background: var(--success);
            box-shadow: 0 0 8px var(--success);
        }

        .status-dot.evaluated {
            background: var(--accent-primary);
            box-shadow: 0 0 8px var(--accent-primary);
        }

        .status-text {
            font-size: 13px;
            font-weight: 500;
        }

        .status-subtext {
            font-size: 11px;
            color: var(--text-muted);
            margin-top: 2px;
        }

        /* Score Display */
        .score-display {
            margin-top: 12px;
            padding: 12px;
            background: var(--bg-primary);
            border-radius: var(--radius-sm);
        }

        .score-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 8px;
        }

        .score-label {
            font-size: 11px;
            font-weight: 600;
            color: var(--text-muted);
            text-transform: uppercase;
        }

        .score-value {
            font-size: 18px;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
        }

        .score-value.excellent { color: var(--success); }
        .score-value.good { color: var(--info); }
        .score-value.average { color: var(--warning); }
        .score-value.poor { color: var(--error); }

        .score-bar {
            height: 6px;
            background: var(--bg-tertiary);
            border-radius: 3px;
            overflow: hidden;
        }

        .score-bar-fill {
            height: 100%;
            border-radius: 3px;
            transition: width 0.5s ease-out;
        }

        .score-bar-fill.excellent { background: var(--success); }
        .score-bar-fill.good { background: var(--info); }
        .score-bar-fill.average { background: var(--warning); }
        .score-bar-fill.poor { background: var(--error); }

        /* Keyboard Shortcuts */
        .shortcuts-section {
            padding: 16px;
            border-top: 1px solid var(--border-primary);
        }

        .shortcut-item {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 6px 0;
        }

        .shortcut-label {
            font-size: 12px;
            color: var(--text-secondary);
        }

        .shortcut-keys {
            display: flex;
            gap: 4px;
        }

        .key {
            padding: 2px 6px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-primary);
            border-radius: 4px;
            font-size: 10px;
            font-family: 'JetBrains Mono', monospace;
            color: var(--text-muted);
        }

        /* ============ Main Content ============ */
        .main-wrapper {
            flex: 1;
            display: flex;
            flex-direction: column;
            min-width: 0;
        }

        /* Header */
        .main-header {
            padding: 12px 24px;
            background: var(--bg-secondary);
            border-bottom: 1px solid var(--border-primary);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .header-title {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .header-title h1 {
            font-size: 15px;
            font-weight: 600;
        }

        .model-pill {
            display: flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-primary);
            border-radius: 20px;
            font-size: 11px;
            color: var(--text-secondary);
        }

        .model-pill .dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: var(--success);
        }

        .header-actions {
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .header-btn {
            width: 32px;
            height: 32px;
            border-radius: var(--radius-sm);
            background: var(--bg-tertiary);
            border: 1px solid var(--border-primary);
            color: var(--text-secondary);
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: var(--transition);
        }

        .header-btn:hover {
            background: var(--bg-elevated);
            color: var(--text-primary);
        }

        /* Chat Container */
        .chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 24px;
            display: flex;
            flex-direction: column;
        }

        /* Welcome Screen */
        .welcome-screen {
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            padding: 40px;
            max-width: 700px;
            margin: 0 auto;
        }

        .welcome-icon {
            width: 80px;
            height: 80px;
            background: var(--accent-gradient);
            border-radius: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 36px;
            color: white;
            margin-bottom: 24px;
            box-shadow: var(--shadow-lg);
        }

        .welcome-title {
            font-size: 28px;
            font-weight: 700;
            margin-bottom: 12px;
            background: linear-gradient(135deg, var(--text-primary) 0%, var(--accent-secondary) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }

        .welcome-subtitle {
            font-size: 15px;
            color: var(--text-secondary);
            line-height: 1.6;
            margin-bottom: 32px;
        }

        .feature-cards {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 16px;
            width: 100%;
            margin-bottom: 32px;
        }

        .feature-card {
            padding: 20px;
            background: var(--bg-secondary);
            border: 1px solid var(--border-primary);
            border-radius: var(--radius-lg);
            text-align: left;
            transition: var(--transition);
        }

        .feature-card:hover {
            border-color: var(--accent-primary);
            transform: translateY(-2px);
        }

        .feature-card i {
            font-size: 20px;
            color: var(--accent-primary);
            margin-bottom: 12px;
        }

        .feature-card h3 {
            font-size: 13px;
            font-weight: 600;
            margin-bottom: 6px;
        }

        .feature-card p {
            font-size: 12px;
            color: var(--text-muted);
            line-height: 1.5;
        }

        .quick-prompts {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            justify-content: center;
        }

        .quick-prompt {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 10px 16px;
            background: var(--bg-secondary);
            border: 1px solid var(--border-primary);
            border-radius: var(--radius-md);
            color: var(--text-primary);
            cursor: pointer;
            font-size: 13px;
            transition: var(--transition);
        }

        .quick-prompt:hover {
            border-color: var(--accent-primary);
            background: var(--bg-tertiary);
        }

        .quick-prompt i {
            color: var(--accent-primary);
            font-size: 14px;
        }

        /* Messages */
        .messages-container {
            max-width: 900px;
            margin: 0 auto;
            width: 100%;
        }

        .message {
            display: flex;
            gap: 16px;
            padding: 20px 0;
            animation: fadeIn 0.3s ease-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .message + .message {
            border-top: 1px solid var(--border-secondary);
        }

        .message-avatar {
            width: 36px;
            height: 36px;
            border-radius: var(--radius-md);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 14px;
            flex-shrink: 0;
        }

        .message-avatar.user {
            background: var(--bg-elevated);
            color: var(--text-secondary);
            border: 1px solid var(--border-primary);
        }

        .message-avatar.assistant {
            background: var(--accent-gradient);
            color: white;
        }

        .message-avatar.system {
            background: var(--info-bg);
            color: var(--info);
            border: 1px solid var(--info);
        }

        .message-body {
            flex: 1;
            min-width: 0;
        }

        .message-header {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 8px;
        }

        .message-sender {
            font-size: 13px;
            font-weight: 600;
        }

        .message-time {
            font-size: 11px;
            color: var(--text-muted);
        }

        .message-content {
            font-size: 14px;
            line-height: 1.7;
            color: var(--text-primary);
        }

        .message-content p {
            margin: 8px 0;
        }

        .message-content h1,
        .message-content h2,
        .message-content h3,
        .message-content h4 {
            margin: 20px 0 12px;
            font-weight: 600;
        }

        .message-content h1 { font-size: 1.5em; }
        .message-content h2 { font-size: 1.3em; color: var(--accent-secondary); }
        .message-content h3 { font-size: 1.15em; }
        .message-content h4 { font-size: 1em; }

        .message-content ul, .message-content ol {
            margin: 12px 0 12px 24px;
        }

        .message-content li {
            margin: 6px 0;
        }

        .message-content strong {
            color: var(--text-primary);
            font-weight: 600;
        }

        .message-content hr {
            border: none;
            border-top: 1px solid var(--border-primary);
            margin: 20px 0;
        }

        .message-content code {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.9em;
            background: var(--bg-tertiary);
            padding: 2px 8px;
            border-radius: 4px;
            color: var(--accent-secondary);
        }

        .message-content pre {
            position: relative;
            background: var(--bg-secondary);
            border: 1px solid var(--border-primary);
            border-radius: var(--radius-md);
            margin: 16px 0;
            overflow: hidden;
        }

        .message-content pre code {
            display: block;
            padding: 16px;
            overflow-x: auto;
            background: transparent;
            color: var(--text-primary);
        }

        .code-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 8px 12px;
            background: var(--bg-tertiary);
            border-bottom: 1px solid var(--border-primary);
            font-size: 12px;
            color: var(--text-muted);
        }

        .copy-btn {
            display: flex;
            align-items: center;
            gap: 6px;
            padding: 4px 10px;
            background: var(--bg-elevated);
            border: 1px solid var(--border-primary);
            border-radius: var(--radius-sm);
            color: var(--text-secondary);
            cursor: pointer;
            font-size: 11px;
            transition: var(--transition);
        }

        .copy-btn:hover {
            background: var(--bg-primary);
            color: var(--text-primary);
        }

        .copy-btn.copied {
            background: var(--success-bg);
            border-color: var(--success);
            color: var(--success);
        }

        /* Evaluation Result Card */
        .eval-result-card {
            background: var(--bg-secondary);
            border: 1px solid var(--border-primary);
            border-radius: var(--radius-lg);
            margin: 16px 0;
            overflow: hidden;
        }

        .eval-result-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 16px 20px;
            background: var(--accent-gradient);
        }

        .eval-result-title {
            display: flex;
            align-items: center;
            gap: 10px;
            color: white;
            font-weight: 600;
        }

        .eval-score-big {
            font-size: 24px;
            font-weight: 700;
            font-family: 'JetBrains Mono', monospace;
            color: white;
        }

        .eval-result-body {
            padding: 20px;
        }

        /* Message Actions */
        .message-actions {
            display: flex;
            gap: 8px;
            margin-top: 12px;
            opacity: 0;
            transition: var(--transition);
        }

        .message:hover .message-actions {
            opacity: 1;
        }

        .action-btn {
            display: flex;
            align-items: center;
            gap: 6px;
            padding: 6px 10px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-primary);
            border-radius: var(--radius-sm);
            color: var(--text-muted);
            cursor: pointer;
            font-size: 11px;
            transition: var(--transition);
        }

        .action-btn:hover {
            background: var(--bg-elevated);
            color: var(--text-primary);
        }

        /* Loading */
        .loading-indicator {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 12px 16px;
            background: var(--bg-secondary);
            border-radius: var(--radius-md);
        }

        .loading-spinner {
            width: 20px;
            height: 20px;
            border: 2px solid var(--border-primary);
            border-top-color: var(--accent-primary);
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .loading-text {
            font-size: 13px;
            color: var(--text-secondary);
        }

        .loading-dots {
            display: inline-flex;
            gap: 4px;
        }

        .loading-dots span {
            width: 6px;
            height: 6px;
            background: var(--accent-primary);
            border-radius: 50%;
            animation: pulse 1.4s ease-in-out infinite;
        }

        .loading-dots span:nth-child(2) { animation-delay: 0.2s; }
        .loading-dots span:nth-child(3) { animation-delay: 0.4s; }

        @keyframes pulse {
            0%, 80%, 100% { opacity: 0.3; transform: scale(0.8); }
            40% { opacity: 1; transform: scale(1); }
        }

        /* ============ Input Area ============ */
        .input-area {
            padding: 16px 24px 24px;
            background: var(--bg-primary);
            border-top: 1px solid var(--border-primary);
        }

        .input-wrapper {
            max-width: 900px;
            margin: 0 auto;
        }

        .input-container {
            display: flex;
            align-items: flex-end;
            gap: 12px;
            background: var(--bg-secondary);
            border: 1px solid var(--border-primary);
            border-radius: var(--radius-lg);
            padding: 12px 16px;
            transition: var(--transition);
        }

        .input-container:focus-within {
            border-color: var(--accent-primary);
            box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.1);
        }

        .input-container textarea {
            flex: 1;
            background: transparent;
            border: none;
            color: var(--text-primary);
            font-size: 14px;
            font-family: inherit;
            resize: none;
            outline: none;
            max-height: 150px;
            line-height: 1.5;
        }

        .input-container textarea::placeholder {
            color: var(--text-muted);
        }

        .input-actions {
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .attach-btn {
            width: 36px;
            height: 36px;
            border-radius: var(--radius-md);
            background: transparent;
            border: 1px solid var(--border-primary);
            color: var(--text-muted);
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: var(--transition);
        }

        .attach-btn:hover {
            background: var(--bg-tertiary);
            color: var(--text-primary);
        }

        .send-btn {
            width: 36px;
            height: 36px;
            border-radius: var(--radius-md);
            background: var(--accent-gradient);
            border: none;
            color: white;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: var(--transition);
            box-shadow: var(--shadow-sm);
        }

        .send-btn:hover {
            transform: scale(1.05);
            box-shadow: var(--shadow-md);
        }

        .send-btn:disabled {
            background: var(--bg-elevated);
            color: var(--text-muted);
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }

        .input-footer {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-top: 10px;
            padding: 0 4px;
        }

        .input-hint {
            font-size: 11px;
            color: var(--text-muted);
        }

        .input-hint kbd {
            padding: 2px 5px;
            background: var(--bg-tertiary);
            border: 1px solid var(--border-primary);
            border-radius: 3px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 10px;
        }

        .char-count {
            font-size: 11px;
            color: var(--text-muted);
            font-family: 'JetBrains Mono', monospace;
        }

        /* Hidden file inputs */
        input[type="file"] {
            display: none;
        }

        /* ============ Responsive ============ */
        @media (max-width: 1024px) {
            .sidebar {
                width: 260px;
            }
            .feature-cards {
                grid-template-columns: 1fr;
            }
        }

        @media (max-width: 768px) {
            .sidebar {
                position: fixed;
                left: 0;
                top: 0;
                height: 100%;
                z-index: 100;
                margin-left: 0;
                box-shadow: var(--shadow-lg);
            }
            .sidebar.collapsed {
                left: -280px;
                margin-left: 0;
            }
            .sidebar-overlay {
                display: block;
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0,0,0,0.5);
                z-index: 99;
            }
            .sidebar-overlay.hidden {
                display: none;
            }
            .feature-cards {
                display: none;
            }
        }

        /* ============ Toast Notifications ============ */
        .toast-container {
            position: fixed;
            bottom: 24px;
            right: 24px;
            z-index: 1000;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .toast {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 14px 18px;
            background: var(--bg-secondary);
            border: 1px solid var(--border-primary);
            border-radius: var(--radius-md);
            box-shadow: var(--shadow-lg);
            animation: slideIn 0.3s ease-out;
        }

        .toast.success {
            border-color: var(--success);
            background: var(--success-bg);
        }

        .toast.error {
            border-color: var(--error);
            background: var(--error-bg);
        }

        @keyframes slideIn {
            from { opacity: 0; transform: translateX(100px); }
            to { opacity: 1; transform: translateX(0); }
        }

        .toast-icon {
            font-size: 18px;
        }

        .toast.success .toast-icon { color: var(--success); }
        .toast.error .toast-icon { color: var(--error); }

        .toast-content {
            flex: 1;
        }

        .toast-title {
            font-size: 13px;
            font-weight: 600;
            margin-bottom: 2px;
        }

        .toast-message {
            font-size: 12px;
            color: var(--text-secondary);
        }
    </style>
</head>
<body>
    <!-- Sidebar -->
    <aside class="sidebar" id="sidebar">
        <div class="sidebar-header">
            <div class="logo">
                <i class="fas fa-graduation-cap"></i>
            </div>
            <div class="logo-text">
                Grader<span>AI</span>
            </div>
            <button class="sidebar-close-btn" onclick="toggleSidebar()" title="Close sidebar">
                <i class="fas fa-chevron-left"></i>
            </button>
        </div>

        <button class="new-session-btn" onclick="newChat()">
            <i class="fas fa-arrow-right"></i>
            Next Submission
        </button>

        <div class="upload-section">
            <div class="section-label">
                <i class="fas fa-folder"></i>
                Files
            </div>

            <!-- Rubric Upload Card -->
            <div class="file-upload-card" id="rubricCard">
                <div class="file-upload-header">
                    <div class="file-type-badge">
                        <i class="fas fa-file-alt"></i>
                        Rubric
                    </div>
                    <div class="file-status-icon pending" id="rubricStatusIcon">
                        <i class="fas fa-circle"></i>
                    </div>
                </div>
                <input type="file" id="rubricInput" accept=".json,.pdf" onchange="uploadRubric(this)">
                <div class="upload-drop-zone" id="rubricDropZone" onclick="document.getElementById('rubricInput').click()">
                    <i class="fas fa-cloud-upload-alt"></i>
                    <p>Drop rubric file here</p>
                    <span class="hint">JSON or PDF format</span>
                </div>
                <div class="uploaded-file-info" id="rubricFileInfo" style="display: none;">
                    <i class="fas fa-file-check"></i>
                    <div class="file-details">
                        <div class="file-name" id="rubricFileName"></div>
                        <div class="file-meta" id="rubricFileMeta">Loaded successfully</div>
                    </div>
                    <button class="remove-file-btn" onclick="removeRubric(event)">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
            </div>

            <!-- Submission Upload Card -->
            <div class="file-upload-card" id="submissionCard">
                <div class="file-upload-header">
                    <div class="file-type-badge">
                        <i class="fas fa-file-code"></i>
                        Submission
                    </div>
                    <div class="file-status-icon pending" id="submissionStatusIcon">
                        <i class="fas fa-circle"></i>
                    </div>
                </div>
                <input type="file" id="submissionInput" accept=".html,.pdf" onchange="uploadSubmission(this)">
                <div class="upload-drop-zone" id="submissionDropZone" onclick="document.getElementById('submissionInput').click()">
                    <i class="fas fa-cloud-upload-alt"></i>
                    <p>Drop submission file here</p>
                    <span class="hint">HTML or PDF format</span>
                </div>
                <div class="uploaded-file-info" id="submissionFileInfo" style="display: none;">
                    <i class="fas fa-file-check"></i>
                    <div class="file-details">
                        <div class="file-name" id="submissionFileName"></div>
                        <div class="file-meta" id="submissionFileMeta">Loaded successfully</div>
                    </div>
                    <button class="remove-file-btn" onclick="removeSubmission(event)">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
            </div>
        </div>

        <!-- Status Card -->
        <div class="status-card" id="statusCard">
            <div class="status-indicator">
                <div class="status-dot" id="statusDot"></div>
                <div>
                    <div class="status-text" id="statusText">Upload files to begin</div>
                    <div class="status-subtext" id="statusSubtext">Rubric + Submission required</div>
                </div>
            </div>
            <div class="score-display" id="scoreDisplay" style="display: none;">
                <div class="score-header">
                    <span class="score-label">Final Score</span>
                    <span class="score-value" id="scoreValue">0/0</span>
                </div>
                <div class="score-bar">
                    <div class="score-bar-fill" id="scoreBarFill" style="width: 0%"></div>
                </div>
            </div>
        </div>

        <!-- Shortcuts -->
        <div class="shortcuts-section">
            <div class="section-label">
                <i class="fas fa-keyboard"></i>
                Shortcuts
            </div>
            <div class="shortcut-item">
                <span class="shortcut-label">Send message</span>
                <div class="shortcut-keys">
                    <span class="key">Enter</span>
                </div>
            </div>
            <div class="shortcut-item">
                <span class="shortcut-label">New line</span>
                <div class="shortcut-keys">
                    <span class="key">Shift</span>
                    <span class="key">Enter</span>
                </div>
            </div>
            <div class="shortcut-item">
                <span class="shortcut-label">New session</span>
                <div class="shortcut-keys">
                    <span class="key">Ctrl</span>
                    <span class="key">N</span>
                </div>
            </div>
        </div>
    </aside>

    <!-- Mobile Sidebar Overlay -->
    <div class="sidebar-overlay hidden" id="sidebarOverlay" onclick="toggleSidebar()"></div>

    <!-- Main Content -->
    <main class="main-wrapper">
        <header class="main-header">
            <div class="header-title">
                <button class="header-btn" onclick="toggleSidebar()" id="menuBtn" title="Toggle sidebar">
                    <i class="fas fa-bars"></i>
                </button>
                <h1>Assignment Evaluator</h1>
            </div>
            <div class="header-actions">
                <div class="model-pill">
                    <span class="dot"></span>
                    GPT via Azure OpenAI
                </div>
                <button class="header-btn" onclick="toggleTheme()" title="Toggle theme">
                    <i class="fas fa-moon"></i>
                </button>
            </div>
        </header>

        <div class="chat-container" id="chatContainer">
            <!-- Welcome Screen -->
            <div class="welcome-screen" id="welcomeScreen">
                <div class="welcome-icon">
                    <i class="fas fa-graduation-cap"></i>
                </div>
                <h2 class="welcome-title">AI-Powered Assignment Grading</h2>
                <p class="welcome-subtitle">
                    Upload a rubric and student submission, then let AI provide detailed feedback with marks for each section. Get instant, consistent evaluations.
                </p>

                <div class="feature-cards">
                    <div class="feature-card">
                        <i class="fas fa-bolt"></i>
                        <h3>Instant Feedback</h3>
                        <p>Get comprehensive evaluations in seconds with detailed breakdowns.</p>
                    </div>
                    <div class="feature-card">
                        <i class="fas fa-balance-scale"></i>
                        <h3>Consistent Grading</h3>
                        <p>AI ensures fair and uniform assessment across all submissions.</p>
                    </div>
                    <div class="feature-card">
                        <i class="fas fa-comments"></i>
                        <h3>Ask Follow-ups</h3>
                        <p>Chat with AI to clarify scores or get improvement suggestions.</p>
                    </div>
                </div>

                <div class="quick-prompts">
                    <button class="quick-prompt" onclick="sendMessage('Evaluate this assignment')">
                        <i class="fas fa-play-circle"></i>
                        Evaluate Submission
                    </button>
                    <button class="quick-prompt" onclick="sendMessage('What are the main strengths of this work?')">
                        <i class="fas fa-star"></i>
                        Show Strengths
                    </button>
                    <button class="quick-prompt" onclick="sendMessage('What areas need improvement?')">
                        <i class="fas fa-lightbulb"></i>
                        Suggest Improvements
                    </button>
                    <button class="quick-prompt" onclick="sendMessage('Provide a summary of the evaluation')">
                        <i class="fas fa-file-alt"></i>
                        Get Summary
                    </button>
                </div>
            </div>

            <!-- Messages will be appended here -->
            <div class="messages-container" id="messagesContainer" style="display: none;"></div>
        </div>

        <div class="input-area">
            <div class="input-wrapper">
                <div class="input-container">
                    <textarea
                        id="messageInput"
                        placeholder="Ask me to evaluate the submission or ask follow-up questions..."
                        rows="1"
                        onkeydown="handleKeyDown(event)"
                        oninput="autoResize(this); updateCharCount(this)"
                    ></textarea>
                    <div class="input-actions">
                        <button class="send-btn" id="sendBtn" onclick="sendMessage()">
                            <i class="fas fa-arrow-up"></i>
                        </button>
                    </div>
                </div>
                <div class="input-footer">
                    <span class="input-hint">
                        <kbd>Enter</kbd> to send, <kbd>Shift + Enter</kbd> for new line
                    </span>
                    <span class="char-count" id="charCount"></span>
                </div>
            </div>
        </div>
    </main>

    <!-- Toast Container -->
    <div class="toast-container" id="toastContainer"></div>

    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-python.min.js"></script>
    <script>
        // ============ State ============
        let sessionId = null;
        let isLoading = false;
        let rubricUploaded = false;
        let submissionUploaded = false;

        // ============ Initialize ============
        window.onload = async function() {
            await createSession();
            setupDragAndDrop();
            setupKeyboardShortcuts();
            checkResponsive();
        };

        window.onresize = checkResponsive;

        function checkResponsive() {
            // Hamburger menu is always visible — no need to hide it
            const sidebar = document.getElementById('sidebar');
            if (window.innerWidth <= 768) {
                // On mobile, start collapsed
                if (!sidebar.dataset.userToggled) {
                    sidebar.classList.add('collapsed');
                }
            }
        }

        function toggleSidebar() {
            const sidebar = document.getElementById('sidebar');
            const overlay = document.getElementById('sidebarOverlay');
            sidebar.classList.toggle('collapsed');
            sidebar.dataset.userToggled = 'true';
            // Handle mobile overlay
            if (overlay) {
                const isCollapsed = sidebar.classList.contains('collapsed');
                overlay.classList.toggle('hidden', isCollapsed);
            }
        }

        // ============ Session Management ============
        async function createSession() {
            try {
                const response = await fetch('/api/session', { method: 'POST' });
                const data = await response.json();
                sessionId = data.session_id;
                console.log('Session created:', sessionId);
            } catch (error) {
                console.error('Failed to create session:', error);
                showToast('error', 'Connection Error', 'Failed to connect to server');
            }
        }

        // ============ File Upload ============
        function setupDragAndDrop() {
            ['rubricDropZone', 'submissionDropZone'].forEach(id => {
                const zone = document.getElementById(id);
                const input = document.getElementById(id.replace('DropZone', 'Input'));

                zone.addEventListener('dragover', (e) => {
                    e.preventDefault();
                    zone.classList.add('dragover');
                });

                zone.addEventListener('dragleave', () => {
                    zone.classList.remove('dragover');
                });

                zone.addEventListener('drop', (e) => {
                    e.preventDefault();
                    zone.classList.remove('dragover');
                    const files = e.dataTransfer.files;
                    if (files.length) {
                        input.files = files;
                        input.dispatchEvent(new Event('change'));
                    }
                });
            });
        }

        async function uploadRubric(input) {
            if (!input.files.length) return;

            const file = input.files[0];
            const isPdf = file.name.toLowerCase().endsWith('.pdf');
            const formData = new FormData();
            formData.append('file', file);
            formData.append('session_id', sessionId);

            // Show loading
            const dropZone = document.getElementById('rubricDropZone');
            const originalContent = dropZone.innerHTML;
            dropZone.innerHTML = '<div class="loading-spinner"></div><p>Processing...</p>';

            try {
                const response = await fetch('/api/upload/rubric', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (response.ok) {
                    rubricUploaded = true;
                    document.getElementById('rubricCard').classList.add('uploaded');
                    document.getElementById('rubricDropZone').style.display = 'none';
                    document.getElementById('rubricFileInfo').style.display = 'flex';
                    document.getElementById('rubricFileName').textContent = file.name;
                    document.getElementById('rubricFileMeta').textContent = isPdf ? 'PDF converted successfully' : 'JSON loaded';
                    document.getElementById('rubricStatusIcon').className = 'file-status-icon success';
                    document.getElementById('rubricStatusIcon').innerHTML = '<i class="fas fa-check"></i>';

                    updateStatus();
                    showToast('success', 'Rubric Loaded', `${file.name} is ready`);
                } else {
                    dropZone.innerHTML = originalContent;
                    showToast('error', 'Upload Failed', data.detail || 'Failed to process rubric');
                }
            } catch (error) {
                dropZone.innerHTML = originalContent;
                showToast('error', 'Upload Failed', error.message);
            }
        }

        async function uploadSubmission(input) {
            if (!input.files.length) return;

            const file = input.files[0];
            const formData = new FormData();
            formData.append('file', file);
            formData.append('session_id', sessionId);

            // Show loading
            const dropZone = document.getElementById('submissionDropZone');
            const originalContent = dropZone.innerHTML;
            dropZone.innerHTML = '<div class="loading-spinner"></div><p>Processing...</p>';

            try {
                const response = await fetch('/api/upload/submission', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (response.ok) {
                    submissionUploaded = true;
                    document.getElementById('submissionCard').classList.add('uploaded');
                    document.getElementById('submissionDropZone').style.display = 'none';
                    document.getElementById('submissionFileInfo').style.display = 'flex';
                    document.getElementById('submissionFileName').textContent = file.name;
                    document.getElementById('submissionFileMeta').textContent = file.name.endsWith('.pdf') ? 'PDF submission' : 'HTML submission';
                    document.getElementById('submissionStatusIcon').className = 'file-status-icon success';
                    document.getElementById('submissionStatusIcon').innerHTML = '<i class="fas fa-check"></i>';

                    updateStatus();
                    showToast('success', 'Submission Loaded', `${file.name} is ready`);
                } else {
                    dropZone.innerHTML = originalContent;
                    showToast('error', 'Upload Failed', data.detail || 'Failed to process submission');
                }
            } catch (error) {
                dropZone.innerHTML = originalContent;
                showToast('error', 'Upload Failed', error.message);
            }
        }

        function removeRubric(event) {
            event.stopPropagation();
            rubricUploaded = false;
            document.getElementById('rubricCard').classList.remove('uploaded');
            document.getElementById('rubricDropZone').style.display = 'block';
            document.getElementById('rubricFileInfo').style.display = 'none';
            document.getElementById('rubricStatusIcon').className = 'file-status-icon pending';
            document.getElementById('rubricStatusIcon').innerHTML = '<i class="fas fa-circle"></i>';
            document.getElementById('rubricInput').value = '';
            updateStatus();
        }

        function removeSubmission(event) {
            event.stopPropagation();
            submissionUploaded = false;
            document.getElementById('submissionCard').classList.remove('uploaded');
            document.getElementById('submissionDropZone').style.display = 'block';
            document.getElementById('submissionFileInfo').style.display = 'none';
            document.getElementById('submissionStatusIcon').className = 'file-status-icon pending';
            document.getElementById('submissionStatusIcon').innerHTML = '<i class="fas fa-circle"></i>';
            document.getElementById('submissionInput').value = '';
            updateStatus();
        }

        function updateStatus() {
            const statusCard = document.getElementById('statusCard');
            const statusDot = document.getElementById('statusDot');
            const statusText = document.getElementById('statusText');
            const statusSubtext = document.getElementById('statusSubtext');

            if (rubricUploaded && submissionUploaded) {
                statusCard.className = 'status-card ready';
                statusDot.className = 'status-dot ready';
                statusText.textContent = 'Ready to evaluate';
                statusSubtext.textContent = 'Type "evaluate" or click a quick action';
            } else if (rubricUploaded) {
                statusCard.className = 'status-card';
                statusDot.className = 'status-dot';
                statusText.textContent = 'Rubric loaded';
                statusSubtext.textContent = 'Upload submission to continue';
            } else if (submissionUploaded) {
                statusCard.className = 'status-card';
                statusDot.className = 'status-dot';
                statusText.textContent = 'Submission loaded';
                statusSubtext.textContent = 'Upload rubric to continue';
            } else {
                statusCard.className = 'status-card';
                statusDot.className = 'status-dot';
                statusText.textContent = 'Upload files to begin';
                statusSubtext.textContent = 'Rubric + Submission required';
            }
        }

        // ============ Chat ============
        async function sendMessage(text = null) {
            const input = document.getElementById('messageInput');
            const message = text || input.value.trim();

            if (!message || isLoading) return;

            // Hide welcome screen, show messages
            document.getElementById('welcomeScreen').style.display = 'none';
            document.getElementById('messagesContainer').style.display = 'block';

            // Add user message
            addMessage('user', message);
            input.value = '';
            autoResize(input);
            updateCharCount(input);

            // Show loading
            isLoading = true;
            document.getElementById('sendBtn').disabled = true;
            const loadingId = addLoading();

            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        session_id: sessionId,
                        message: message
                    })
                });

                const data = await response.json();
                removeLoading(loadingId);

                if (response.ok) {
                    addMessage('assistant', data.response);

                    // Update score display if evaluation done
                    if (data.evaluation_done && data.total_marks !== null) {
                        showScore(data.total_marks, data.max_marks);
                    }
                } else {
                    addMessage('assistant', `Error: ${data.detail || 'Something went wrong'}`);
                }
            } catch (error) {
                removeLoading(loadingId);
                addMessage('assistant', `Error: ${error.message}`);
            }

            isLoading = false;
            document.getElementById('sendBtn').disabled = false;
        }

        function addMessage(role, content) {
            const container = document.getElementById('messagesContainer');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message';

            const now = new Date();
            const time = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

            let avatarIcon, avatarClass, senderName;
            if (role === 'user') {
                avatarIcon = 'fa-user';
                avatarClass = 'user';
                senderName = 'You';
            } else if (role === 'assistant') {
                avatarIcon = 'fa-robot';
                avatarClass = 'assistant';
                senderName = 'GraderAI';
            } else {
                avatarIcon = 'fa-info-circle';
                avatarClass = 'system';
                senderName = 'System';
            }

            // Parse markdown for assistant messages
            let htmlContent;
            if (role === 'assistant') {
                htmlContent = marked.parse(content);
                // Add copy buttons to code blocks
                htmlContent = htmlContent.replace(/<pre><code/g, '<pre><div class="code-header"><span>Code</span><button class="copy-btn" onclick="copyCode(this)"><i class="fas fa-copy"></i> Copy</button></div><code');
            } else {
                htmlContent = escapeHtml(content);
            }

            messageDiv.innerHTML = `
                <div class="message-avatar ${avatarClass}">
                    <i class="fas ${avatarIcon}"></i>
                </div>
                <div class="message-body">
                    <div class="message-header">
                        <span class="message-sender">${senderName}</span>
                        <span class="message-time">${time}</span>
                    </div>
                    <div class="message-content">${htmlContent}</div>
                    ${role === 'assistant' ? `
                    <div class="message-actions">
                        <button class="action-btn" onclick="copyMessage(this)">
                            <i class="fas fa-copy"></i> Copy
                        </button>
                        <button class="action-btn" onclick="regenerate()">
                            <i class="fas fa-redo"></i> Regenerate
                        </button>
                    </div>
                    ` : ''}
                </div>
            `;

            container.appendChild(messageDiv);
            container.parentElement.scrollTop = container.parentElement.scrollHeight;

            // Highlight code blocks
            Prism.highlightAllUnder(messageDiv);
        }

        function addLoading() {
            const container = document.getElementById('messagesContainer');
            const id = 'loading-' + Date.now();
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message';
            messageDiv.id = id;

            messageDiv.innerHTML = `
                <div class="message-avatar assistant">
                    <i class="fas fa-robot"></i>
                </div>
                <div class="message-body">
                    <div class="message-header">
                        <span class="message-sender">GraderAI</span>
                    </div>
                    <div class="loading-indicator">
                        <div class="loading-spinner"></div>
                        <span class="loading-text">Analyzing submission<span class="loading-dots"><span></span><span></span><span></span></span></span>
                    </div>
                </div>
            `;

            container.appendChild(messageDiv);
            container.parentElement.scrollTop = container.parentElement.scrollHeight;
            return id;
        }

        function removeLoading(id) {
            const element = document.getElementById(id);
            if (element) element.remove();
        }

        function showScore(total, max) {
            const percentage = Math.round((total / max) * 100);
            const scoreClass = percentage >= 80 ? 'excellent' : percentage >= 60 ? 'good' : percentage >= 40 ? 'average' : 'poor';

            document.getElementById('statusCard').className = 'status-card evaluated';
            document.getElementById('statusDot').className = 'status-dot evaluated';
            document.getElementById('statusText').textContent = 'Evaluation Complete';
            document.getElementById('statusSubtext').textContent = 'Ask follow-up questions';

            const scoreDisplay = document.getElementById('scoreDisplay');
            scoreDisplay.style.display = 'block';
            document.getElementById('scoreValue').textContent = `${total}/${max}`;
            document.getElementById('scoreValue').className = `score-value ${scoreClass}`;
            document.getElementById('scoreBarFill').style.width = `${percentage}%`;
            document.getElementById('scoreBarFill').className = `score-bar-fill ${scoreClass}`;
        }

        // ============ Utilities ============
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        function handleKeyDown(event) {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                sendMessage();
            }
        }

        function autoResize(textarea) {
            textarea.style.height = 'auto';
            textarea.style.height = Math.min(textarea.scrollHeight, 150) + 'px';
        }

        function updateCharCount(textarea) {
            const count = textarea.value.length;
            document.getElementById('charCount').textContent = count > 0 ? `${count} chars` : '';
        }

        function setupKeyboardShortcuts() {
            document.addEventListener('keydown', (e) => {
                // Ctrl+N - New session
                if (e.ctrlKey && e.key === 'n') {
                    e.preventDefault();
                    newChat();
                }
            });
        }

        async function newChat() {
            // Reset UI — clear chat and submission, but keep rubric
            document.getElementById('welcomeScreen').style.display = 'flex';
            document.getElementById('messagesContainer').style.display = 'none';
            document.getElementById('messagesContainer').innerHTML = '';

            // Only reset submission, keep rubric
            removeSubmission({ stopPropagation: () => {} });

            // Reset score display
            document.getElementById('scoreDisplay').style.display = 'none';

            // Reset session on backend — keeps rubric
            if (sessionId) {
                try {
                    const response = await fetch(`/api/session/${sessionId}/reset`, { method: 'POST' });
                    const data = await response.json();
                    if (data.rubric_preserved) {
                        showToast('success', 'New Evaluation', `Ready for next submission — rubric "${data.rubric_name}" is still loaded`);
                    } else {
                        showToast('success', 'New Evaluation', 'Ready for a new evaluation');
                    }
                } catch (error) {
                    // Fallback: create entirely new session
                    removeRubric({ stopPropagation: () => {} });
                    await createSession();
                    showToast('success', 'New Session', 'Ready for a new evaluation');
                }
            } else {
                await createSession();
                showToast('success', 'New Session', 'Ready for a new evaluation');
            }
        }

        function copyMessage(button) {
            const content = button.closest('.message-body').querySelector('.message-content').innerText;
            navigator.clipboard.writeText(content);
            button.innerHTML = '<i class="fas fa-check"></i> Copied';
            setTimeout(() => {
                button.innerHTML = '<i class="fas fa-copy"></i> Copy';
            }, 2000);
        }

        function copyCode(button) {
            const code = button.closest('pre').querySelector('code').innerText;
            navigator.clipboard.writeText(code);
            button.classList.add('copied');
            button.innerHTML = '<i class="fas fa-check"></i> Copied';
            setTimeout(() => {
                button.classList.remove('copied');
                button.innerHTML = '<i class="fas fa-copy"></i> Copy';
            }, 2000);
        }

        function regenerate() {
            showToast('info', 'Coming Soon', 'Regeneration feature is in development');
        }

        // ============ Toast Notifications ============
        function showToast(type, title, message) {
            const container = document.getElementById('toastContainer');
            const toast = document.createElement('div');
            toast.className = `toast ${type}`;

            const icon = type === 'success' ? 'fa-check-circle' : type === 'error' ? 'fa-exclamation-circle' : 'fa-info-circle';

            toast.innerHTML = `
                <i class="fas ${icon} toast-icon"></i>
                <div class="toast-content">
                    <div class="toast-title">${title}</div>
                    <div class="toast-message">${message}</div>
                </div>
            `;

            container.appendChild(toast);

            setTimeout(() => {
                toast.style.opacity = '0';
                toast.style.transform = 'translateX(100px)';
                setTimeout(() => toast.remove(), 300);
            }, 4000);
        }

        function toggleTheme() {
            showToast('info', 'Coming Soon', 'Theme toggle is in development');
        }
    </script>
</body>
</html>
"""

# =============================================================================
# Run with: uvicorn app:app --reload --port 8000
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
