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
from langchain_anthropic import ChatAnthropic

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
    """Initialize the LLM."""
    config = EvaluatorConfig()
    if not config.azure_endpoint or not config.api_key:
        return None
    return ChatAnthropic(
        model=config.model,
        base_url=config.azure_endpoint,
        api_key=config.api_key,
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
            response = "Error: LLM not configured. Please check Azure Foundry credentials."
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
    <title>Assignment Evaluator</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: #343541;
            color: #ececf1;
            height: 100vh;
            display: flex;
        }

        /* Sidebar */
        .sidebar {
            width: 260px;
            background-color: #202123;
            padding: 12px;
            display: flex;
            flex-direction: column;
            border-right: 1px solid #4d4d4f;
        }

        .new-chat-btn {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 12px;
            border: 1px solid #4d4d4f;
            border-radius: 8px;
            background: transparent;
            color: #ececf1;
            cursor: pointer;
            font-size: 14px;
            transition: background 0.2s;
        }

        .new-chat-btn:hover {
            background: #2a2b32;
        }

        .sidebar-title {
            font-size: 12px;
            color: #8e8ea0;
            padding: 12px 8px;
            text-transform: uppercase;
        }

        .upload-section {
            margin-top: 20px;
            padding: 12px;
            background: #2a2b32;
            border-radius: 8px;
        }

        .upload-section h3 {
            font-size: 13px;
            margin-bottom: 12px;
            color: #ececf1;
        }

        .upload-btn {
            display: flex;
            align-items: center;
            gap: 8px;
            width: 100%;
            padding: 10px 12px;
            margin-bottom: 8px;
            border: 1px dashed #4d4d4f;
            border-radius: 6px;
            background: transparent;
            color: #ececf1;
            cursor: pointer;
            font-size: 13px;
            transition: all 0.2s;
        }

        .upload-btn:hover {
            border-color: #8e8ea0;
            background: #343541;
        }

        .upload-btn.uploaded {
            border-color: #10a37f;
            background: rgba(16, 163, 127, 0.1);
        }

        .upload-btn i {
            font-size: 14px;
        }

        .file-name {
            font-size: 11px;
            color: #10a37f;
            margin-top: 4px;
            word-break: break-all;
        }

        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 10px;
            background: #2a2b32;
            border-radius: 20px;
            font-size: 12px;
            margin-top: 12px;
        }

        .status-badge.ready {
            background: rgba(16, 163, 127, 0.2);
            color: #10a37f;
        }

        /* Main Content */
        .main-content {
            flex: 1;
            display: flex;
            flex-direction: column;
            max-width: 900px;
            margin: 0 auto;
            width: 100%;
        }

        /* Header */
        .header {
            padding: 16px 24px;
            border-bottom: 1px solid #4d4d4f;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .header h1 {
            font-size: 18px;
            font-weight: 600;
        }

        .model-badge {
            font-size: 12px;
            padding: 4px 12px;
            background: #2a2b32;
            border-radius: 20px;
            color: #8e8ea0;
        }

        /* Chat Container */
        .chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 24px;
        }

        .welcome-message {
            text-align: center;
            padding: 60px 20px;
        }

        .welcome-message h2 {
            font-size: 28px;
            margin-bottom: 16px;
        }

        .welcome-message p {
            color: #8e8ea0;
            max-width: 500px;
            margin: 0 auto 24px;
            line-height: 1.6;
        }

        .quick-actions {
            display: flex;
            gap: 12px;
            justify-content: center;
            flex-wrap: wrap;
        }

        .quick-action {
            padding: 12px 20px;
            background: #2a2b32;
            border: 1px solid #4d4d4f;
            border-radius: 8px;
            color: #ececf1;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.2s;
        }

        .quick-action:hover {
            background: #343541;
            border-color: #8e8ea0;
        }

        /* Messages */
        .message {
            display: flex;
            gap: 16px;
            padding: 24px 0;
            border-bottom: 1px solid #4d4d4f;
        }

        .message:last-child {
            border-bottom: none;
        }

        .message-avatar {
            width: 36px;
            height: 36px;
            border-radius: 4px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 14px;
            flex-shrink: 0;
        }

        .message-avatar.user {
            background: #5436da;
        }

        .message-avatar.assistant {
            background: #10a37f;
        }

        .message-content {
            flex: 1;
            line-height: 1.7;
            overflow-wrap: break-word;
        }

        .message-content h1, .message-content h2, .message-content h3 {
            margin: 16px 0 8px;
        }

        .message-content h1 { font-size: 1.5em; }
        .message-content h2 { font-size: 1.3em; }
        .message-content h3 { font-size: 1.1em; }

        .message-content p {
            margin: 8px 0;
        }

        .message-content ul, .message-content ol {
            margin: 8px 0 8px 24px;
        }

        .message-content li {
            margin: 4px 0;
        }

        .message-content strong {
            color: #fff;
        }

        .message-content code {
            background: #2a2b32;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.9em;
        }

        .message-content pre {
            background: #2a2b32;
            padding: 16px;
            border-radius: 8px;
            overflow-x: auto;
            margin: 12px 0;
        }

        .message-content hr {
            border: none;
            border-top: 1px solid #4d4d4f;
            margin: 16px 0;
        }

        /* Input Area */
        .input-area {
            padding: 16px 24px 24px;
            background: #343541;
        }

        .input-container {
            display: flex;
            gap: 12px;
            background: #40414f;
            border-radius: 12px;
            padding: 12px 16px;
            border: 1px solid #4d4d4f;
        }

        .input-container:focus-within {
            border-color: #8e8ea0;
        }

        .input-container textarea {
            flex: 1;
            background: transparent;
            border: none;
            color: #ececf1;
            font-size: 15px;
            font-family: inherit;
            resize: none;
            outline: none;
            max-height: 200px;
        }

        .input-container textarea::placeholder {
            color: #8e8ea0;
        }

        .send-btn {
            width: 36px;
            height: 36px;
            border-radius: 8px;
            background: #10a37f;
            border: none;
            color: white;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: background 0.2s;
            align-self: flex-end;
        }

        .send-btn:hover {
            background: #0d8a6a;
        }

        .send-btn:disabled {
            background: #4d4d4f;
            cursor: not-allowed;
        }

        .input-hint {
            text-align: center;
            font-size: 12px;
            color: #8e8ea0;
            margin-top: 8px;
        }

        /* Loading */
        .loading {
            display: flex;
            gap: 4px;
            padding: 8px 0;
        }

        .loading span {
            width: 8px;
            height: 8px;
            background: #8e8ea0;
            border-radius: 50%;
            animation: bounce 1.4s infinite ease-in-out;
        }

        .loading span:nth-child(1) { animation-delay: -0.32s; }
        .loading span:nth-child(2) { animation-delay: -0.16s; }

        @keyframes bounce {
            0%, 80%, 100% { transform: scale(0); }
            40% { transform: scale(1); }
        }

        /* Score Card */
        .score-card {
            background: linear-gradient(135deg, #10a37f, #0d8a6a);
            border-radius: 12px;
            padding: 20px;
            margin: 16px 0;
            text-align: center;
        }

        .score-card .score {
            font-size: 36px;
            font-weight: 700;
        }

        .score-card .percentage {
            font-size: 18px;
            opacity: 0.9;
        }

        /* Hidden file inputs */
        input[type="file"] {
            display: none;
        }

        /* Responsive */
        @media (max-width: 768px) {
            .sidebar {
                display: none;
            }
        }
    </style>
</head>
<body>
    <!-- Sidebar -->
    <aside class="sidebar">
        <button class="new-chat-btn" onclick="newChat()">
            <i class="fas fa-plus"></i>
            New Chat
        </button>

        <div class="upload-section">
            <h3><i class="fas fa-file-alt"></i> Files</h3>

            <input type="file" id="rubricInput" accept=".json,.pdf" onchange="uploadRubric(this)">
            <button class="upload-btn" id="rubricBtn" onclick="document.getElementById('rubricInput').click()">
                <i class="fas fa-file-code"></i>
                Upload Rubric (JSON/PDF)
            </button>
            <div class="file-name" id="rubricName"></div>

            <input type="file" id="submissionInput" accept=".html,.pdf" onchange="uploadSubmission(this)">
            <button class="upload-btn" id="submissionBtn" onclick="document.getElementById('submissionInput').click()">
                <i class="fas fa-file-upload"></i>
                Upload Submission
            </button>
            <div class="file-name" id="submissionName"></div>

            <div class="status-badge" id="statusBadge">
                <i class="fas fa-circle"></i>
                <span>Upload files to start</span>
            </div>
        </div>
    </aside>

    <!-- Main Content -->
    <main class="main-content">
        <header class="header">
            <h1><i class="fas fa-graduation-cap"></i> Assignment Evaluator</h1>
            <span class="model-badge">Claude via Azure Foundry</span>
        </header>

        <div class="chat-container" id="chatContainer">
            <div class="welcome-message" id="welcomeMessage">
                <h2>Assignment Evaluator</h2>
                <p>Upload a rubric and student submission, then ask me to evaluate. I'll provide detailed feedback with marks for each section.</p>
                <div class="quick-actions">
                    <button class="quick-action" onclick="sendMessage('Evaluate this submission')">
                        <i class="fas fa-check-circle"></i> Evaluate Submission
                    </button>
                    <button class="quick-action" onclick="sendMessage('What are the main strengths?')">
                        <i class="fas fa-star"></i> Show Strengths
                    </button>
                    <button class="quick-action" onclick="sendMessage('What improvements are needed?')">
                        <i class="fas fa-lightbulb"></i> Improvements
                    </button>
                </div>
            </div>
        </div>

        <div class="input-area">
            <div class="input-container">
                <textarea
                    id="messageInput"
                    placeholder="Ask me to evaluate the submission or ask follow-up questions..."
                    rows="1"
                    onkeydown="handleKeyDown(event)"
                    oninput="autoResize(this)"
                ></textarea>
                <button class="send-btn" id="sendBtn" onclick="sendMessage()">
                    <i class="fas fa-paper-plane"></i>
                </button>
            </div>
            <p class="input-hint">Press Enter to send, Shift+Enter for new line</p>
        </div>
    </main>

    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <script>
        let sessionId = null;
        let isLoading = false;

        // Initialize session on page load
        window.onload = async function() {
            await createSession();
        };

        async function createSession() {
            try {
                const response = await fetch('/api/session', { method: 'POST' });
                const data = await response.json();
                sessionId = data.session_id;
                console.log('Session created:', sessionId);
            } catch (error) {
                console.error('Failed to create session:', error);
            }
        }

        async function uploadRubric(input) {
            if (!input.files.length) return;

            const file = input.files[0];
            const isPdf = file.name.toLowerCase().endsWith('.pdf');
            const formData = new FormData();
            formData.append('file', file);
            formData.append('session_id', sessionId);

            // Show loading for PDF conversion
            const rubricBtn = document.getElementById('rubricBtn');
            const originalText = rubricBtn.innerHTML;
            if (isPdf) {
                rubricBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Converting PDF...';
                rubricBtn.disabled = true;
            }

            try {
                const response = await fetch('/api/upload/rubric', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (response.ok) {
                    rubricBtn.classList.add('uploaded');
                    rubricBtn.innerHTML = '<i class="fas fa-check"></i> Rubric Loaded';
                    document.getElementById('rubricName').textContent = file.name;
                    updateStatus();
                    const msg = isPdf ? `PDF rubric converted and loaded: ${file.name}` : `Rubric uploaded: ${file.name}`;
                    addMessage('system', msg);
                } else {
                    rubricBtn.innerHTML = originalText;
                    alert(data.detail || 'Upload failed');
                }
            } catch (error) {
                rubricBtn.innerHTML = originalText;
                alert('Upload failed: ' + error.message);
            } finally {
                rubricBtn.disabled = false;
            }
        }

        async function uploadSubmission(input) {
            if (!input.files.length) return;

            const file = input.files[0];
            const formData = new FormData();
            formData.append('file', file);
            formData.append('session_id', sessionId);

            try {
                const response = await fetch('/api/upload/submission', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (response.ok) {
                    document.getElementById('submissionBtn').classList.add('uploaded');
                    document.getElementById('submissionName').textContent = file.name;
                    updateStatus();
                    addMessage('system', `Submission uploaded: ${file.name}`);
                } else {
                    alert(data.detail || 'Upload failed');
                }
            } catch (error) {
                alert('Upload failed: ' + error.message);
            }
        }

        function updateStatus() {
            const rubricUploaded = document.getElementById('rubricBtn').classList.contains('uploaded');
            const submissionUploaded = document.getElementById('submissionBtn').classList.contains('uploaded');
            const badge = document.getElementById('statusBadge');

            if (rubricUploaded && submissionUploaded) {
                badge.className = 'status-badge ready';
                badge.innerHTML = '<i class="fas fa-check-circle"></i><span>Ready to evaluate</span>';
            } else if (rubricUploaded || submissionUploaded) {
                badge.innerHTML = '<i class="fas fa-circle"></i><span>Upload both files</span>';
            }
        }

        async function sendMessage(text = null) {
            const input = document.getElementById('messageInput');
            const message = text || input.value.trim();

            if (!message || isLoading) return;

            // Hide welcome message
            document.getElementById('welcomeMessage').style.display = 'none';

            // Add user message
            addMessage('user', message);
            input.value = '';
            autoResize(input);

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

                // Remove loading
                removeLoading(loadingId);

                if (response.ok) {
                    addMessage('assistant', data.response);

                    // Update status if evaluation done
                    if (data.evaluation_done) {
                        const badge = document.getElementById('statusBadge');
                        badge.className = 'status-badge ready';
                        badge.innerHTML = `<i class="fas fa-check-circle"></i><span>Score: ${data.total_marks}/${data.max_marks}</span>`;
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
            const container = document.getElementById('chatContainer');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message';

            const avatarIcon = role === 'user' ? 'fa-user' : (role === 'assistant' ? 'fa-robot' : 'fa-info-circle');
            const avatarClass = role === 'user' ? 'user' : 'assistant';

            // Parse markdown for assistant messages
            const htmlContent = role === 'assistant' ? marked.parse(content) : escapeHtml(content);

            messageDiv.innerHTML = `
                <div class="message-avatar ${avatarClass}">
                    <i class="fas ${avatarIcon}"></i>
                </div>
                <div class="message-content">${htmlContent}</div>
            `;

            container.appendChild(messageDiv);
            container.scrollTop = container.scrollHeight;
        }

        function addLoading() {
            const container = document.getElementById('chatContainer');
            const id = 'loading-' + Date.now();
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message';
            messageDiv.id = id;

            messageDiv.innerHTML = `
                <div class="message-avatar assistant">
                    <i class="fas fa-robot"></i>
                </div>
                <div class="message-content">
                    <div class="loading">
                        <span></span>
                        <span></span>
                        <span></span>
                    </div>
                </div>
            `;

            container.appendChild(messageDiv);
            container.scrollTop = container.scrollHeight;
            return id;
        }

        function removeLoading(id) {
            const element = document.getElementById(id);
            if (element) element.remove();
        }

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
            textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
        }

        async function newChat() {
            // Reset UI
            document.getElementById('chatContainer').innerHTML = `
                <div class="welcome-message" id="welcomeMessage">
                    <h2>Assignment Evaluator</h2>
                    <p>Upload a rubric and student submission, then ask me to evaluate. I'll provide detailed feedback with marks for each section.</p>
                    <div class="quick-actions">
                        <button class="quick-action" onclick="sendMessage('Evaluate this submission')">
                            <i class="fas fa-check-circle"></i> Evaluate Submission
                        </button>
                        <button class="quick-action" onclick="sendMessage('What are the main strengths?')">
                            <i class="fas fa-star"></i> Show Strengths
                        </button>
                        <button class="quick-action" onclick="sendMessage('What improvements are needed?')">
                            <i class="fas fa-lightbulb"></i> Improvements
                        </button>
                    </div>
                </div>
            `;

            document.getElementById('rubricBtn').classList.remove('uploaded');
            document.getElementById('submissionBtn').classList.remove('uploaded');
            document.getElementById('rubricName').textContent = '';
            document.getElementById('submissionName').textContent = '';
            document.getElementById('statusBadge').className = 'status-badge';
            document.getElementById('statusBadge').innerHTML = '<i class="fas fa-circle"></i><span>Upload files to start</span>';

            // Create new session
            await createSession();
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
