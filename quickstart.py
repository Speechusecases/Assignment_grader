#!/usr/bin/env python3
"""
Quick Start Script for LangGraph Evaluator Agent

This script provides simple commands to:
1. Check Azure Foundry configuration
2. Test the evaluator with sample files
3. Batch process multiple submissions
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def check_azure_foundry():
    """Check if Azure Foundry is configured."""
    print("🔍 Checking Azure Foundry configuration...")

    endpoint = os.environ.get("AZURE_FOUNDRY_ENDPOINT")
    api_key = os.environ.get("AZURE_FOUNDRY_API_KEY")

    if endpoint and api_key:
        print("✅ Azure Foundry credentials are configured!")
        print(f"   Endpoint: {endpoint[:50]}..." if len(endpoint) > 50 else f"   Endpoint: {endpoint}")
        print("   API Key: ****" + api_key[-4:] if len(api_key) > 4 else "   API Key: ****")
        return True
    else:
        print("❌ Azure Foundry credentials not fully configured.")
        if not endpoint:
            print("   Missing: AZURE_FOUNDRY_ENDPOINT")
        if not api_key:
            print("   Missing: AZURE_FOUNDRY_API_KEY")
        print("\n   Set environment variables:")
        print("     export AZURE_FOUNDRY_ENDPOINT='https://your-endpoint.azure.com'")
        print("     export AZURE_FOUNDRY_API_KEY='your-api-key'")
        return False


def list_models():
    """List available Claude models."""
    print("\n📋 Available Claude Models (via Azure Foundry):")
    models = [
        ("claude-3-5-sonnet-20241022", "Balanced performance (recommended)"),
        ("claude-3-opus-20240229", "Most capable, best for complex tasks"),
        ("claude-3-sonnet-20240229", "Fast and capable"),
        ("claude-3-haiku-20240307", "Fastest, most economical")
    ]
    for model, description in models:
        print(f"   - {model}: {description}")


def test_evaluation(file_path: str, model: str = "claude-3-5-sonnet-20241022"):
    """Run a test evaluation."""
    print(f"\n🧪 Testing evaluation with {model}...")

    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        return False

    # Check Azure Foundry config first
    if not check_azure_foundry():
        print("\n⚠️  Please configure Azure Foundry credentials first.")
        return False

    try:
        from langraph_evaluator_agent import run_evaluation

        results = run_evaluation(
            file_path=file_path,
            model=model,
            temperature=0.3
        )

        print(f"\n✅ Test completed!")
        print(f"   Score: {results['total_marks']}/{results['max_marks']}")
        return True

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def batch_evaluate(directory: str, model: str = "claude-3-5-sonnet-20241022"):
    """Evaluate all submissions in a directory."""
    print(f"\n📂 Batch processing: {directory}")

    dir_path = Path(directory)
    if not dir_path.exists():
        print(f"❌ Directory not found: {directory}")
        return

    # Find all HTML and PDF files
    files = list(dir_path.glob("*.html")) + list(dir_path.glob("*.pdf"))

    if not files:
        print("   No HTML or PDF files found")
        return

    print(f"   Found {len(files)} submission(s)")

    from langraph_evaluator_agent import run_evaluation

    for i, file_path in enumerate(files, 1):
        print(f"\n{'='*60}")
        print(f"Processing {i}/{len(files)}: {file_path.name}")
        print('='*60)

        try:
            results = run_evaluation(
                file_path=str(file_path),
                model=model,
                temperature=0.3
            )

            print(f"\n✅ {file_path.name}: {results['total_marks']}/{results['max_marks']}")

        except Exception as e:
            print(f"\n❌ {file_path.name}: Failed - {e}")

    print(f"\n{'='*60}")
    print("Batch processing complete!")
    print('='*60)


def setup():
    """Run setup - install dependencies and configure Azure Foundry."""
    print("🚀 Setting up LangGraph Evaluator Agent...")

    # Check Python version
    if sys.version_info < (3, 9):
        print("❌ Python 3.9+ required")
        return False

    # Install requirements
    print("\n📦 Installing requirements...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            check=True
        )
        print("✅ Requirements installed")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

    # Check Azure Foundry configuration
    print("\n🔐 Checking Azure Foundry configuration...")
    if not check_azure_foundry():
        print("\n⚠️  Please configure Azure Foundry credentials:")
        print("   export AZURE_FOUNDRY_ENDPOINT='https://your-endpoint.azure.com'")
        print("   export AZURE_FOUNDRY_API_KEY='your-api-key'")

    # List available models
    list_models()

    print("\n✅ Setup complete!")
    print("\nNext steps:")
    print("   1. Configure Azure Foundry credentials (if not done)")
    print("   2. Place your submission files in this directory")
    print("   3. Run: python quickstart.py test <file.html or file.pdf>")
    print("   4. Or use directly: python langraph_evaluator_agent.py <file>")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Quick start for LangGraph Evaluator Agent (Azure Foundry + Claude)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Setup everything
    python quickstart.py setup

    # Check Azure Foundry configuration
    python quickstart.py check

    # List available Claude models
    python quickstart.py models

    # Test with a file
    python quickstart.py test submission.html

    # Test with specific Claude model
    python quickstart.py test submission.html --model claude-3-opus-20240229

    # Batch evaluate directory
    python quickstart.py batch ./submissions
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Setup command
    subparsers.add_parser("setup", help="Install dependencies and setup")

    # Check command
    subparsers.add_parser("check", help="Check Azure Foundry configuration")

    # Models command
    subparsers.add_parser("models", help="List available Claude models")

    # Test command
    test_parser = subparsers.add_parser("test", help="Test evaluation with a file")
    test_parser.add_argument("file", help="Path to HTML or PDF file")
    test_parser.add_argument("--model", default="claude-3-5-sonnet-20241022", help="Claude model to use")

    # Batch command
    batch_parser = subparsers.add_parser("batch", help="Batch evaluate directory")
    batch_parser.add_argument("directory", help="Directory containing submissions")
    batch_parser.add_argument("--model", default="claude-3-5-sonnet-20241022", help="Claude model to use")

    args = parser.parse_args()

    if args.command == "setup":
        setup()
    elif args.command == "check":
        check_azure_foundry()
    elif args.command == "models":
        list_models()
    elif args.command == "test":
        test_evaluation(args.file, args.model)
    elif args.command == "batch":
        if check_azure_foundry():
            batch_evaluate(args.directory, args.model)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
