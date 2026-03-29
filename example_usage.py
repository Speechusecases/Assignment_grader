#!/usr/bin/env python3
"""
Example Usage: LangGraph Evaluator Agent

This script demonstrates how to use the evaluator agent programmatically
in your own Python code.
"""

from langraph_evaluator_agent import run_evaluation


def example_single_evaluation():
    """Example: Evaluate a single submission."""

    # Evaluate an HTML notebook submission
    # Note: Ensure AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY are set
    results = run_evaluation(
        file_path="AnothercopyofAIML_ML_Project_Full_Code_Notebook.html",
        azure_deployment="gpt-4o",  # GPT model via Azure OpenAI
        temperature=0.3,                      # Lower = more deterministic
        extract_images=True                   # Extract and include images
    )

    # Access results
    print(f"\n📊 Score: {results['total_marks']}/{results['max_marks']}")
    print(f"📈 Percentage: {results['percentage']}%")

    # Access section-wise breakdown
    for section in results['evaluations'].get('sections', []):
        print(f"\n{section['name']}: {section['marks']}/{section['max_marks']}")

    return results


def example_batch_evaluation():
    """Example: Evaluate multiple submissions."""

    from pathlib import Path

    # Find all submissions
    submissions = [
        "AnothercopyofAIML_ML_Project_Full_Code_Notebook.html",
        "Assignment-Personal Loan Campaign.pdf"
    ]

    all_results = []

    for submission in submissions:
        if Path(submission).exists():
            print(f"\n{'='*60}")
            print(f"Evaluating: {submission}")
            print('='*60)

            results = run_evaluation(
                file_path=submission,
                azure_deployment="gpt-4o"
            )

            all_results.append({
                'file': submission,
                'score': results['total_marks'],
                'max': results['max_marks'],
                'percentage': results['percentage']
            })

    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print('='*60)
    for r in all_results:
        print(f"{r['file']}: {r['score']}/{r['max']} ({r['percentage']}%)")


def example_custom_model():
    """Example: Use different GPT models via Azure OpenAI."""

    file_path = "AnothercopyofAIML_ML_Project_Full_Code_Notebook.html"

    models = [
        "gpt-4o",
        "gpt-4",
        "gpt-35-turbo"
    ]

    for model in models:
        print(f"\n{'='*60}")
        print(f"Testing with model: {model}")
        print('='*60)

        try:
            results = run_evaluation(
                file_path=file_path,
                azure_deployment=model,
                temperature=0.3
            )
            print(f"✅ {model}: {results['total_marks']}/{results['max_marks']}")
        except Exception as e:
            print(f"❌ {model}: Failed - {e}")


def example_extract_feedback():
    """Example: Extract and use feedback programmatically."""

    results = run_evaluation(
        file_path="AnothercopyofAIML_ML_Project_Full_Code_Notebook.html",
        azure_deployment="gpt-4o"
    )

    # Get recommendations
    recommendations = results['evaluations'].get('recommendations', [])

    print("\n📝 Key Recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")

    # Get section feedback
    print("\n📋 Section Feedback:")
    for section in results['evaluations'].get('sections', []):
        print(f"\n{section['name']} ({section['marks']}/{section['max_marks']}):")
        # Print first 200 chars of feedback
        feedback = section['feedback'][:200] + "..."
        print(f"  {feedback}")


def main():
    """Run examples based on available files."""

    from pathlib import Path

    # Check which example files exist
    html_file = "AnothercopyofAIML_ML_Project_Full_Code_Notebook.html"
    pdf_file = "Assignment-Personal Loan Campaign.pdf"

    examples = []

    if Path(html_file).exists():
        examples.append(("Single HTML Evaluation", example_single_evaluation))

    if Path(html_file).exists() or Path(pdf_file).exists():
        examples.append(("Batch Evaluation", example_batch_evaluation))
        examples.append(("Extract Feedback", example_extract_feedback))

    examples.append(("Custom Model Testing", example_custom_model))

    print("="*60)
    print("LangGraph Evaluator Agent - Example Usage")
    print("="*60)

    for i, (name, func) in enumerate(examples, 1):
        print(f"\n[{i}] {name}")

    print("\n" + "="*60)

    # Run first available example
    if examples:
        name, func = examples[0]
        print(f"\nRunning: {name}\n")
        try:
            func()
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\nNo example files found in current directory.")
        print("Available examples:")
        print("  1. Place HTML/PDF submissions in this directory")
        print("  2. Run: python example_usage.py")


if __name__ == "__main__":
    main()
