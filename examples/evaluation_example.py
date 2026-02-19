"""
Example: RAG System Evaluation

This example demonstrates:
1. Creating test cases with ground truth
2. Running evaluations
3. Analyzing results
4. Identifying areas for improvement
"""

import os
import sys
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_chatbot.vector_store import VectorStore, Document
from rag_chatbot.rag_pipeline import RAGPipeline
from rag_chatbot.evaluator import RAGEvaluator, evaluate_batch, summarize_evaluations


def main():
    """Run evaluation example."""
    
    print("=" * 80)
    print("RAG System Evaluation Example")
    print("=" * 80)
    print()
    
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  ERROR: OPENAI_API_KEY not set!")
        print("   Set it in .env file or environment variable to run evaluations")
        print("   Example: export OPENAI_API_KEY='your-key-here'")
        return
    
    # Step 1: Setup RAG system with test data
    print("1. Setting up RAG system with test documents...")
    vector_store = VectorStore(
        collection_name="eval_test",
        persist_directory="./chroma_db_eval"
    )
    vector_store.clear()
    
    # Add test documents
    documents = [
        Document(
            content="Python was created by Guido van Rossum and first released in 1991. "
                   "It emphasizes code readability and uses significant indentation.",
            metadata={"topic": "python", "type": "history"},
            doc_id="doc_python_1"
        ),
        Document(
            content="Python supports multiple programming paradigms including procedural, "
                   "object-oriented, and functional programming. It has a comprehensive "
                   "standard library.",
            metadata={"topic": "python", "type": "features"},
            doc_id="doc_python_2"
        ),
        Document(
            content="JavaScript was created by Brendan Eich in 1995. It is primarily used "
                   "for web development and runs in web browsers.",
            metadata={"topic": "javascript", "type": "history"},
            doc_id="doc_js_1"
        )
    ]
    vector_store.add_documents(documents)
    print(f"   Added {len(documents)} test documents")
    print()
    
    # Step 2: Generate answers for test queries
    print("2. Generating answers for test queries...")
    rag_pipeline = RAGPipeline(vector_store=vector_store, temperature=0.1)
    
    test_queries = [
        {
            "query": "Who created Python?",
            "ground_truth": "Python was created by Guido van Rossum."
        },
        {
            "query": "When was Python first released?",
            "ground_truth": "Python was first released in 1991."
        },
        {
            "query": "What programming paradigms does Python support?",
            "ground_truth": "Python supports procedural, object-oriented, and functional programming paradigms."
        }
    ]
    
    # Generate answers
    test_cases = []
    for test_query in test_queries:
        response = rag_pipeline.query(test_query["query"], top_k=3)
        test_cases.append({
            "query": test_query["query"],
            "answer": response.answer,
            "context": [s["content"] for s in response.sources],
            "ground_truth": test_query["ground_truth"]
        })
        print(f"   ✓ Generated answer for: {test_query['query']}")
    print()
    
    # Step 3: Run evaluations
    print("3. Running evaluations...")
    print("   This may take a minute as we evaluate multiple dimensions...")
    print()
    
    evaluator = RAGEvaluator(model="gpt-4")
    results = evaluate_batch(evaluator, test_cases)
    
    # Step 4: Display results
    print("4. Evaluation Results:")
    print()
    
    for i, result in enumerate(results, 1):
        print(f"\n{'═' * 80}")
        print(f"Test Case {i}")
        print('═' * 80)
        print(f"\nQuery: {result.query}")
        print(f"\nGenerated Answer:\n{result.answer}")
        print(f"\nGround Truth:\n{result.ground_truth}")
        
        print(f"\n📊 Metrics:")
        print(f"   • Faithfulness (Groundedness): {result.metrics.faithfulness:.2f}")
        print(f"   • Answer Relevance:            {result.metrics.answer_relevance:.2f}")
        print(f"   • Context Relevance:           {result.metrics.context_relevance:.2f}")
        if result.metrics.factual_correctness:
            print(f"   • Factual Correctness:         {result.metrics.factual_correctness:.2f}")
        print(f"   • Overall Score:               {result.metrics.overall_score:.2f}")
        
        print(f"\n💡 Detailed Feedback:")
        for metric_name, feedback in result.detailed_feedback.items():
            if feedback and feedback != "No ground truth provided":
                print(f"\n   {metric_name.replace('_', ' ').title()}:")
                # Print first 150 chars of feedback
                feedback_preview = feedback[:150] + "..." if len(feedback) > 150 else feedback
                print(f"   {feedback_preview}")
    
    # Step 5: Summary statistics
    print(f"\n\n{'═' * 80}")
    print("Summary Statistics")
    print('═' * 80)
    
    summary = summarize_evaluations(results)
    
    print(f"\nTotal Test Cases: {summary['total_cases']}")
    print(f"\nAverage Scores:")
    print(f"   • Faithfulness:        {summary['faithfulness']['mean']:.2f}")
    print(f"   • Answer Relevance:    {summary['answer_relevance']['mean']:.2f}")
    print(f"   • Context Relevance:   {summary['context_relevance']['mean']:.2f}")
    if 'factual_correctness' in summary:
        print(f"   • Factual Correctness: {summary['factual_correctness']['mean']:.2f}")
    print(f"   • Overall Score:       {summary['overall_score']['mean']:.2f}")
    
    # Step 6: Recommendations
    print(f"\n\n{'═' * 80}")
    print("Recommendations for Improvement")
    print('═' * 80)
    
    avg_faithfulness = summary['faithfulness']['mean']
    avg_relevance = summary['answer_relevance']['mean']
    avg_context = summary['context_relevance']['mean']
    
    recommendations = []
    
    if avg_faithfulness < 0.7:
        recommendations.append(
            "📌 Low Faithfulness: Answers may not be well-grounded in context.\n"
            "   → Strengthen prompts to require explicit citations\n"
            "   → Implement stricter answer verification\n"
            "   → Consider lower temperature settings"
        )
    
    if avg_relevance < 0.7:
        recommendations.append(
            "📌 Low Answer Relevance: Answers may not directly address questions.\n"
            "   → Improve question understanding\n"
            "   → Add query refinement step\n"
            "   → Use few-shot examples in prompts"
        )
    
    if avg_context < 0.7:
        recommendations.append(
            "📌 Low Context Relevance: Retrieved documents may not be optimal.\n"
            "   → Improve embedding model or chunking strategy\n"
            "   → Add query expansion or rewriting\n"
            "   → Consider hybrid search (semantic + keyword)"
        )
    
    if recommendations:
        print("\n" + "\n\n".join(recommendations))
    else:
        print("\n✅ All metrics are above 0.7 threshold - system performing well!")
    
    print("\n" + "=" * 80)
    print()


if __name__ == "__main__":
    main()
