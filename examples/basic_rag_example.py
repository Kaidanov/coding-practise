"""
Example: Basic RAG Chatbot Usage

This example demonstrates:
1. Setting up a vector store
2. Adding documents
3. Querying the RAG pipeline
4. Understanding response structure with anti-hallucination features
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_chatbot.vector_store import VectorStore, Document
from rag_chatbot.rag_pipeline import RAGPipeline


def main():
    """Run a basic RAG chatbot example."""
    
    print("=" * 80)
    print("RAG Chatbot Example: Insurance Policy Q&A")
    print("=" * 80)
    print()
    
    # Step 1: Initialize vector store
    print("1. Initializing vector store...")
    vector_store = VectorStore(
        collection_name="insurance_policies",
        persist_directory="./chroma_db_example"
    )
    
    # Clear any existing data
    vector_store.clear()
    
    # Step 2: Add sample documents (insurance policy information)
    print("2. Adding sample insurance policy documents...")
    documents = [
        Document(
            content=(
                "General Liability Insurance provides coverage for bodily injury "
                "and property damage claims. The policy covers up to $1 million per "
                "occurrence and $2 million aggregate. This includes legal defense costs."
            ),
            metadata={
                "policy_type": "general_liability",
                "section": "coverage_details",
                "page": 1
            },
            doc_id="policy_gl_001"
        ),
        Document(
            content=(
                "Professional Liability Insurance (Errors & Omissions) protects against "
                "claims of negligence or inadequate work. Coverage limits are $500,000 per "
                "claim with a $5,000 deductible. Claims must be reported within 60 days."
            ),
            metadata={
                "policy_type": "professional_liability",
                "section": "coverage_details",
                "page": 1
            },
            doc_id="policy_pl_001"
        ),
        Document(
            content=(
                "Claim Filing Process: All claims must be submitted within 30 days of the "
                "incident. Required documentation includes incident report, photographs if "
                "applicable, and witness statements. Claims are typically processed within "
                "15 business days."
            ),
            metadata={
                "policy_type": "general",
                "section": "claims_process",
                "page": 5
            },
            doc_id="policy_gen_001"
        ),
        Document(
            content=(
                "Exclusions: The following are NOT covered under General Liability: "
                "intentional acts, pollution, professional services, cyber incidents, "
                "and employee injuries (covered under Workers Compensation instead)."
            ),
            metadata={
                "policy_type": "general_liability",
                "section": "exclusions",
                "page": 3
            },
            doc_id="policy_gl_002"
        ),
        Document(
            content=(
                "Premium Calculation: Premiums are based on business size, industry risk "
                "rating, claims history, and coverage limits. Small businesses (under $500k "
                "revenue) qualify for a 15% discount. Annual premiums must be paid in full "
                "or in quarterly installments."
            ),
            metadata={
                "policy_type": "general",
                "section": "pricing",
                "page": 8
            },
            doc_id="policy_gen_002"
        )
    ]
    
    vector_store.add_documents(documents)
    print(f"   Added {vector_store.count()} documents to vector store")
    print()
    
    # Step 3: Initialize RAG pipeline
    print("3. Initializing RAG pipeline...")
    print("   Note: Requires OPENAI_API_KEY environment variable")
    print()
    
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  WARNING: OPENAI_API_KEY not set!")
        print("   Set it in .env file or environment variable to run queries")
        print("   Example: export OPENAI_API_KEY='your-key-here'")
        print()
        print("Skipping query examples...")
        return
    
    rag_pipeline = RAGPipeline(
        vector_store=vector_store,
        model="gpt-4",
        temperature=0.1,  # Low temperature for factual responses
        confidence_threshold=0.7
    )
    
    # Step 4: Ask questions
    print("4. Querying the RAG chatbot...")
    print()
    
    questions = [
        "What is the coverage limit for General Liability Insurance?",
        "How long do I have to file a claim?",
        "Does the policy cover cyber incidents?",
        "What discount is available for small businesses?",
        "What is the weather like today?"  # Out-of-context question
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'─' * 80}")
        print(f"Question {i}: {question}")
        print('─' * 80)
        
        response = rag_pipeline.query(question, top_k=3)
        
        print(f"\n📝 Answer:")
        print(f"   {response.answer}")
        print(f"\n📊 Metadata:")
        print(f"   • Confidence: {response.confidence:.2%}")
        print(f"   • Grounded: {'✅ Yes' if response.is_grounded else '❌ No'}")
        print(f"   • Sources Used: {len(response.sources)}")
        
        if response.sources:
            print(f"\n📚 Top Sources:")
            for source in response.sources[:2]:  # Show top 2
                print(f"   • Source {source['id']} (relevance: {source['relevance_score']:.2f}):")
                preview = source['content'][:100] + "..." if len(source['content']) > 100 else source['content']
                print(f"     {preview}")
        
        print(f"\n💭 Reasoning:")
        print(f"   {response.reasoning}")
    
    print("\n" + "=" * 80)
    print("Key Anti-Hallucination Features Demonstrated:")
    print("=" * 80)
    print("1. ✅ Source Attribution: Every answer cites specific sources")
    print("2. ✅ Confidence Scoring: Quantifies answer reliability")
    print("3. ✅ Grounding Verification: Ensures answers are based on context")
    print("4. ✅ Explicit Uncertainty: Admits when information is unavailable")
    print("5. ✅ Low Temperature: Reduces creative hallucinations")
    print()


if __name__ == "__main__":
    main()
