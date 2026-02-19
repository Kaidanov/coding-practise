# RAG Chatbot with Evaluations - Interview Preparation Guide

## 🎯 Overview

This project demonstrates a production-ready **RAG (Retrieval-Augmented Generation)** chatbot implementation with comprehensive evaluation framework and anti-hallucination techniques. It's designed as interview preparation for a Team Lead AI/Backend Engineering position.

## 🏗️ Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                      RAG Pipeline                            │
│                                                              │
│  User Query → Retrieval → Context Building → LLM Generation │
│                  ↓            ↓                  ↓           │
│            Vector Store   Citations         Verification    │
└─────────────────────────────────────────────────────────────┘
```

### Key Modules

1. **`vector_store.py`**: Document storage and retrieval
   - ChromaDB for vector database
   - Sentence transformers for embeddings
   - Similarity search with metadata filtering

2. **`rag_pipeline.py`**: Core RAG implementation
   - Context-aware question answering
   - Citation and source attribution
   - Confidence scoring
   - Answer grounding verification

3. **`evaluator.py`**: Evaluation framework
   - Faithfulness/groundedness metrics
   - Answer relevance evaluation
   - Context relevance assessment
   - Factual correctness comparison

## 🛡️ Anti-Hallucination Techniques

### 1. **Source Attribution**
Every claim in the answer must cite specific sources:
```python
# Example output
"The coverage limit is $1 million per occurrence [Source 1]"
```

### 2. **Confidence Scoring**
Multi-factor confidence calculation:
- **Citation presence** (40% weight): Does answer include source citations?
- **Retrieval score** (40% weight): How relevant are retrieved documents?
- **Reasoning depth** (20% weight): Quality of explanation

```python
confidence = citation_score + retrieval_score + reasoning_score
```

### 3. **Grounding Verification**
Explicit check that answers are based on retrieved context:
- Detects uncertain language ("I don't have enough information")
- Verifies presence of citations
- Calculates confidence based on multiple factors

### 4. **Low Temperature Sampling**
```python
temperature=0.1  # Reduces creative hallucinations
```

### 5. **Explicit System Prompts**
```python
system_prompt = """You ONLY answer based on provided context.
CRITICAL RULES:
1. Answer MUST be directly supported by context
2. Cite specific sources for every claim
3. If unsure, say "I don't have enough information"
4. Never make assumptions"""
```

### 6. **Confidence Thresholds**
```python
if confidence < threshold:
    return low_confidence_response()
```

## 📊 Evaluation Framework

### Metrics Implemented

#### 1. **Faithfulness (Groundedness)**
- **Question**: Is the answer supported by the context?
- **Range**: 0.0 (not grounded) to 1.0 (fully grounded)
- **Method**: LLM-as-judge evaluation

#### 2. **Answer Relevance**
- **Question**: Does the answer address the question?
- **Range**: 0.0 (off-topic) to 1.0 (directly addresses)
- **Method**: Semantic similarity + LLM evaluation

#### 3. **Context Relevance**
- **Question**: Are retrieved documents relevant?
- **Range**: 0.0 (irrelevant) to 1.0 (highly relevant)
- **Method**: Retrieval quality assessment

#### 4. **Factual Correctness** (when ground truth available)
- **Question**: How accurate is the answer?
- **Range**: 0.0 (incorrect) to 1.0 (correct)
- **Method**: Comparison with reference answer

### Evaluation Process

```python
# 1. Define test cases with ground truth
test_cases = [
    {
        "query": "What is the coverage limit?",
        "ground_truth": "$1 million per occurrence"
    }
]

# 2. Generate answers
response = rag_pipeline.query(query)

# 3. Evaluate
result = evaluator.evaluate(
    query=query,
    answer=response.answer,
    context=response.sources,
    ground_truth=ground_truth
)

# 4. Analyze metrics
print(f"Faithfulness: {result.metrics.faithfulness}")
print(f"Overall Score: {result.metrics.overall_score}")
```

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Basic Usage

```python
from rag_chatbot.vector_store import VectorStore, Document
from rag_chatbot.rag_pipeline import RAGPipeline

# 1. Initialize vector store
vector_store = VectorStore(collection_name="my_docs")

# 2. Add documents
documents = [
    Document(
        content="Your document content here",
        metadata={"source": "doc1.pdf", "page": 1}
    )
]
vector_store.add_documents(documents)

# 3. Create RAG pipeline
rag = RAGPipeline(vector_store=vector_store)

# 4. Query
response = rag.query("Your question here")
print(f"Answer: {response.answer}")
print(f"Confidence: {response.confidence:.2%}")
print(f"Sources: {len(response.sources)}")
```

### Run Examples

```bash
# Basic RAG chatbot example
python examples/basic_rag_example.py

# Evaluation example
python examples/evaluation_example.py
```

## 💡 Interview Topics Covered

### 1. **RAG Architecture**
- Vector databases and embeddings
- Retrieval strategies
- Context injection
- LLM integration

### 2. **Evaluation Strategies**
- Automated metrics (faithfulness, relevance)
- LLM-as-judge evaluation
- Ground truth comparison
- A/B testing principles

### 3. **Non-Hallucination Approaches**
- Source attribution and citations
- Confidence scoring
- Answer verification
- Fallback mechanisms
- Temperature control
- Prompt engineering

### 4. **Production Considerations**
- Scalability (vector DB, caching)
- Monitoring and observability
- Cost optimization
- Error handling
- Testing strategies

## 🔍 Key Design Decisions

### Why ChromaDB?
- Lightweight, easy to set up
- Good for prototypes and small-scale production
- For large-scale: Consider Pinecone, Weaviate, or Qdrant

### Why Low Temperature (0.1)?
- Reduces randomness and creativity
- More deterministic, factual responses
- Critical for accuracy in insurance/legal domains

### Why Multi-Factor Confidence?
- Single metric insufficient
- Citation presence + retrieval quality + reasoning depth
- Threshold-based fallback for safety

### Why LLM-as-Judge Evaluation?
- Scalable automated evaluation
- Captures semantic similarity
- Complements human evaluation
- Best practice: Combine with human review

## 📈 Extending the System

### Add Hybrid Search
```python
# Combine semantic + keyword search
results = vector_store.hybrid_search(
    query=query,
    alpha=0.5  # Balance between semantic and keyword
)
```

### Add Query Rewriting
```python
# Improve retrieval with query expansion
expanded_query = query_rewriter.rewrite(original_query)
results = vector_store.search(expanded_query)
```

### Add Caching
```python
# Cache common queries
@cache(ttl=3600)
def query_with_cache(question):
    return rag_pipeline.query(question)
```

### Add Monitoring
```python
# Track metrics in production
monitor.log_query(
    query=query,
    confidence=response.confidence,
    latency=response_time,
    sources_used=len(response.sources)
)
```

## 🎤 Interview Discussion Points

### When asked about "non-hallucinating approach":

1. **Multi-layered defense**:
   - Source attribution (citations)
   - Confidence scoring
   - Grounding verification
   - Temperature control
   - Explicit uncertainty

2. **Evaluation is key**:
   - Automated metrics (faithfulness, relevance)
   - Continuous monitoring
   - Human-in-the-loop validation
   - A/B testing improvements

3. **It's about trade-offs**:
   - Strictness vs. helpfulness
   - Confidence threshold tuning
   - Sometimes "I don't know" is the right answer

### When asked about evaluations:

1. **Multiple dimensions**:
   - Faithfulness (grounding)
   - Relevance (answers question)
   - Context quality (retrieval)
   - Factual correctness (with ground truth)

2. **Automated + Human**:
   - LLM-as-judge for scale
   - Human review for quality
   - Continuous feedback loop

3. **Production metrics**:
   - User satisfaction (thumbs up/down)
   - Task completion rate
   - Response time
   - Cost per query

## 📚 References

Based on the provided blog posts:
1. LLM Application Expert Guide (Hebrew blog post)
2. LLM System Testing/Evaluation Processes (Hebrew blog post)

## 🔐 Security Notes

- Never commit API keys
- Use environment variables
- Implement rate limiting
- Sanitize user inputs
- Monitor for prompt injection

## 🧪 Testing

```bash
# Run tests
pytest tests/

# Run specific test
pytest tests/test_vector_store.py -v
```

## 📝 License

This is a coding practice/interview preparation project.
