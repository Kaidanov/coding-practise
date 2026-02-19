# coding-practise
leet code, ai , team lead questions, data structures, interesting cases

## 🤖 RAG Chatbot with Evaluations

Production-ready RAG (Retrieval-Augmented Generation) chatbot implementation with comprehensive evaluation framework and anti-hallucination techniques.

**[📖 Full Interview Guide](./RAG_INTERVIEW_GUIDE.md)**

### Key Features
- ✅ **Anti-Hallucination Techniques**: Source attribution, confidence scoring, grounding verification
- ✅ **Comprehensive Evaluations**: Faithfulness, relevance, context quality, factual correctness
- ✅ **Production-Ready**: Vector storage, error handling, monitoring-ready
- ✅ **Well-Tested**: Unit tests with mocks and fixtures

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Add your OPENAI_API_KEY to .env

# Run examples
python examples/basic_rag_example.py
python examples/evaluation_example.py

# Run tests
pytest tests/ -v
```

### Structure
```
rag_chatbot/
├── vector_store.py   # Document storage and retrieval
├── rag_pipeline.py   # Core RAG with anti-hallucination
└── evaluator.py      # Evaluation framework

examples/
├── basic_rag_example.py     # Basic usage demonstration
└── evaluation_example.py    # Evaluation workflow

tests/
├── test_vector_store.py     # Vector store tests
├── test_rag_pipeline.py     # RAG pipeline tests
└── test_evaluator.py        # Evaluator tests
```

### Topics Covered (Interview Prep)
- RAG architecture and implementation
- Vector databases and embeddings
- LLM integration and prompt engineering
- Evaluation strategies (faithfulness, relevance, accuracy)
- Anti-hallucination techniques
- Production considerations (monitoring, scaling, cost)

See [RAG_INTERVIEW_GUIDE.md](./RAG_INTERVIEW_GUIDE.md) for detailed documentation.
