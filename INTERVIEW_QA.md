# Interview Questions & Answers - RAG & LLM Systems

## Common Interview Questions for Team Lead AI/Backend Engineering Role

### 1. Hallucination Prevention

**Q: What techniques do you use to prevent LLM hallucinations in production?**

**A:** I implement a multi-layered approach:

1. **Source Attribution & Citations**
   - Every claim must cite specific sources: `[Source 1]`, `[Source 2]`
   - Forces the model to ground answers in retrieved context
   - Makes it easy to verify claims

2. **Confidence Scoring**
   - Multi-factor confidence calculation:
     - Citation presence (40%)
     - Retrieval quality (40%)  
     - Reasoning depth (20%)
   - Threshold-based fallback when confidence < 0.7

3. **Explicit System Prompts**
   ```
   "You ONLY answer based on provided context.
   If information is not in context, say 'I don't have enough information'"
   ```

4. **Low Temperature Sampling** (0.1 vs 0.7)
   - Reduces creativity and randomness
   - More deterministic, factual responses

5. **Answer Verification**
   - Check for uncertainty phrases
   - Verify citation presence
   - Cross-reference with retrieval scores

6. **Human-in-the-Loop**
   - Thumbs up/down feedback
   - Flag suspicious answers for review
   - Continuous improvement loop

**Code Example:**
```python
# In rag_pipeline.py
system_prompt = """CRITICAL RULES:
1. Answer MUST be directly supported by context
2. Cite specific sources for every claim
3. If unsure, say "I don't have enough information"
4. Never make assumptions"""

# Low temperature
response = self.client.chat.completions.create(
    model=self.model,
    temperature=0.1,  # Key: Low temperature
    messages=[...]
)

# Confidence threshold
if confidence < self.confidence_threshold:
    return self._low_confidence_response()
```

---

### 2. RAG System Evaluation

**Q: How do you evaluate the quality of a RAG system?**

**A:** I use a comprehensive evaluation framework with multiple dimensions:

**Automated Metrics:**

1. **Faithfulness (Groundedness)** - 0 to 1
   - Are answers supported by retrieved context?
   - LLM-as-judge evaluation
   - Key metric for hallucination prevention

2. **Answer Relevance** - 0 to 1
   - Does the answer address the question?
   - Semantic similarity + LLM evaluation

3. **Context Relevance** - 0 to 1
   - Are retrieved documents relevant?
   - Measures retrieval quality

4. **Factual Correctness** - 0 to 1 (when ground truth available)
   - How accurate vs. reference answer?

**Production Metrics:**

5. **User Satisfaction**
   - Thumbs up/down feedback
   - Task completion rate
   - User retention

6. **System Performance**
   - Response latency (p50, p95, p99)
   - Cost per query
   - Error rate

7. **Business Impact**
   - Reduction in support tickets
   - Customer satisfaction (CSAT)
   - Conversion rate impact

**Evaluation Process:**
```python
# 1. Create test dataset with ground truth
test_cases = [
    {
        "query": "What is the coverage limit?",
        "ground_truth": "$1M per occurrence"
    }
]

# 2. Generate answers
response = rag_pipeline.query(query)

# 3. Evaluate all dimensions
result = evaluator.evaluate(
    query=query,
    answer=response.answer,
    context=response.sources,
    ground_truth=ground_truth
)

# 4. Aggregate and analyze
summary = summarize_evaluations(results)
print(f"Average Faithfulness: {summary['faithfulness']['mean']}")
```

**Best Practices:**
- Combine automated + human evaluation
- Continuous monitoring in production
- A/B test improvements
- Regular re-evaluation as system evolves

---

### 3. RAG Architecture Design

**Q: How would you architect a production RAG system?**

**A:** Here's my production architecture:

```
┌─────────────────────────────────────────────────────────┐
│                    User Interface                        │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                  API Gateway                             │
│  • Rate limiting  • Authentication  • Logging            │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                  RAG Orchestrator                        │
│  • Query preprocessing  • Routing  • Caching             │
└──────────┬────────────────────────────────┬─────────────┘
           │                                │
┌──────────▼──────────┐         ┌──────────▼──────────────┐
│  Vector Database    │         │    LLM Service          │
│  (Pinecone/Weaviate)│         │    (OpenAI/Anthropic)   │
│  • Embeddings       │         │    • Generation         │
│  • Similarity search│         │    • With citations     │
└──────────┬──────────┘         └──────────┬──────────────┘
           │                               │
           └───────────┬───────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│               Monitoring & Evaluation                    │
│  • Metrics  • Alerting  • A/B Testing                   │
└─────────────────────────────────────────────────────────┘
```

**Key Components:**

1. **Document Ingestion Pipeline**
   - Chunk documents intelligently (semantic boundaries)
   - Generate embeddings
   - Store with metadata
   - Regular re-indexing

2. **Retrieval Layer**
   - Hybrid search (semantic + keyword)
   - Reranking for precision
   - Query expansion/rewriting
   - Metadata filtering

3. **Generation Layer**
   - Context building with citations
   - Prompt engineering
   - Temperature control
   - Response streaming

4. **Verification Layer**
   - Confidence scoring
   - Grounding check
   - Fallback mechanisms
   - Source attribution

5. **Monitoring**
   - Latency tracking
   - Cost monitoring
   - Quality metrics
   - User feedback

**Scalability Considerations:**
- Cache common queries (Redis)
- Async processing
- Load balancing
- Vector DB sharding
- Rate limiting per user/tier

---

### 4. Context Window Management

**Q: How do you handle context window limitations in RAG?**

**A:** Several strategies:

1. **Smart Chunking**
   - Chunk at semantic boundaries (paragraphs, sections)
   - Overlap between chunks for continuity
   - Store chunk metadata for reconstruction

2. **Selective Retrieval**
   - Only retrieve most relevant chunks (top-k)
   - Dynamic k based on relevance scores
   - Reranking after initial retrieval

3. **Context Compression**
   - Summarize long documents
   - Extract key information
   - Remove redundancy

4. **Tiered Retrieval**
   - First pass: High-level retrieval
   - Second pass: Drill down to relevant sections
   - Hierarchical document structure

5. **Token Counting**
   - Track token usage (tiktoken)
   - Ensure: query + context + response < max_tokens
   - Truncate context if needed (keep most relevant)

**Code Example:**
```python
import tiktoken

def build_context_within_limit(
    docs: List[str],
    max_tokens: int = 4000
) -> str:
    encoder = tiktoken.encoding_for_model("gpt-4")
    context = []
    token_count = 0
    
    for doc in docs:
        doc_tokens = len(encoder.encode(doc))
        if token_count + doc_tokens < max_tokens:
            context.append(doc)
            token_count += doc_tokens
        else:
            break
    
    return "\n\n".join(context)
```

---

### 5. Improving Retrieval Quality

**Q: What techniques improve retrieval accuracy in RAG?**

**A:** Multiple approaches:

1. **Query Optimization**
   - Query expansion (add synonyms)
   - Query rewriting (clarify ambiguity)
   - Multi-query generation

2. **Hybrid Search**
   - Combine semantic (vector) + keyword (BM25)
   - Balance with alpha parameter
   - Best of both worlds

3. **Reranking**
   - Initial retrieval: Fast but less precise
   - Reranking: Slower but more accurate
   - Cross-encoder models for reranking

4. **Better Embeddings**
   - Domain-specific fine-tuned models
   - Larger embedding models
   - Multi-vector representations

5. **Metadata Filtering**
   - Filter by date, category, source
   - Hybrid filtering + semantic search
   - User-specific contexts

6. **Feedback Loop**
   - Track which retrievals led to good answers
   - Negative examples (bad retrievals)
   - Fine-tune retrieval model

**Example:**
```python
# Hybrid search
from hybrid_search import HybridRetriever

retriever = HybridRetriever(
    vector_store=vector_store,
    bm25_index=bm25_index,
    alpha=0.5  # 50% semantic, 50% keyword
)

# With reranking
results = retriever.search(query, top_k=20)
reranked = reranker.rerank(query, results, top_n=5)
```

---

### 6. Cost Optimization

**Q: How do you optimize costs in production LLM systems?**

**A:** Cost optimization strategy:

1. **Caching**
   - Cache identical queries
   - Semantic similarity cache (similar queries)
   - TTL-based invalidation
   - Can reduce costs by 40-60%

2. **Model Selection**
   - Use cheaper models when possible
   - GPT-3.5 for simple queries, GPT-4 for complex
   - Route based on query complexity
   - Consider open-source models (Llama, Mistral)

3. **Prompt Optimization**
   - Shorter prompts = lower costs
   - Remove unnecessary examples
   - Efficient context building
   - Token counting

4. **Batching**
   - Batch similar queries
   - Shared context across batch
   - Reduces per-query overhead

5. **Smart Retrieval**
   - Retrieve only what's needed
   - Dynamic top-k based on query
   - Don't over-retrieve

6. **Monitoring & Alerts**
   - Track cost per query
   - Set budget alerts
   - Identify expensive patterns
   - A/B test cost vs. quality trade-offs

**Cost Breakdown Example:**
```
Typical Query Cost:
- Embedding generation: $0.0001
- Vector search: $0.00001
- LLM generation (GPT-4): $0.03
- Total: ~$0.03 per query

With caching (50% hit rate):
- Average: $0.015 per query
- 50% cost reduction
```

---

### 7. Team Leadership in AI Projects

**Q: How do you lead a team building AI/ML systems?**

**A:** My leadership approach:

1. **Clear Vision & Roadmap**
   - Define success metrics early
   - Break into phases (MVP → Production → Optimization)
   - Communicate trade-offs clearly

2. **Technical Excellence**
   - Code reviews focused on quality
   - Establish best practices (testing, monitoring)
   - Encourage experimentation
   - Share learnings across team

3. **Cross-Functional Collaboration**
   - Work with Product on requirements
   - Partner with Data Science on model selection
   - Engage with Business on metrics that matter
   - Regular stakeholder updates

4. **Continuous Learning**
   - Stay updated on latest LLM developments
   - Attend conferences, read papers
   - Internal knowledge sharing sessions
   - Experiment with new techniques

5. **Risk Management**
   - Identify technical risks early
   - Plan for failure modes
   - Implement monitoring and alerts
   - Have rollback strategies

6. **Team Development**
   - Mentoring and coaching
   - Career growth conversations
   - Delegate challenging problems
   - Celebrate wins

**AI-Specific Challenges:**
- Managing non-determinism
- Evaluating qualitative outputs
- Balancing quality vs. cost
- Keeping up with rapid changes

---

### 8. Production Readiness

**Q: What makes an AI system production-ready?**

**A:** Production checklist:

**Reliability:**
- [ ] Error handling and graceful degradation
- [ ] Retry logic with exponential backoff
- [ ] Circuit breakers for external services
- [ ] Fallback responses

**Monitoring:**
- [ ] Latency tracking (p50, p95, p99)
- [ ] Error rate monitoring
- [ ] Cost tracking per query
- [ ] Quality metrics (faithfulness, relevance)
- [ ] User feedback collection

**Security:**
- [ ] API key management (secrets manager)
- [ ] Input sanitization (prompt injection prevention)
- [ ] Rate limiting per user
- [ ] Data privacy compliance
- [ ] Audit logging

**Scalability:**
- [ ] Load testing
- [ ] Auto-scaling
- [ ] Caching strategy
- [ ] Database optimization

**Observability:**
- [ ] Structured logging
- [ ] Distributed tracing
- [ ] Dashboards
- [ ] Alerting

**Testing:**
- [ ] Unit tests
- [ ] Integration tests
- [ ] Evaluation test suite
- [ ] A/B testing framework

**Documentation:**
- [ ] API documentation
- [ ] Architecture diagrams
- [ ] Runbooks for incidents
- [ ] Onboarding guides

---

## Quick Reference: Key Metrics

| Metric | What it Measures | Target |
|--------|------------------|--------|
| Faithfulness | Grounding in context | > 0.8 |
| Answer Relevance | Addresses question | > 0.8 |
| Context Relevance | Retrieval quality | > 0.7 |
| User Satisfaction | Thumbs up ratio | > 80% |
| Latency (p95) | Response time | < 3s |
| Cost per query | Efficiency | < $0.05 |
| Error rate | Reliability | < 1% |

## Interview Tips

1. **Be Specific**: Use concrete examples from this codebase
2. **Show Trade-offs**: Discuss when you'd choose different approaches
3. **Production Focus**: Emphasize monitoring, costs, reliability
4. **Team Perspective**: Show leadership and collaboration mindset
5. **Stay Current**: Mention recent LLM developments

## Hands-On Exercise Prep

Be ready to:
- Write a RAG pipeline from scratch (30-60 min)
- Debug a hallucination issue
- Design evaluation metrics
- Optimize for cost or latency
- Review and improve existing code

Use this codebase as reference!
