"""
RAG Pipeline - Core chatbot implementation with anti-hallucination techniques.

This module implements:
- Context-aware question answering
- Citation and source attribution
- Confidence scoring
- Answer verification
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import os
from openai import OpenAI

from .vector_store import VectorStore, RetrievedDocument


@dataclass
class RAGResponse:
    """Response from the RAG pipeline."""
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    is_grounded: bool
    reasoning: str


class RAGPipeline:
    """
    Production-ready RAG pipeline with anti-hallucination techniques.
    
    Key features:
    - Source attribution for every claim
    - Confidence scoring
    - Answer grounding verification
    - Fallback for low-confidence responses
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        temperature: float = 0.1,  # Low temperature for factual responses
        confidence_threshold: float = 0.7
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            vector_store: VectorStore instance for document retrieval
            api_key: OpenAI API key (or from environment)
            model: LLM model to use
            temperature: Sampling temperature (lower = more deterministic)
            confidence_threshold: Minimum confidence for accepting answers
        """
        self.vector_store = vector_store
        self.model = model
        self.temperature = temperature
        self.confidence_threshold = confidence_threshold
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    
    def query(
        self,
        question: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> RAGResponse:
        """
        Process a query through the RAG pipeline.
        
        Args:
            question: User's question
            top_k: Number of documents to retrieve
            filter_metadata: Optional metadata filters for retrieval
        
        Returns:
            RAGResponse with answer, sources, and confidence metrics
        """
        # Step 1: Retrieve relevant documents
        retrieved_docs = self.vector_store.search(
            query=question,
            top_k=top_k,
            filter_metadata=filter_metadata
        )
        
        if not retrieved_docs:
            return self._no_context_response()
        
        # Step 2: Build context with source attribution
        context = self._build_context(retrieved_docs)
        
        # Step 3: Generate answer with explicit grounding requirement
        answer, reasoning = self._generate_grounded_answer(question, context)
        
        # Step 4: Verify answer grounding and calculate confidence
        is_grounded, confidence = self._verify_grounding(
            answer, retrieved_docs, reasoning
        )
        
        # Step 5: Apply confidence threshold
        if confidence < self.confidence_threshold:
            return self._low_confidence_response(confidence, retrieved_docs)
        
        # Step 6: Extract and format sources
        sources = self._format_sources(retrieved_docs)
        
        return RAGResponse(
            answer=answer,
            sources=sources,
            confidence=confidence,
            is_grounded=is_grounded,
            reasoning=reasoning
        )
    
    def _build_context(self, docs: List[RetrievedDocument]) -> str:
        """Build context string from retrieved documents with source IDs."""
        context_parts = []
        for i, doc in enumerate(docs, 1):
            context_parts.append(f"[Source {i}]:\n{doc.content}\n")
        return "\n".join(context_parts)
    
    def _generate_grounded_answer(
        self, question: str, context: str
    ) -> Tuple[str, str]:
        """
        Generate an answer that must be grounded in the provided context.
        
        Returns:
            Tuple of (answer, reasoning)
        """
        system_prompt = """You are a precise AI assistant that ONLY answers based on the provided context.

CRITICAL RULES:
1. Your answer MUST be directly supported by the context provided
2. Cite specific source numbers [Source X] for every claim
3. If the context doesn't contain enough information, say "I don't have enough information to answer this question"
4. Never make assumptions or add information not in the context
5. If conflicting information exists, acknowledge it

Your response should have two parts:
1. ANSWER: The factual answer with citations
2. REASONING: Brief explanation of which sources support your answer"""

        user_prompt = f"""Context:
{context}

Question: {question}

Provide your response in this format:
ANSWER: [Your answer with [Source X] citations]
REASONING: [Which sources you used and why]"""

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        content = response.choices[0].message.content
        
        # Parse answer and reasoning
        if "REASONING:" in content:
            parts = content.split("REASONING:")
            answer = parts[0].replace("ANSWER:", "").strip()
            reasoning = parts[1].strip()
        else:
            answer = content.replace("ANSWER:", "").strip()
            reasoning = "No explicit reasoning provided"
        
        return answer, reasoning
    
    def _verify_grounding(
        self,
        answer: str,
        docs: List[RetrievedDocument],
        reasoning: str
    ) -> Tuple[bool, float]:
        """
        Verify that the answer is grounded in the retrieved documents.
        
        Returns:
            Tuple of (is_grounded, confidence_score)
        """
        # Check for "I don't have enough information" response
        uncertain_phrases = [
            "don't have enough information",
            "cannot answer",
            "not enough context",
            "insufficient information"
        ]
        
        if any(phrase in answer.lower() for phrase in uncertain_phrases):
            return False, 0.3
        
        # Check for source citations in answer
        has_citations = "[Source" in answer
        
        # Calculate confidence based on:
        # 1. Presence of citations
        # 2. Retrieval scores of top documents
        # 3. Length and specificity of reasoning
        
        citation_score = 0.4 if has_citations else 0.1
        retrieval_score = min(docs[0].score if docs else 0, 0.4)
        reasoning_score = min(len(reasoning) / 500, 0.2)  # Up to 0.2 for detailed reasoning
        
        confidence = citation_score + retrieval_score + reasoning_score
        is_grounded = has_citations and confidence >= 0.6
        
        return is_grounded, min(confidence, 1.0)
    
    def _format_sources(
        self, docs: List[RetrievedDocument]
    ) -> List[Dict[str, Any]]:
        """Format retrieved documents as sources with metadata."""
        sources = []
        for i, doc in enumerate(docs, 1):
            sources.append({
                "id": i,
                "content": doc.content,
                "metadata": doc.metadata,
                "relevance_score": round(doc.score, 3),
                "doc_id": doc.doc_id
            })
        return sources
    
    def _no_context_response(self) -> RAGResponse:
        """Response when no relevant documents are found."""
        return RAGResponse(
            answer="I couldn't find any relevant information to answer your question.",
            sources=[],
            confidence=0.0,
            is_grounded=False,
            reasoning="No relevant documents retrieved"
        )
    
    def _low_confidence_response(
        self, confidence: float, docs: List[RetrievedDocument]
    ) -> RAGResponse:
        """Response when confidence is below threshold."""
        return RAGResponse(
            answer=(
                f"I found some potentially relevant information, but I'm not confident "
                f"enough (confidence: {confidence:.2f}) to provide a definitive answer. "
                f"Please rephrase your question or consult the source documents directly."
            ),
            sources=self._format_sources(docs),
            confidence=confidence,
            is_grounded=False,
            reasoning="Confidence below threshold"
        )
