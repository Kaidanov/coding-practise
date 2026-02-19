"""
Tests for RAG Pipeline functionality.
"""

import pytest
from unittest.mock import Mock, patch
from rag_chatbot.rag_pipeline import RAGPipeline, RAGResponse
from rag_chatbot.vector_store import VectorStore, RetrievedDocument


@pytest.fixture
def mock_vector_store():
    """Create a mock VectorStore."""
    mock = Mock(spec=VectorStore)
    return mock


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    with patch('rag_chatbot.rag_pipeline.OpenAI') as mock_openai:
        yield mock_openai


def test_rag_pipeline_initialization(mock_vector_store):
    """Test RAG pipeline initialization."""
    pipeline = RAGPipeline(
        vector_store=mock_vector_store,
        api_key="test-key",
        model="gpt-4",
        temperature=0.1
    )
    
    assert pipeline.model == "gpt-4"
    assert pipeline.temperature == 0.1
    assert pipeline.confidence_threshold == 0.7


def test_no_context_response(mock_vector_store):
    """Test response when no documents are retrieved."""
    mock_vector_store.search.return_value = []
    
    pipeline = RAGPipeline(vector_store=mock_vector_store, api_key="test-key")
    response = pipeline.query("test question")
    
    assert isinstance(response, RAGResponse)
    assert response.confidence == 0.0
    assert response.is_grounded == False
    assert len(response.sources) == 0


def test_build_context():
    """Test context building from retrieved documents."""
    mock_store = Mock(spec=VectorStore)
    pipeline = RAGPipeline(vector_store=mock_store, api_key="test-key")
    
    docs = [
        RetrievedDocument(
            content="Document 1 content",
            metadata={},
            score=0.9,
            doc_id="doc1"
        ),
        RetrievedDocument(
            content="Document 2 content",
            metadata={},
            score=0.8,
            doc_id="doc2"
        )
    ]
    
    context = pipeline._build_context(docs)
    
    assert "[Source 1]" in context
    assert "[Source 2]" in context
    assert "Document 1 content" in context
    assert "Document 2 content" in context


def test_verify_grounding_with_citations():
    """Test grounding verification with citations."""
    mock_store = Mock(spec=VectorStore)
    pipeline = RAGPipeline(vector_store=mock_store, api_key="test-key")
    
    answer = "The answer is X [Source 1] and Y [Source 2]"
    docs = [
        RetrievedDocument(
            content="X is true",
            metadata={},
            score=0.9,
            doc_id="doc1"
        )
    ]
    reasoning = "Used Source 1 for X and Source 2 for Y"
    
    is_grounded, confidence = pipeline._verify_grounding(answer, docs, reasoning)
    
    assert isinstance(is_grounded, bool)
    assert isinstance(confidence, float)
    assert 0.0 <= confidence <= 1.0
    assert is_grounded == True  # Has citations
    assert confidence > 0.5  # Should have decent confidence with citations


def test_verify_grounding_uncertain_response():
    """Test grounding verification with uncertain response."""
    mock_store = Mock(spec=VectorStore)
    pipeline = RAGPipeline(vector_store=mock_store, api_key="test-key")
    
    answer = "I don't have enough information to answer this question"
    docs = []
    reasoning = "No relevant context"
    
    is_grounded, confidence = pipeline._verify_grounding(answer, docs, reasoning)
    
    assert is_grounded == False
    assert confidence < 0.5  # Should have low confidence


def test_format_sources():
    """Test source formatting."""
    mock_store = Mock(spec=VectorStore)
    pipeline = RAGPipeline(vector_store=mock_store, api_key="test-key")
    
    docs = [
        RetrievedDocument(
            content="Content 1",
            metadata={"page": 1, "source": "doc.pdf"},
            score=0.95,
            doc_id="doc1"
        ),
        RetrievedDocument(
            content="Content 2",
            metadata={"page": 2},
            score=0.85,
            doc_id="doc2"
        )
    ]
    
    sources = pipeline._format_sources(docs)
    
    assert len(sources) == 2
    assert sources[0]["id"] == 1
    assert sources[0]["content"] == "Content 1"
    assert sources[0]["metadata"]["page"] == 1
    assert sources[0]["relevance_score"] == 0.95
    assert sources[1]["id"] == 2


def test_low_confidence_response(mock_vector_store):
    """Test low confidence response handling."""
    pipeline = RAGPipeline(
        vector_store=mock_vector_store,
        api_key="test-key",
        confidence_threshold=0.8
    )
    
    docs = [
        RetrievedDocument(
            content="Some content",
            metadata={},
            score=0.5,
            doc_id="doc1"
        )
    ]
    
    response = pipeline._low_confidence_response(0.5, docs)
    
    assert isinstance(response, RAGResponse)
    assert response.confidence == 0.5
    assert response.is_grounded == False
    assert "not confident enough" in response.answer.lower()
    assert len(response.sources) > 0
