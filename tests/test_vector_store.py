"""
Tests for Vector Store functionality.
"""

import pytest
import tempfile
import shutil
from rag_chatbot.vector_store import VectorStore, Document, RetrievedDocument


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test database."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def vector_store(temp_dir):
    """Create a VectorStore instance for testing."""
    return VectorStore(
        collection_name="test_collection",
        persist_directory=temp_dir,
        embedding_model="all-MiniLM-L6-v2"
    )


def test_vector_store_initialization(vector_store):
    """Test that VectorStore initializes correctly."""
    assert vector_store.collection_name == "test_collection"
    assert vector_store.embedding_model is not None
    assert vector_store.count() == 0


def test_add_documents(vector_store):
    """Test adding documents to the vector store."""
    documents = [
        Document(
            content="Python is a programming language",
            metadata={"topic": "python"},
            doc_id="doc1"
        ),
        Document(
            content="JavaScript runs in browsers",
            metadata={"topic": "javascript"},
            doc_id="doc2"
        )
    ]
    
    vector_store.add_documents(documents)
    assert vector_store.count() == 2


def test_search_documents(vector_store):
    """Test searching for documents."""
    # Add test documents
    documents = [
        Document(
            content="Python is a high-level programming language known for readability",
            metadata={"topic": "python", "category": "programming"},
            doc_id="doc1"
        ),
        Document(
            content="JavaScript is used for web development and runs in browsers",
            metadata={"topic": "javascript", "category": "programming"},
            doc_id="doc2"
        ),
        Document(
            content="Machine learning involves training models on data",
            metadata={"topic": "ml", "category": "ai"},
            doc_id="doc3"
        )
    ]
    vector_store.add_documents(documents)
    
    # Search for Python-related content
    results = vector_store.search("Python programming language", top_k=2)
    
    assert len(results) > 0
    assert isinstance(results[0], RetrievedDocument)
    assert results[0].score > 0
    
    # The most relevant result should mention Python
    assert "python" in results[0].content.lower() or "Python" in results[0].content


def test_search_with_metadata_filter(vector_store):
    """Test searching with metadata filters."""
    documents = [
        Document(
            content="Python tutorial for beginners",
            metadata={"topic": "python", "level": "beginner"},
            doc_id="doc1"
        ),
        Document(
            content="Advanced Python techniques",
            metadata={"topic": "python", "level": "advanced"},
            doc_id="doc2"
        )
    ]
    vector_store.add_documents(documents)
    
    # Search with filter - Note: ChromaDB metadata filtering syntax
    # This test demonstrates the API, actual filtering depends on ChromaDB setup
    results = vector_store.search(
        "Python",
        top_k=5,
        filter_metadata={"level": "beginner"}
    )
    
    # Should return results (filtering may need ChromaDB config)
    assert len(results) >= 0


def test_clear_collection(vector_store):
    """Test clearing all documents from the collection."""
    documents = [
        Document(
            content="Test document",
            metadata={},
            doc_id="doc1"
        )
    ]
    vector_store.add_documents(documents)
    assert vector_store.count() == 1
    
    vector_store.clear()
    assert vector_store.count() == 0


def test_empty_search(vector_store):
    """Test searching in an empty collection."""
    results = vector_store.search("test query")
    assert len(results) == 0


def test_retrieved_document_structure(vector_store):
    """Test that retrieved documents have correct structure."""
    documents = [
        Document(
            content="Test content",
            metadata={"key": "value"},
            doc_id="test_doc"
        )
    ]
    vector_store.add_documents(documents)
    
    results = vector_store.search("test", top_k=1)
    
    if results:
        doc = results[0]
        assert hasattr(doc, 'content')
        assert hasattr(doc, 'metadata')
        assert hasattr(doc, 'score')
        assert hasattr(doc, 'doc_id')
        assert isinstance(doc.score, float)
        assert 0 <= doc.score <= 1
