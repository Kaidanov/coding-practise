"""
Vector Store Manager for document storage and retrieval.

This module handles:
- Document chunking and embedding
- Vector database operations
- Similarity search with metadata filtering
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


@dataclass
class Document:
    """Represents a document with content and metadata."""
    content: str
    metadata: Dict[str, Any]
    doc_id: Optional[str] = None


@dataclass
class RetrievedDocument:
    """Represents a retrieved document with relevance score."""
    content: str
    metadata: Dict[str, Any]
    score: float
    doc_id: str


class VectorStore:
    """
    Manages vector database for document storage and retrieval.
    
    Uses ChromaDB for local vector storage and sentence-transformers
    for generating embeddings.
    """
    
    def __init__(
        self,
        collection_name: str = "rag_documents",
        persist_directory: str = "./chroma_db",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize the vector store.
        
        Args:
            collection_name: Name of the collection in ChromaDB
            persist_directory: Directory to persist the database
            embedding_model: Name of the sentence-transformer model
        """
        self.collection_name = collection_name
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize ChromaDB client
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of Document objects to add
        """
        if not documents:
            return
        
        # Extract contents and generate embeddings
        contents = [doc.content for doc in documents]
        embeddings = self.embedding_model.encode(contents).tolist()
        
        # Prepare data for ChromaDB
        ids = [doc.doc_id or f"doc_{i}" for i, doc in enumerate(documents)]
        metadatas = [doc.metadata for doc in documents]
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings,
            documents=contents,
            metadatas=metadatas,
            ids=ids
        )
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedDocument]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            filter_metadata: Optional metadata filters
        
        Returns:
            List of RetrievedDocument objects
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0].tolist()
        
        # Search in collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_metadata
        )
        
        # Parse results
        retrieved_docs = []
        if results['documents'] and results['documents'][0]:
            for i, doc_content in enumerate(results['documents'][0]):
                retrieved_docs.append(RetrievedDocument(
                    content=doc_content,
                    metadata=results['metadatas'][0][i] if results['metadatas'][0] else {},
                    score=1 - results['distances'][0][i],  # Convert distance to similarity
                    doc_id=results['ids'][0][i]
                ))
        
        return retrieved_docs
    
    def clear(self) -> None:
        """Clear all documents from the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def count(self) -> int:
        """Get the number of documents in the collection."""
        return self.collection.count()
