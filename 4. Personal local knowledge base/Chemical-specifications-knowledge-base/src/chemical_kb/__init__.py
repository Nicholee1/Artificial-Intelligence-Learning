"""
Chemical Specifications Knowledge Base

A comprehensive knowledge base system for chemical engineering documents
with AI-powered search and question answering capabilities.
"""

__version__ = "1.0.0"
__author__ = "Chemical KB Team"
__email__ = "contact@chemical-kb.com"

# Import main components
from .core.pipeline import IntegratedPipeline
from .core.vector_store import ChemicalVectorStore
from .core.pdf_processor import ChemicalPDFProcessor
from .ai.service import AIService
from .ai.rag import RAGPipeline
from .ai.chat import AIChatInterface
from .api.server import create_app

__all__ = [
    "IntegratedPipeline",
    "ChemicalVectorStore", 
    "ChemicalPDFProcessor",
    "AIService",
    "RAGPipeline",
    "AIChatInterface",
    "create_app",
]
