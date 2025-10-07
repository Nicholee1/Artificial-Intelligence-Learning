"""
AI services for the Chemical Knowledge Base.

This module provides AI-powered features including
question answering, document analysis, and intelligent search.
"""

from .service import AIService
from .rag import RAGPipeline
from .chat import AIChatInterface

__all__ = [
    "AIService",
    "RAGPipeline", 
    "AIChatInterface",
]
