"""
API services for the Chemical Knowledge Base.

This module provides REST API endpoints for
web-based access to the knowledge base functionality.
"""

from .server import create_app

__all__ = [
    "create_app",
]
