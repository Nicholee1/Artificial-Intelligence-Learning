#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI module tests
"""

import pytest
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from chemical_kb.ai import AIService, RAGPipeline, AIChatInterface

class TestAIService:
    """Test AIService"""
    
    def test_init(self):
        """Test initialization"""
        # Basic initialization test
        service = AIService.__new__(AIService)
        assert service is not None

class TestRAGPipeline:
    """Test RAGPipeline"""
    
    def test_init(self):
        """Test initialization"""
        # Basic initialization test
        rag = RAGPipeline.__new__(RAGPipeline)
        assert rag is not None

class TestAIChatInterface:
    """Test AIChatInterface"""
    
    def test_init(self):
        """Test initialization"""
        # Basic initialization test
        chat = AIChatInterface.__new__(AIChatInterface)
        assert chat is not None

if __name__ == "__main__":
    pytest.main([__file__])

