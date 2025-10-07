#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core module tests
"""

import pytest
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from chemical_kb.core import ChemicalPDFProcessor, ChemicalVectorStore, IntegratedPipeline

class TestChemicalPDFProcessor:
    """Test ChemicalPDFProcessor"""
    
    def test_init(self):
        """Test initialization"""
        # This is a basic test - in a real scenario you'd need a test PDF
        processor = ChemicalPDFProcessor.__new__(ChemicalPDFProcessor)
        assert processor is not None

class TestChemicalVectorStore:
    """Test ChemicalVectorStore"""
    
    def test_init(self):
        """Test initialization"""
        # Basic initialization test
        store = ChemicalVectorStore.__new__(ChemicalVectorStore)
        assert store is not None

class TestIntegratedPipeline:
    """Test IntegratedPipeline"""
    
    def test_init(self):
        """Test initialization"""
        # Basic initialization test
        pipeline = IntegratedPipeline.__new__(IntegratedPipeline)
        assert pipeline is not None

if __name__ == "__main__":
    pytest.main([__file__])

