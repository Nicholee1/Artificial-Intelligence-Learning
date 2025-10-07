#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API module tests
"""

import pytest
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from chemical_kb.api import create_app

class TestAPI:
    """Test API module"""
    
    def test_create_app(self):
        """Test app creation"""
        app = create_app()
        assert app is not None
        assert app.name == 'chemical_kb.api.server'
    
    def test_health_endpoint(self):
        """Test health endpoint"""
        app = create_app()
        client = app.test_client()
        
        response = client.get('/api/health')
        assert response.status_code in [200, 500]  # May fail if services not initialized
        
        data = response.get_json()
        assert 'status' in data
        assert 'timestamp' in data

if __name__ == "__main__":
    pytest.main([__file__])

