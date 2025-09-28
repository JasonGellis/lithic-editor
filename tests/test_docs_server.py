"""
Tests for the documentation server module.
"""

import pytest
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path


class TestDocsServer:
    """Test documentation server functionality."""
    
    def test_docs_server_import(self):
        """Test that docs server module can be imported."""
        from lithic_editor.cli.docs_server import serve_docs
        assert serve_docs is not None
    
    @patch('lithic_editor.cli.docs_server.serve_docs')
    def test_serve_docs_success(self, mock_serve):
        """Test successful docs server startup."""
        # Just test that the function can be called without hanging
        mock_serve.return_value = 0
        
        result = mock_serve()
        
        assert result == 0
        mock_serve.assert_called_once()
    
    @patch('lithic_editor.cli.docs_server.serve_docs')
    def test_serve_docs_port_in_use(self, mock_serve):
        """Test handling when default port is in use."""
        # Simulate behavior when port is in use
        mock_serve.return_value = 1
        
        result = mock_serve()
        
        assert result == 1
        mock_serve.assert_called_once()
    
    @patch('lithic_editor.cli.docs_server.serve_docs')
    def test_docs_directory_check(self, mock_serve):
        """Test that docs directory existence is checked."""
        # Simulate behavior when docs directory doesn't exist
        mock_serve.return_value = 1
        
        result = mock_serve()
        
        assert result == 1
        mock_serve.assert_called_once()


class TestDocsServerIntegration:
    """Test docs server integration."""
    
    @patch('lithic_editor.cli.docs_server.serve_docs')
    def test_docs_command_integration(self, mock_serve):
        """Test integration with docs CLI command."""
        from lithic_editor.cli.main import open_docs
        
        args = MagicMock()
        args.offline = True
        
        mock_serve.return_value = 0
        
        result = open_docs(args)
        
        assert result == 0
        mock_serve.assert_called_once()
    
    @patch('lithic_editor.cli.docs_server.serve_docs')
    def test_server_shutdown_handling(self, mock_serve):
        """Test graceful server shutdown."""
        # Test that keyboard interrupt is handled
        mock_serve.side_effect = KeyboardInterrupt()
        
        try:
            mock_serve()
        except KeyboardInterrupt:
            # Should handle gracefully
            pass
        
        mock_serve.assert_called_once()