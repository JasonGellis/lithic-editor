"""
Simple HTTP server for serving bundled documentation.
This allows users to view docs without installing mkdocs.
"""

import http.server
import socketserver
import os
import webbrowser
from pathlib import Path
import threading


def serve_docs(port=8000):
    """
    Serve the bundled documentation using Python's built-in HTTP server.
    
    Args:
        port: Port to serve on (default 8000)
        
    Returns:
        int: Exit code
    """
    # Find the documentation directory
    package_dir = Path(__file__).parent.parent
    docs_dir = package_dir / 'docs_dist'
    
    if not docs_dir.exists():
        # Try alternate location (development mode)
        project_root = package_dir.parent
        site_dir = project_root / 'site'
        
        if site_dir.exists():
            docs_dir = site_dir
        else:
            print("✗ Documentation not found. The documentation may not be built yet.")
            print("  For developers: run 'mkdocs build' to build the documentation")
            print("  For users: view online at https://jasongellis.github.io/lithic-editor/")
            return 1
    
    # Change to the documentation directory
    os.chdir(docs_dir)
    
    # Create a simple HTTP server
    handler = http.server.SimpleHTTPRequestHandler
    
    # Suppress log messages
    handler.log_message = lambda *args: None
    
    try:
        with socketserver.TCPServer(("", port), handler) as httpd:
            url = f"http://127.0.0.1:{port}"
            print(f"✓ Documentation server started")
            print(f"  View documentation at: {url}")
            print(f"  Press Ctrl+C to stop the server")
            
            # Open browser in a separate thread to avoid blocking
            threading.Thread(target=lambda: webbrowser.open(url), daemon=True).start()
            
            # Serve forever (until interrupted)
            httpd.serve_forever()
            
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"✗ Port {port} is already in use")
            print(f"  Try a different port with: lithic-editor docs --offline --port {port+1}")
            return 1
        else:
            raise
    except KeyboardInterrupt:
        print("\n✓ Documentation server stopped")
        return 0
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(serve_docs())