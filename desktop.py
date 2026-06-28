import sys
import os
import threading
import socket
import time

# Helper to find a free port to avoid conflicts
def find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
        s.close()
        return port
    except Exception:
        return 8000

# Verify pywebview is installed
try:
    import webview
except ImportError:
    print("Error: 'pywebview' is not installed.")
    print("Please install it by running: pip install pywebview")
    sys.exit(1)

import uvicorn

# Start FastAPI server in a background thread
def start_server(port):
    # Import FastAPI app here to ensure sys.frozen path setup works
    from main import app
    print(f"Starting local server in desktop thread on port {port}...")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")

if __name__ == "__main__":
    # Configure path resolution for PyInstaller frozen execution
    if getattr(sys, 'frozen', False):
        # We are running as an executable
        # The temporary folder path is stored in sys._MEIPASS
        # Set working directory to the directory of the executable
        os.chdir(os.path.dirname(sys.executable))
    
    port = find_free_port()
    
    # Start server thread
    server_thread = threading.Thread(target=start_server, args=(port,), daemon=True)
    server_thread.start()
    
    # Give the server a small moment to initialize
    time.sleep(1.0)
    
    # Start the desktop window wrapper
    print(f"Launching desktop window for FormuLatex at http://127.0.0.1:{port}")
    webview.create_window(
        title="FormuLatex Desktop App",
        url=f"http://127.0.0.1:{port}",
        width=1200,
        height=800,
        resizable=True
    )
    
    webview.start()
    print("Desktop application closed.")
