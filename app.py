import subprocess
import sys
import os
import time
import signal
import webbrowser
from threading import Thread
import requests
from urllib3.exceptions import InsecureRequestWarning

# Suppress SSL warnings
requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)

def is_backend_ready():
    """Check if the backend API is ready"""
    try:
        response = requests.get('http://localhost:5001/health', verify=False)
        return response.status_code == 200
    except:
        return False

def is_frontend_ready():
    """Check if the frontend is ready"""
    try:
        response = requests.get('http://localhost:8501', verify=False)
        return response.status_code == 200
    except:
        return False

def run_backend():
    """Run the Flask backend API"""
    os.environ['FLASK_APP'] = 'src.backend.api'
    os.environ['FLASK_RUN_PORT'] = '5001'
    backend_cmd = ['flask', 'run', '--port=5001']
    return subprocess.Popen(backend_cmd)

def run_frontend():
    """Run the Streamlit frontend"""
    frontend_cmd = ["streamlit", "run", "src/frontend/streamlit.py"]
    return subprocess.Popen(frontend_cmd)

def main():
    print("Starting backend...")
    backend_process = run_backend()
    
    # Wait for backend to be ready
    print("Waiting for backend to be ready...")
    while not is_backend_ready():
        time.sleep(1)
    print("Backend is ready!")

    print("Starting frontend...")
    frontend_process = run_frontend()
    
    # Wait for frontend to be ready
    print("Waiting for frontend to be ready...")
    while not is_frontend_ready():
        time.sleep(1)
    print("Frontend is ready!")
    
    print("\nBoth services are running!")
    print("Backend: http://localhost:5001")
    print("Frontend: http://localhost:8501")
    
    try:
        # Keep the script running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        backend_process.terminate()
        frontend_process.terminate()
        backend_process.wait()
        frontend_process.wait()
        print("All processes terminated.")

if __name__ == "__main__":
    main() 