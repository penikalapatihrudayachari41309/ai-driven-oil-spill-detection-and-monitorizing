#!/usr/bin/env python3
"""
Run Streamlit app with ngrok for permanent public URL deployment.
Usage: python3 run_with_ngrok.py
"""
import os
import subprocess
import time
from pyngrok import ngrok

# Get ngrok public URL
print("Starting ngrok tunnel...")
public_url = ngrok.connect(addr="8501", proto="http", bind_tls=True)
print(f"\n✅ Public URL: {public_url}")

# Kill existing Streamlit processes
os.system("pkill -f streamlit 2>/dev/null")
time.sleep(1)

# Start Streamlit app
print("\nStarting Streamlit app...")
os.system("streamlit run app.py --server.enableCORS false --server.enableXsrfProtection false &")

print("\n" + "="*60)
print("Oil Spill Detection App is now accessible at:")
print(f"  {public_url}")
print("="*60)
print("\nPress Ctrl+C to stop the tunnel and app.")

