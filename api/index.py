import os
import sys

# Ensure the root project directory is on sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Also ensure current working directory is on sys.path
cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.insert(0, cwd)

from app import app

# Expose WSGI application for all Vercel Python runtime discovery mechanisms
app = app
application = app
handler = app
