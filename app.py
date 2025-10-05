#!/usr/bin/env python3
"""
ExoVision - Hugging Face Spaces Entry Point
"""

import sys
import os

# Add the backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

# Import and run the main application
from agent import main
import asyncio

if __name__ == "__main__":
    asyncio.run(main())
