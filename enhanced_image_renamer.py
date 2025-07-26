#!/usr/bin/env python3
"""
Enhanced AI Image Renamer
"""

from gui import GUI

if __name__ == "__main__":
    # Check dependencies
    try:
        import torch
        from transformers import BlipProcessor
        from PIL import Image
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install required packages:")
        print("pip install torch torchvision transformers pillow")
        exit(1)
    
    # Start the application
    app = GUI()
    app.run()