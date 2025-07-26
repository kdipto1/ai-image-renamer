#!/usr/bin/env python3
"""
Enhanced AI Image Renamer
Optimized for better performance and accuracy with multiple model support
"""

import os
import re
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple, List, Dict

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import torch
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    Blip2Processor, Blip2ForConditionalGeneration,
    pipeline
)

class EnhancedImageRenamer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self.model_type = "blip"
        self.batch_size = 4 if self.device == "cuda" else 2
        self.max_workers = 3  # Conservative for your 8GB RAM
        
        # Supported formats
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
        
        # Progress tracking
        self.progress_var = None
        self.status_var = None
        self.current_file_var = None
        
        # Results tracking
        self.processed_count = 0
        self.failed_count = 0
        self.results_log = []

    def load_model(self, model_type: str = "blip2"):
        """Load the specified model with optimizations"""
        try:
            if model_type == "blip2":
                # BLIP-2 - More accurate but slower
                self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
                self.model = Blip2ForConditionalGeneration.from_pretrained(
                    "Salesforce/blip2-opt-2.7b",
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None
                ).to(self.device)
            else:
                # BLIP-1 - Faster but less accurate
                self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
                self.model = BlipForConditionalGeneration.from_pretrained(
                    "Salesforce/blip-image-captioning-large",
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                ).to(self.device)
            
            self.model_type = model_type
            
            # Enable optimizations
            if self.device == "cuda":
                self.model.half()  # Use FP16 for better performance
                torch.backends.cudnn.benchmark = True
                
            return True
            
        except Exception as e:
            messagebox.showerror("Model Error", f"Failed to load model: {str(e)}")
            return False

    def clean_filename(self, text: str, max_length: int = 80) -> str:
        """Clean and format text for use as filename"""
        # Remove unwanted patterns
        text = re.sub(r'^(a |an |the )', '', text.lower())
        text = re.sub(r'\b(image|photo|picture|shot|view) of\b', '', text)
        text = re.sub(r'\b(showing|depicting|featuring)\b', '', text)
        
        # Clean up the text
        text = re.sub(r'[^\w\s-]', '', text)  # Remove special chars except hyphens
        text = re.sub(r'\s+', '-', text.strip())  # Replace spaces with hyphens
        text = re.sub(r'-+', '-', text)  # Remove multiple consecutive hyphens
        text = text.strip('-')  # Remove leading/trailing hyphens
        
        # Truncate if too long
        if len(text) > max_length:
            text = text[:max_length].rsplit('-', 1)[0]
        
        return text.capitalize()

    def describe_image(self, image_path: str) -> Optional[str]:
        """Generate description for a single image with error handling"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            
            # Resize large images for faster processing
            if max(image.size) > 1024:
                image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            
            # Generate caption
            if self.model_type == "blip2":
                inputs = self.processor(image, return_tensors="pt").to(self.device)
                if self.device == "cuda":
                    inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
                
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_length=50,
                        num_beams=4,
                        do_sample=True,
                        temperature=0.7
                    )
                caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
            else:
                inputs = self.processor(image, return_tensors="pt").to(self.device)
                if self.device == "cuda":
                    inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
                
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_length=30,
                        num_beams=3
                    )
                caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
            
            return self.clean_filename(caption)
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

    def get_unique_filename(self, base_name: str, extension: str, folder_path: str, used_names: set) -> str:
        """Generate unique filename to avoid conflicts"""
        filename = f"{base_name}{extension}"
        counter = 1
        
        while filename in used_names or os.path.exists(os.path.join(folder_path, filename)):
            filename = f"{base_name}-{counter:02d}{extension}"
            counter += 1
            
        used_names.add(filename)
        return filename

    def process_images_batch(self, image_paths: List[str], folder_path: str, used_names: set) -> List[Tuple[str, str, bool]]:
        """Process a batch of images"""
        results = []
        
        for image_path in image_paths:
            try:
                filename = os.path.basename(image_path)
                if self.current_file_var:
                    self.current_file_var.set(f"Processing: {filename}")
                
                description = self.describe_image(image_path)
                
                if description:
                    # Get original extension
                    original_ext = Path(image_path).suffix.lower()
                    new_filename = self.get_unique_filename(description, original_ext, folder_path, used_names)
                    new_path = os.path.join(folder_path, new_filename)
                    
                    # Rename the file
                    os.rename(image_path, new_path)
                    results.append((filename, new_filename, True))
                    self.processed_count += 1
                    
                    self.results_log.append(f"✓ {filename} → {new_filename}")
                else:
                    results.append((filename, "Failed to generate description", False))
                    self.failed_count += 1
                    self.results_log.append(f"✗ {filename} - Failed to process")
                    
            except Exception as e:
                results.append((filename, f"Error: {str(e)}", False))
                self.failed_count += 1
                self.results_log.append(f"✗ {filename} - Error: {str(e)}")
        
        return results

    def rename_images(self, folder_path: str, progress_callback=None):
        """Main function to rename all images in folder with parallel processing"""
        try:
            # Find all image files
            image_files = []
            for file_path in Path(folder_path).iterdir():
                if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                    image_files.append(str(file_path))
            
            if not image_files:
                messagebox.showinfo("No Images", "No supported image files found in the selected folder.")
                return
            
            total_files = len(image_files)
            used_names = set()
            
            # Reset counters
            self.processed_count = 0
            self.failed_count = 0
            self.results_log = []
            
            if self.status_var:
                self.status_var.set(f"Found {total_files} images. Starting processing...")
            
            # Process images in batches with threading
            batch_size = self.batch_size
            batches = [image_files[i:i + batch_size] for i in range(0, len(image_files), batch_size)]
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_batch = {
                    executor.submit(self.process_images_batch, batch, folder_path, used_names): batch 
                    for batch in batches
                }
                
                completed_batches = 0
                for future in as_completed(future_to_batch):
                    batch_results = future.result()
                    completed_batches += 1
                    
                    # Update progress
                    progress = (completed_batches / len(batches)) * 100
                    if self.progress_var:
                        self.progress_var.set(progress)
                    if self.status_var:
                        self.status_var.set(f"Processed: {self.processed_count}, Failed: {self.failed_count}")
            
            # Show completion message
            completion_msg = f"""
Processing Complete!

Successfully renamed: {self.processed_count} files
Failed to process: {self.failed_count} files
Total processed: {self.processed_count + self.failed_count} files

Results saved to: renaming_log.txt
"""
            
            # Save detailed log
            log_path = os.path.join(folder_path, "renaming_log.txt")
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write(f"Image Renaming Results - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 60 + "\n\n")
                for log_entry in self.results_log:
                    f.write(log_entry + "\n")
            
            messagebox.showinfo("Complete", completion_msg)
            
        except Exception as e:
            messagebox.showerror("Processing Error", f"An error occurred: {str(e)}")

class GUI:
    def __init__(self):
        self.renamer = EnhancedImageRenamer()
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the GUI interface"""
        self.root = tk.Tk()
        self.root.title("Enhanced AI Image Renamer")
        self.root.geometry("600x500")
        self.root.resizable(True, True)
        
        # Style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Enhanced AI Image Renamer", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Model selection
        ttk.Label(main_frame, text="AI Model:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.model_var = tk.StringVar(value="blip2")
        model_frame = ttk.Frame(main_frame)
        model_frame.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Radiobutton(model_frame, text="BLIP-2 (More Accurate)", 
                       variable=self.model_var, value="blip2").pack(side=tk.LEFT)
        ttk.Radiobutton(model_frame, text="BLIP-1 (Faster)", 
                       variable=self.model_var, value="blip").pack(side=tk.LEFT, padx=(20, 0))
        
        # Device info
        device_info = f"Device: {self.renamer.device.upper()}"
        if self.renamer.device == "cuda":
            device_info += f" (GPU: {torch.cuda.get_device_name(0)})"
        ttk.Label(main_frame, text=device_info, font=('Arial', 9)).grid(
            row=2, column=0, columnspan=2, pady=5)
        
        # Load model button
        self.load_button = ttk.Button(main_frame, text="Load Model", command=self.load_model)
        self.load_button.grid(row=3, column=0, columnspan=2, pady=10)
        
        # Folder selection
        ttk.Label(main_frame, text="Image Folder:").grid(row=4, column=0, sticky=tk.W, pady=5)
        folder_frame = ttk.Frame(main_frame)
        folder_frame.grid(row=4, column=1, sticky=(tk.W, tk.E), pady=5)
        folder_frame.columnconfigure(0, weight=1)
        
        self.folder_var = tk.StringVar()
        ttk.Entry(folder_frame, textvariable=self.folder_var, state="readonly").grid(
            row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        ttk.Button(folder_frame, text="Browse", command=self.browse_folder).grid(row=0, column=1)
        
        # Process button
        self.process_button = ttk.Button(main_frame, text="Start Renaming", 
                                       command=self.start_processing, state="disabled")
        self.process_button.grid(row=5, column=0, columnspan=2, pady=20)
        
        # Progress section
        progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding="10")
        progress_frame.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        progress_frame.columnconfigure(0, weight=1)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, 
                                          maximum=100, length=400)
        self.progress_bar.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(progress_frame, textvariable=self.status_var).grid(row=1, column=0, pady=2)
        
        self.current_file_var = tk.StringVar(value="")
        ttk.Label(progress_frame, textvariable=self.current_file_var, 
                 font=('Arial', 8)).grid(row=2, column=0, pady=2)
        
        # Setup renamer progress tracking
        self.renamer.progress_var = self.progress_var
        self.renamer.status_var = self.status_var
        self.renamer.current_file_var = self.current_file_var
        
    def load_model(self):
        """Load the selected model"""
        self.load_button.config(state="disabled", text="Loading...")
        self.status_var.set("Loading model... This may take a few minutes.")
        self.root.update()
        
        def load_in_thread():
            success = self.renamer.load_model(self.model_var.get())
            self.root.after(0, lambda: self.on_model_loaded(success))
        
        threading.Thread(target=load_in_thread, daemon=True).start()
    
    def on_model_loaded(self, success):
        """Handle model loading completion"""
        if success:
            self.load_button.config(state="normal", text="Model Loaded ✓")
            self.status_var.set(f"Model loaded successfully! Using {self.renamer.model_type.upper()}")
            if self.folder_var.get():
                self.process_button.config(state="normal")
        else:
            self.load_button.config(state="normal", text="Load Model")
            self.status_var.set("Failed to load model")
    
    def browse_folder(self):
        """Browse for image folder"""
        folder = filedialog.askdirectory(title="Select Image Folder")
        if folder:
            self.folder_var.set(folder)
            if self.renamer.model is not None:
                self.process_button.config(state="normal")
    
    def start_processing(self):
        """Start the image processing in a separate thread"""
        if not self.folder_var.get():
            messagebox.showwarning("No Folder", "Please select a folder first.")
            return
        
        self.process_button.config(state="disabled", text="Processing...")
        self.progress_var.set(0)
        
        def process_in_thread():
            self.renamer.rename_images(self.folder_var.get())
            self.root.after(0, self.on_processing_complete)
        
        threading.Thread(target=process_in_thread, daemon=True).start()
    
    def on_processing_complete(self):
        """Handle processing completion"""
        self.process_button.config(state="normal", text="Start Renaming")
        self.progress_var.set(100)
        self.current_file_var.set("Processing complete!")
    
    def run(self):
        """Start the GUI"""
        self.root.mainloop()

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
