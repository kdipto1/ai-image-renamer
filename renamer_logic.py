#!/usr/bin/env python3
"""
Enhanced AI Image Renamer - Logic
"""

import os
import re
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple, List, Dict, NamedTuple, Callable
from tkinter import messagebox

import torch
from PIL import Image
from transformers import (
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer,
    BlipProcessor,
    BlipForConditionalGeneration,
    Blip2Processor,
    Blip2ForConditionalGeneration,
)

class RenameOperation(NamedTuple):
    original_path: str
    new_path: str
    status: str # "pending", "success", "failed"

class RenamePlan(NamedTuple):
    operations: List[RenameOperation]

class EnhancedImageRenamer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.model_type = "vit"
        self.batch_size = 4 if self.device == "cuda" else 2
        self.max_workers = 3
        self._is_cancelled = False

        self.supported_formats = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}

        self.progress_var = None
        self.status_var = None
        self.current_file_var = None
        self.log_callback: Optional[Callable[[str], None]] = None

    def load_model(self, model_type: str = "vit"):
        """Load the specified model with optimizations"""
        try:
            if model_type == "vit":
                model_name = "nlpconnect/vit-gpt2-image-captioning"
                self.processor = ViTImageProcessor.from_pretrained(model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = VisionEncoderDecoderModel.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                ).to(self.device)
            elif model_type == "blip2":
                processor_class = Blip2Processor
                model_class = Blip2ForConditionalGeneration
                pretrained_model = "Salesforce/blip2-opt-2.7b"
                self.processor = processor_class.from_pretrained(pretrained_model)
                self.model = model_class.from_pretrained(
                    pretrained_model,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None
                )
            else: # blip
                processor_class = BlipProcessor
                model_class = BlipForConditionalGeneration
                pretrained_model = "Salesforce/blip-image-captioning-large"
                self.processor = processor_class.from_pretrained(pretrained_model)
                self.model = model_class.from_pretrained(
                    pretrained_model,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None
                )

            if self.device == "cuda" and model_type != "vit":
                self.model.to(self.device)

            self.model_type = model_type

            if self.device == "cuda":
                self.model.half()
                torch.backends.cudnn.benchmark = True

            return True

        except Exception as e:
            if self.log_callback:
                self.log_callback(f"Error loading model: {e}")
            messagebox.showerror("Model Error", f"Failed to load model: {str(e)}")
            return False

    def _clean_filename(self, text: str, max_length: int = 80) -> str:
        """Clean and format text for use as filename"""
        text = re.sub(r'^(a |an |the )', '', text.lower())
        text = re.sub(r'\b(image|photo|picture|shot|view) of\b', '', text)
        text = re.sub(r'\b(showing|depicting|featuring)\b', '', text)
        text = re.sub(r'[^\w\s-]', '', text)
        text = re.sub(r'\s+', '-', text.strip())
        text = re.sub(r'-+', '-', text)
        text = text.strip('-')
        if len(text) > max_length:
            text = text[:max_length].rsplit('-', 1)[0]
        return text.capitalize()

    def _describe_image(self, image_path: str) -> Optional[str]:
        """Generate description for a single image with error handling"""
        try:
            image = Image.open(image_path).convert("RGB")
            if max(image.size) > 1024:
                image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)

            if self.model_type == "vit":
                pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)
                with torch.no_grad():
                    output_ids = self.model.generate(pixel_values, max_length=16, num_beams=4, return_dict_in_generate=True).sequences
                preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                caption = preds[0].strip()
            else: # blip or blip2
                if self.model_type == "blip2":
                    inputs = self.processor(image, return_tensors="pt").to(self.device, torch.float16 if self.device == 'cuda' else torch.float32)
                else:
                    inputs = self.processor(image, return_tensors="pt").to(self.device)
                
                if self.device == "cuda":
                    inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}

                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_length=50 if self.model_type == "blip2" else 30,
                        num_beams=4 if self.model_type == "blip2" else 3,
                        do_sample=True if self.model_type == "blip2" else False,
                        temperature=0.7 if self.model_type == "blip2" else None
                    )
                caption = self.processor.decode(generated_ids[0], skip_special_special_tokens=True)

            return self._clean_filename(caption)
        except Exception as e:
            if self.log_callback:
                self.log_callback(f"Error processing {image_path}: {e}")
            return None

    def _get_unique_filename(self, base_name: str, extension: str, folder_path: str, used_names: set) -> str:
        """Generate unique filename to avoid conflicts"""
        filename = f"{base_name}{extension}"
        counter = 1
        while filename in used_names or os.path.exists(os.path.join(folder_path, filename)):
            filename = f"{base_name}-{counter:02d}{extension}"
            counter += 1
        return filename

    def _process_batch(self, image_paths: List[str], folder_path: str, used_names: set) -> List[RenameOperation]:
        """Processes a batch of images and returns rename operations."""
        results = []
        for image_path in image_paths:
            if self._is_cancelled:
                break
            filename = os.path.basename(image_path)
            if self.current_file_var:
                self.current_file_var.set(f"Processing: {filename}")

            description = self._describe_image(image_path)

            if description:
                original_ext = Path(image_path).suffix.lower()
                new_filename_base = self._clean_filename(description)
                new_filename = self._get_unique_filename(new_filename_base, original_ext, folder_path, used_names)
                new_path = os.path.join(folder_path, new_filename)
                used_names.add(new_filename)
                results.append(RenameOperation(original_path=image_path, new_path=new_path, status="pending"))
            else:
                results.append(RenameOperation(original_path=image_path, new_path="", status="failed"))
        return results

    def generate_rename_plan(self, folder_path: str) -> RenamePlan:
        """Generate a plan for renaming images in a folder."""
        self._is_cancelled = False
        image_files = sorted([str(p) for p in Path(folder_path).iterdir() if p.is_file() and p.suffix.lower() in self.supported_formats])

        if not image_files:
            messagebox.showinfo("No Images", "No supported image files found in the selected folder.")
            return RenamePlan(operations=[])

        total_files = len(image_files)
        used_names = set(os.path.basename(p) for p in image_files)
        
        if self.status_var:
            self.status_var.set(f"Found {total_files} images. Generating descriptions...")

        batches = [image_files[i:i + self.batch_size] for i in range(0, len(image_files), self.batch_size)]
        
        operations = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_batch = {
                executor.submit(self._process_batch, batch, folder_path, used_names): batch
                for batch in batches
            }

            completed_batches = 0
            for future in as_completed(future_to_batch):
                if self._is_cancelled:
                    for f in future_to_batch:
                        f.cancel()
                    break
                
                batch_results = future.result()
                operations.extend(batch_results)
                completed_batches += 1
                
                progress = (completed_batches / len(batches)) * 100
                if self.progress_var:
                    self.progress_var.set(progress)
                if self.status_var:
                    processed_count = len(operations)
                    failed_count = len([op for op in operations if op.status == 'failed'])
                    self.status_var.set(f"Processed: {processed_count}/{total_files}, Failed: {failed_count}")

        return RenamePlan(operations=operations)

    def execute_rename_plan(self, plan: RenamePlan, folder_path: str):
        """Executes a rename plan, renaming files and logging results."""
        processed_count = 0
        failed_count = 0
        results_log = []

        for op in plan.operations:
            if op.status == 'pending':
                try:
                    os.rename(op.original_path, op.new_path)
                    processed_count += 1
                    log_entry = f"✓ {os.path.basename(op.original_path)} → {os.path.basename(op.new_path)}"
                    results_log.append(log_entry)
                    if self.log_callback:
                        self.log_callback(log_entry)
                except Exception as e:
                    failed_count += 1
                    log_entry = f"✗ {os.path.basename(op.original_path)} - Error: {e}"
                    results_log.append(log_entry)
                    if self.log_callback:
                        self.log_callback(log_entry)
            else:
                failed_count += 1
                log_entry = f"✗ {os.path.basename(op.original_path)} - Failed to generate description"
                results_log.append(log_entry)
                if self.log_callback:
                    self.log_callback(log_entry)

        completion_msg = f"""
Processing Complete!

Successfully renamed: {processed_count} files
Failed to process: {failed_count} files
Total processed: {processed_count + failed_count} files

Results saved to: renaming_log.txt
"""
        log_path = os.path.join(folder_path, "renaming_log.txt")
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write(f"Image Renaming Results - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 60 + "\n\n")
            for log_entry in results_log:
                f.write(log_entry + "\n")

        messagebox.showinfo("Complete", completion_msg)

    def cancel_processing(self):
        self._is_cancelled = True
