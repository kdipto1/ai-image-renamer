#!/usr/bin/env python3
"""
Enhanced AI Image Renamer - GUI
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
import threading
import torch

from renamer_logic import EnhancedImageRenamer, RenamePlan

class PreviewWindow(tk.Toplevel):
    def __init__(self, parent, rename_plan: RenamePlan, renamer: EnhancedImageRenamer, folder_path: str):
        super().__init__(parent)
        self.title("Preview Rename Plan")
        self.geometry("800x600")
        self.rename_plan = rename_plan
        self.renamer = renamer
        self.folder_path = folder_path

        self.create_widgets()

    def create_widgets(self):
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        self.tree = ttk.Treeview(main_frame, columns=("original", "new"), show="headings")
        self.tree.heading("original", text="Original Filename")
        self.tree.heading("new", text="New Filename")
        self.tree.pack(fill=tk.BOTH, expand=True)

        for op in self.rename_plan.operations:
            if op.status == "pending":
                self.tree.insert("", "end", values=(op.original_path, op.new_path))

        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)

        self.confirm_button = ttk.Button(button_frame, text="Confirm", command=self.confirm_renaming)
        self.confirm_button.pack(side=tk.RIGHT, padx=5)

        self.cancel_button = ttk.Button(button_frame, text="Cancel", command=self.destroy)
        self.cancel_button.pack(side=tk.RIGHT)

    def confirm_renaming(self):
        self.renamer.execute_rename_plan(self.rename_plan, self.folder_path)
        self.destroy()

class GUI:
    def __init__(self):
        self.renamer = EnhancedImageRenamer()
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the GUI interface"""
        self.root = tk.Tk()
        self.root.title("Enhanced AI Image Renamer")
        self.root.geometry("700x600")
        self.root.resizable(True, True)
        
        style = ttk.Style()
        style.theme_use('clam')

        # Create a Notebook (tabbed interface)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(pady=10, padx=10, fill="both", expand=True)

        # Create frames for each tab
        self.renamer_frame = ttk.Frame(self.notebook, padding="10")
        self.log_frame = ttk.Frame(self.notebook, padding="10")

        self.notebook.add(self.renamer_frame, text='Renamer')
        self.notebook.add(self.log_frame, text='Log')
        
        # --- Renamer Tab ---
        self.renamer_frame.columnconfigure(1, weight=1)
        
        title_label = ttk.Label(self.renamer_frame, text="Enhanced AI Image Renamer", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20), sticky="w")
        
        ttk.Label(self.renamer_frame, text="AI Model:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.model_var = tk.StringVar(value="vit")
        model_frame = ttk.Frame(self.renamer_frame)
        model_frame.grid(row=1, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Radiobutton(model_frame, text="ViT-GPT2 (Recommended)", 
                       variable=self.model_var, value="vit", command=self.toggle_api_key_entry).pack(side=tk.LEFT)
        ttk.Radiobutton(model_frame, text="BLIP-2 (More Accurate)", 
                       variable=self.model_var, value="blip2", command=self.toggle_api_key_entry).pack(side=tk.LEFT, padx=(20, 0))
        ttk.Radiobutton(model_frame, text="BLIP-1 (Faster)", 
                       variable=self.model_var, value="blip", command=self.toggle_api_key_entry).pack(side=tk.LEFT, padx=(20, 0))
        ttk.Radiobutton(model_frame, text="Google AI", 
                       variable=self.model_var, value="google", command=self.toggle_api_key_entry).pack(side=tk.LEFT, padx=(20, 0))

        self.api_key_label = ttk.Label(self.renamer_frame, text="Google AI API Key:")
        self.api_key_var = tk.StringVar()
        self.api_key_entry = ttk.Entry(self.renamer_frame, textvariable=self.api_key_var, show="*")

        device_info = f"Device: {self.renamer.device.upper()}"
        if self.renamer.device == "cuda":
            try:
                device_info += f" (GPU: {torch.cuda.get_device_name(0)})"
            except Exception as e:
                print(f"Could not get GPU name: {e}")
        self.device_label = ttk.Label(self.renamer_frame, text=device_info, font=('Arial', 9))
        self.device_label.grid(row=3, column=0, columnspan=3, pady=5, sticky="w")
        
        self.load_button = ttk.Button(self.renamer_frame, text="Load Model", command=self.load_model)
        self.load_button.grid(row=4, column=0, columnspan=3, pady=10)
        
        ttk.Label(self.renamer_frame, text="Image Folder:").grid(row=5, column=0, sticky=tk.W, pady=5)
        folder_frame = ttk.Frame(self.renamer_frame)
        folder_frame.grid(row=5, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        folder_frame.columnconfigure(0, weight=1)
        
        self.folder_var = tk.StringVar()
        ttk.Entry(folder_frame, textvariable=self.folder_var, state="readonly").grid(
            row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        ttk.Button(folder_frame, text="Browse", command=self.browse_folder).grid(row=0, column=1)
        
        self.process_button = ttk.Button(self.renamer_frame, text="Start Renaming", 
                                       command=self.start_processing, state="disabled")
        self.process_button.grid(row=6, column=0, columnspan=3, pady=20)
        
        progress_frame = ttk.LabelFrame(self.renamer_frame, text="Progress", padding="10")
        progress_frame.grid(row=7, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        progress_frame.columnconfigure(0, weight=1)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, 
                                          maximum=100, length=400)
        self.progress_bar.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(progress_frame, textvariable=self.status_var).grid(row=1, column=0, pady=2, sticky="w")
        
        self.current_file_var = tk.StringVar(value="")
        ttk.Label(progress_frame, textvariable=self.current_file_var, 
                 font=('Arial', 8)).grid(row=2, column=0, pady=2, sticky="w")

        self.cancel_button = ttk.Button(self.renamer_frame, text="Cancel", command=self.cancel_processing, state="disabled")
        self.cancel_button.grid(row=7, column=0, columnspan=2, pady=10)

        # --- Log Tab ---
        self.log_text = ScrolledText(self.log_frame, wrap=tk.WORD, state='disabled')
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        self.renamer.progress_var = self.progress_var
        self.renamer.status_var = self.status_var
        self.renamer.current_file_var = self.current_file_var
        self.renamer.log_callback = self.log_message

        self.toggle_api_key_entry() # Call to set initial state

    def toggle_api_key_entry(self):
        if self.model_var.get() == "google":
            self.api_key_label.grid(row=2, column=0, sticky=tk.W, pady=5)
            self.api_key_entry.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=5)
            self.device_label.grid_remove()
        else:
            self.api_key_label.grid_remove()
            self.api_key_entry.grid_remove()
            self.device_label.grid(row=3, column=0, columnspan=3, pady=5, sticky="w")

    def log_message(self, message):
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.config(state='disabled')
        self.log_text.see(tk.END)
        
    def load_model(self):
        self.load_button.config(state="disabled", text="Loading...")
        self.status_var.set("Loading model... This may take a few minutes.")
        self.root.update()
        
        def load_in_thread():
            model_type = self.model_var.get()
            api_key = self.api_key_var.get() if model_type == "google" else None
            success = self.renamer.load_model(model_type, api_key)
            self.root.after(0, lambda: self.on_model_loaded(success))
        
        threading.Thread(target=load_in_thread, daemon=True).start()
    
    def on_model_loaded(self, success):
        if success:
            self.load_button.config(state="normal", text="Model Loaded ✓")
            self.status_var.set(f"Model loaded successfully! Using {self.renamer.model_type.upper()}")
            if self.folder_var.get():
                self.process_button.config(state="normal")
        else:
            self.load_button.config(state="normal", text="Load Model")
            self.status_var.set("Failed to load model")
    
    def browse_folder(self):
        folder = filedialog.askdirectory(title="Select Image Folder")
        if folder:
            self.folder_var.set(folder)
            if self.renamer.model is not None:
                self.process_button.config(state="normal")
    
    def start_processing(self):
        if not self.folder_var.get():
            messagebox.showwarning("No Folder", "Please select a folder first.")
            return
        
        self.process_button.config(state="disabled", text="Processing...")
        self.cancel_button.config(state="normal")
        self.progress_var.set(0)
        
        def process_in_thread():
            rename_plan = self.renamer.generate_rename_plan(self.folder_var.get())
            self.root.after(0, lambda: self.on_plan_generated(rename_plan))
        
        threading.Thread(target=process_in_thread, daemon=True).start()

    def on_plan_generated(self, rename_plan):
        self.process_button.config(state="normal", text="Start Renaming")
        self.cancel_button.config(state="disabled")
        if rename_plan and rename_plan.operations:
            PreviewWindow(self.root, rename_plan, self.renamer, self.folder_var.get())
        else:
            self.status_var.set("No images to process or process was cancelled.")

    def cancel_processing(self):
        self.renamer.cancel_processing()
        self.status_var.set("Cancelling...")

    def on_processing_complete(self):
        self.process_button.config(state="normal", text="Start Renaming")
        self.progress_var.set(100)
        self.current_file_var.set("Processing complete!")
    
    def run(self):
        self.root.mainloop()
