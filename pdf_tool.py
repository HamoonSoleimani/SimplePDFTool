#!/usr/bin/env python3

"""
Advanced PDF Toolkit - A modern GUI application for PDF operations
Features:
- Split PDFs by page ranges or into single pages
- Merge multiple PDFs into one
- Convert images to PDF
- Extract images from PDFs
- Clean, responsive interface with dark mode support

Dependencies:
- pypdf: For PDF manipulation
- Pillow: For image processing
- tkinter: For GUI
- ttkthemes: For enhanced theming (optional but recommended)

Install dependencies with:
pip install pypdf Pillow ttkthemes
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import sys
import threading
import queue
import re
from typing import List, Tuple, Optional, Dict, Any, Union
import tempfile
from datetime import datetime
import logging
import uuid

# --- Setup Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(tempfile.gettempdir(), "pdf_toolkit.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PDFToolkit")

# --- Import Required Libraries ---
MISSING_DEPENDENCIES = []

try:
    from pypdf import PdfReader, PdfWriter, PageRange
    from pypdf.errors import PdfReadError, DependencyError
except ImportError:
    MISSING_DEPENDENCIES.append("pypdf (pip install pypdf)")

try:
    from PIL import Image, UnidentifiedImageError
    import PIL.ImageOps
except ImportError:
    MISSING_DEPENDENCIES.append("Pillow (pip install Pillow)")

# Optional theme library
try:
    from ttkthemes import ThemedTk
    THEMED_TK_AVAILABLE = True
except ImportError:
    THEMED_TK_AVAILABLE = False

# --- Constants and Configuration ---
DEFAULT_THEME = "equilux"  # Dark theme
LIGHT_THEMES = ["clearlooks", "arc", "plastik", "winxpblue"]
DARK_THEMES = ["equilux", "black", "yaru", "breeze-dark"]
SUPPORTED_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".gif"]
ICON_BASE64 = """
    <base64 icon data would go here - removed for brevity>
"""

# --- Core PDF Processing Logic ---

def parse_page_ranges(ranges_str: str, num_pages: int) -> List[Tuple[int, int]]:
    """
    Parses a comma-separated string of page ranges (1-based index) into a list of
    tuples representing (start_page_index, end_page_index) (0-based, inclusive).
    """
    if not ranges_str:
        raise ValueError("Page range string cannot be empty.")

    parsed_ranges = []
    parts = ranges_str.split(',')

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if '-' in part:
            start_str, end_str = part.split('-', 1)
            start_str = start_str.strip()
            end_str = end_str.strip()

            try:
                if not start_str: start_page = 1
                else:
                    start_page = int(start_str)
                    if start_page < 1: raise ValueError(f"Start page cannot be less than 1: '{start_str}'")

                if not end_str: end_page = num_pages
                else:
                    end_page = int(end_str)
                    if end_page < 1: raise ValueError(f"End page cannot be less than 1: '{end_str}'")

                if start_page > end_page:
                    raise ValueError(f"Start page ({start_page}) cannot be greater than end page ({end_page}) in range '{part}'")
                if start_page > num_pages:
                     raise ValueError(f"Start page ({start_page}) exceeds total pages ({num_pages}) in range '{part}'")
                
                effective_end_page = min(end_page, num_pages)
                # Convert to 0-based inclusive indices
                parsed_ranges.append((start_page - 1, effective_end_page - 1))

            except ValueError as e:
                if "invalid literal for int()" in str(e):
                    raise ValueError(f"Invalid number in page range '{part}'.")
                raise
        else:
            try:
                page = int(part)
                if page < 1 or page > num_pages:
                    raise ValueError(f"Page number {page} is out of valid range (1-{num_pages}).")
                parsed_ranges.append((page - 1, page - 1))
            except ValueError as e:
                if "invalid literal for int()" in str(e):
                     raise ValueError(f"Invalid page number '{part}'. Must be an integer or a range (e.g., '1-5').")
                raise

    # Sort and merge overlapping/adjacent ranges
    if not parsed_ranges: return []
    parsed_ranges.sort(key=lambda x: x[0])
    merged = []
    
    current_start, current_end = parsed_ranges[0]
    if current_start <= current_end:
        merged = [(current_start, current_end)]

    for next_start, next_end in parsed_ranges[1:]:
         if next_start > next_end: continue  # Skip invalid ranges
         last_start, last_end = merged[-1]
         if next_start <= last_end + 1:  # Overlap or adjacent
             merged[-1] = (last_start, max(last_end, next_end))
         else:
             merged.append((next_start, next_end))

    # Final validation: Ensure all indices are within 0 to num_pages-1
    validated_merged = []
    for start, end in merged:
        valid_start = max(0, start)
        valid_end = min(num_pages - 1, end)
        if valid_start <= valid_end:
            validated_merged.append((valid_start, valid_end))

    return validated_merged


def perform_split(input_path: str, output_dir: str, ranges_str: Optional[str],
                  single_pages: bool, output_prefix: Optional[str],
                  progress_queue: queue.Queue, status_queue: queue.Queue):
    """Performs the PDF splitting in a separate thread."""
    try:
        status_queue.put(("info", f"Reading '{os.path.basename(input_path)}'..."))
        reader = PdfReader(input_path)
        num_pages = len(reader.pages)

        if num_pages == 0:
            status_queue.put(("warning", f"Input PDF '{os.path.basename(input_path)}' has no pages. Nothing to split."))
            progress_queue.put(("done", 1, 1))
            return

        if not os.path.isdir(output_dir):
             status_queue.put(("info", f"Creating output directory: {output_dir}"))
             try:
                 os.makedirs(output_dir, exist_ok=True)
             except OSError as e:
                 raise OSError(f"Could not create output directory '{output_dir}': {e}")

        base_name = os.path.splitext(os.path.basename(input_path))[0]
        prefix = output_prefix if output_prefix else base_name + "_"

        split_instructions = []  # List of (output_filepath, list_of_page_indices)

        if single_pages:
            status_queue.put(("info", "Preparing to split into single pages..."))
            pad_width = len(str(num_pages))
            for i in range(num_pages):
                page_num_str = str(i + 1).zfill(pad_width)
                output_filename = f"{prefix}page_{page_num_str}.pdf"
                output_filepath = os.path.join(output_dir, output_filename)
                split_instructions.append((output_filepath, [i]))
        elif ranges_str:
            status_queue.put(("info", f"Parsing ranges: '{ranges_str}'..."))
            page_indices_ranges = parse_page_ranges(ranges_str, num_pages)

            if not page_indices_ranges:
                status_queue.put(("warning", "No valid page ranges found after parsing. Nothing to split."))
                progress_queue.put(("done", 1, 1))
                return

            status_queue.put(("info", f"Preparing {len(page_indices_ranges)} output file(s) based on ranges..."))
            for i, (start_index, end_index) in enumerate(page_indices_ranges):
                range_desc = f"{start_index + 1}"
                if start_index != end_index:
                    range_desc += f"-{end_index + 1}"
                output_filename = f"{prefix}pages_{range_desc}.pdf"
                output_filepath = os.path.join(output_dir, output_filename)
                pages_in_range = list(range(start_index, end_index + 1))
                split_instructions.append((output_filepath, pages_in_range))
        else:
             raise ValueError("Internal Error: No split method specified.")

        total_splits = len(split_instructions)
        if total_splits == 0:
            status_queue.put(("warning", "No output files to generate based on instructions."))
            progress_queue.put(("done", 1, 1))
            return

        status_queue.put(("info", f"Generating {total_splits} output file(s)..."))
        progress_queue.put(("start", 0, total_splits))

        for i, (output_filepath, page_indices) in enumerate(split_instructions):
            if not page_indices:
                status_queue.put(("warning", f"Skipping empty page range for '{os.path.basename(output_filepath)}'"))
                progress_queue.put(("update", i + 1, total_splits))
                continue

            current_pages_str = f"{min(page_indices)+1}-{max(page_indices)+1}" if len(page_indices) > 1 else f"{page_indices[0]+1}"
            status_queue.put(("info", f"  [{i+1}/{total_splits}] Creating '{os.path.basename(output_filepath)}' (Pages {current_pages_str})..."))

            writer = PdfWriter()
            pages_added = 0
            for page_index in page_indices:
                if 0 <= page_index < num_pages:
                    try:
                        writer.add_page(reader.pages[page_index])
                        pages_added += 1
                    except Exception as page_err:
                         status_queue.put(("warning", f"Error adding page {page_index+1}: {page_err}"))

            if pages_added > 0:
                 with open(output_filepath, "wb") as output_pdf:
                     writer.write(output_pdf)
            else:
                 status_queue.put(("warning", f"No valid pages were added for '{os.path.basename(output_filepath)}'. File not created."))

            progress_queue.put(("update", i + 1, total_splits))

        status_queue.put(("success", f"Splitting completed! Output files are in '{output_dir}'"))
        progress_queue.put(("done", total_splits, total_splits))

    except PdfReadError as e:
        status_queue.put(("error", f"Failed to read PDF '{os.path.basename(input_path)}'. It might be corrupted or password-protected."))
        logger.error(f"PDF read error: {e}")
        progress_queue.put(("error",))
    except DependencyError as e:
        status_queue.put(("error", f"Missing dependency: {e}"))
        progress_queue.put(("error",))
    except ValueError as e:
        status_queue.put(("error", f"Invalid input: {e}"))
        progress_queue.put(("error",))
    except OSError as e:
        status_queue.put(("error", f"File system operation failed: {e}"))
        progress_queue.put(("error",))
    except Exception as e:
        status_queue.put(("error", f"An unexpected error occurred: {e}"))
        logger.exception("Unexpected error in split operation")
        progress_queue.put(("error",))


def perform_merge(input_paths: List[str], output_path: str,
                  progress_queue: queue.Queue, status_queue: queue.Queue):
    """Performs the PDF merging in a separate thread."""
    merger = PdfWriter()
    total_input_files = len(input_paths)
    files_processed = 0
    total_pages_merged = 0

    try:
        status_queue.put(("info", f"Starting merge of {total_input_files} file(s)..."))
        progress_queue.put(("start", 0, total_input_files + 1))

        for i, input_path in enumerate(input_paths):
            status_queue.put(("info", f"  [{i+1}/{total_input_files}] Processing '{os.path.basename(input_path)}'..."))
            if not os.path.exists(input_path):
                status_queue.put(("warning", f"File not found, skipping: '{os.path.basename(input_path)}'"))
                progress_queue.put(("update", i + 1, total_input_files + 1))
                continue

            try:
                reader = PdfReader(input_path)
                num_pages_in_file = len(reader.pages)

                if num_pages_in_file > 0:
                    merger.append(reader)
                    status_queue.put(("info", f"    Appended {num_pages_in_file} pages"))
                    total_pages_merged += num_pages_in_file
                    files_processed += 1
                else:
                    status_queue.put(("warning", f"'{os.path.basename(input_path)}' contains no pages. Skipping."))

            except PdfReadError as e:
                status_queue.put(("warning", f"Error reading '{os.path.basename(input_path)}': {e}. Skipping."))
            except DependencyError as e:
                 status_queue.put(("warning", f"Missing dependency for '{os.path.basename(input_path)}': {e}. Skipping."))
            except Exception as e:
                 status_queue.put(("warning", f"Error processing '{os.path.basename(input_path)}': {e}. Skipping."))

            progress_queue.put(("update", i + 1, total_input_files + 1))

        if files_processed == 0:
            status_queue.put(("error", "No valid input PDF files could be processed. Output file not created."))
            progress_queue.put(("error",))
            merger.close()
            return

        status_queue.put(("info", f"Writing final merged PDF ({total_pages_merged} pages from {files_processed} files)..."))
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
             status_queue.put(("info", f"Creating output directory: {output_dir}"))
             os.makedirs(output_dir, exist_ok=True)

        with open(output_path, "wb") as output_pdf:
            merger.write(output_pdf)

        status_queue.put(("success", f"Merging completed! Output saved as '{output_path}'"))
        progress_queue.put(("done", total_input_files + 1, total_input_files + 1))

    except OSError as e:
        status_queue.put(("error", f"File system operation failed: {e}"))
        progress_queue.put(("error",))
    except Exception as e:
        status_queue.put(("error", f"An unexpected error occurred: {e}"))
        logger.exception("Unexpected error in merge operation")
        progress_queue.put(("error",))
    finally:
        merger.close()


def convert_images_to_pdf(image_paths: List[str], output_path: str, 
                          dpi: int, page_size: str, margin: int,
                          progress_queue: queue.Queue, status_queue: queue.Queue):
    """Converts a list of images to a single PDF file."""
    try:
        total_images = len(image_paths)
        status_queue.put(("info", f"Starting conversion of {total_images} image(s) to PDF..."))
        progress_queue.put(("start", 0, total_images + 1))  # +1 for final saving step
        
        # Standard page sizes (width, height) in pixels at specified DPI
        page_sizes = {
            "A4": (int(8.27 * dpi), int(11.69 * dpi)),
            "Letter": (int(8.5 * dpi), int(11 * dpi)),
            "Legal": (int(8.5 * dpi), int(14 * dpi)),
            "A3": (int(11.7 * dpi), int(16.5 * dpi))
        }
        
        # If page_size is "auto", we'll determine size from first image
        target_width, target_height = page_sizes.get(page_size, (None, None))
        
        # List to store processed images
        processed_images = []
        
        for i, img_path in enumerate(image_paths):
            try:
                status_queue.put(("info", f"  [{i+1}/{total_images}] Processing '{os.path.basename(img_path)}'..."))
                
                # Open and process image
                with Image.open(img_path) as img:
                    # Convert to RGB mode if necessary (required for PDF)
                    if img.mode not in ('RGB', 'L'):
                        img = img.convert('RGB')
                    
                    # Auto page size from first image
                    if page_size == "auto" and i == 0:
                        target_width, target_height = img.width, img.height
                        status_queue.put(("info", f"    Using automatic page size: {target_width}x{target_height} pixels"))
                    
                    # Resize and position image on page
                    if page_size != "auto":
                        # Create a white background image of the target size
                        bg = Image.new('RGB', (target_width, target_height), (255, 255, 255))
                        
                        # Calculate resize dimensions while maintaining aspect ratio
                        img_ratio = img.width / img.height
                        max_width = target_width - 2 * margin
                        max_height = target_height - 2 * margin
                        
                        if img.width > max_width or img.height > max_height:
                            if img.width / max_width > img.height / max_height:
                                # Width is the limiting factor
                                new_width = max_width
                                new_height = int(new_width / img_ratio)
                            else:
                                # Height is the limiting factor
                                new_height = max_height
                                new_width = int(new_height * img_ratio)
                            
                            img = img.resize((new_width, new_height), Image.LANCZOS)
                        
                        # Calculate position to center the image
                        x = (target_width - img.width) // 2
                        y = (target_height - img.height) // 2
                        
                        # Paste image onto the background
                        bg.paste(img, (x, y))
                        processed_images.append(bg)
                    else:
                        # For auto size, use original image
                        processed_images.append(img.copy())
                
            except UnidentifiedImageError:
                status_queue.put(("warning", f"Could not identify image '{os.path.basename(img_path)}'. Skipping."))
            except Exception as e:
                status_queue.put(("warning", f"Error processing image '{os.path.basename(img_path)}': {e}. Skipping."))
            
            progress_queue.put(("update", i + 1, total_images + 1))
        
        if not processed_images:
            status_queue.put(("error", "No valid images could be processed. PDF not created."))
            progress_queue.put(("error",))
            return
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            status_queue.put(("info", f"Creating output directory: {output_dir}"))
            os.makedirs(output_dir, exist_ok=True)
        
        # Save images as PDF
        status_queue.put(("info", f"Saving PDF with {len(processed_images)} image(s)..."))
        
        # Save first image and append the rest
        first_img = processed_images[0]
        if len(processed_images) == 1:
            first_img.save(output_path, "PDF", resolution=dpi)
        else:
            first_img.save(output_path, "PDF", resolution=dpi, 
                           save_all=True, append_images=processed_images[1:])
        
        status_queue.put(("success", f"Image to PDF conversion completed! Output saved as '{output_path}'"))
        progress_queue.put(("done", total_images + 1, total_images + 1))
        
    except Exception as e:
        status_queue.put(("error", f"An unexpected error occurred: {e}"))
        logger.exception("Unexpected error in image-to-PDF conversion")
        progress_queue.put(("error",))


def extract_images_from_pdf(input_path: str, output_dir: str, 
                            format: str, prefix: Optional[str],
                            progress_queue: queue.Queue, status_queue: queue.Queue):
    """Extracts images from a PDF file."""
    try:
        # Create output directory if it doesn't exist
        if not os.path.isdir(output_dir):
            status_queue.put(("info", f"Creating output directory: {output_dir}"))
            os.makedirs(output_dir, exist_ok=True)
        
        # Initialize reader
        status_queue.put(("info", f"Reading '{os.path.basename(input_path)}'..."))
        reader = PdfReader(input_path)
        num_pages = len(reader.pages)
        
        if num_pages == 0:
            status_queue.put(("warning", f"Input PDF '{os.path.basename(input_path)}' has no pages."))
            progress_queue.put(("done", 1, 1))
            return
        
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        prefix = prefix if prefix else base_name + "_"
        
        # Check if format is valid
        format = format.lower().strip('.')
        if format not in ['jpg', 'jpeg', 'png', 'tiff', 'bmp']:
            format = 'png'  # Default to PNG
        
        status_queue.put(("info", f"Extracting images from {num_pages} pages..."))
        progress_queue.put(("start", 0, num_pages))
        
        # We'll use a temporary directory for the conversion process
        with tempfile.TemporaryDirectory() as temp_dir:
            # First approach: Convert each page to an image
            total_images = 0
            pad_width = len(str(num_pages))
            
            for i in range(num_pages):
                try:
                    status_queue.put(("info", f"  [{i+1}/{num_pages}] Converting page {i+1} to image..."))
                    
                    # Save page as image
                    page_num_str = str(i + 1).zfill(pad_width)
                    output_filename = f"{prefix}page_{page_num_str}.{format}"
                    output_filepath = os.path.join(output_dir, output_filename)
                    
                    # Use a PDF rendering library to convert page to image
                    # Since pypdf doesn't directly support rendering, we need to use an external tool
                    # For simplicity, we'll use Pillow's PDF support
                    try:
                        # Note: This is a simplified approach, better results can be achieved with libraries like pdf2image
                        # which uses Poppler/GhostScript under the hood
                        with Image.open(input_path) as pdf:
                            if hasattr(pdf, 'seek') and callable(pdf.seek):
                                pdf.seek(i)  # Go to specific page
                                img = pdf.copy()
                                img.save(output_filepath)
                                total_images += 1
                                status_queue.put(("info", f"    Saved: {output_filename}"))
                    except Exception as e:
                        status_queue.put(("warning", f"    Could not convert page {i+1} to image: {e}"))
                
                except Exception as page_err:
                    status_queue.put(("warning", f"    Error processing page {i+1}: {page_err}"))
                
                progress_queue.put(("update", i + 1, num_pages))
            
            if total_images == 0:
                # If direct rendering failed, inform the user
                status_queue.put(("warning", "Could not render PDF pages as images directly."))
                status_queue.put(("info", "Consider using specialized tools like pdf2image for better results."))
            
            status_queue.put(("success", f"Extracted {total_images} images to '{output_dir}'"))
            progress_queue.put(("done", num_pages, num_pages))
    
    except PdfReadError as e:
        status_queue.put(("error", f"Failed to read PDF '{os.path.basename(input_path)}'. It might be corrupted or password-protected."))
        logger.error(f"PDF read error: {e}")
        progress_queue.put(("error",))
    except Exception as e:
        status_queue.put(("error", f"An unexpected error occurred: {e}"))
        logger.exception("Unexpected error in PDF-to-image extraction")
        progress_queue.put(("error",))


# --- GUI Application Class ---

class AdvancedPdfToolkit:
    def __init__(self, root):
        self.root = root
        self.setup_window()
        self.create_variables()
        self.create_styles()
        self.create_main_layout()
        self.create_pages()
        self.create_status_bar()
        self.bind_events()
        
        # Start the queue checker
        self.after_id = None
        self.check_queues()
        
        # Apply theme
        self.apply_theme(DEFAULT_THEME)
        
        # Show the splash screen on startup
        # self.show_splash_screen()

    def setup_window(self):
        """Configure the main window."""
        self.root.title("Advanced PDF Toolkit")
        self.root.minsize(800, 600)
        
        # Set initial window size to 80% of screen
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = int(screen_width * 0.8)
        window_height = int(screen_height * 0.8)
        self.root.geometry(f"{window_width}x{window_height}")
        
        # Center the window
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.root.geometry(f"+{x}+{y}")
        
        # Set window icon
        # if ICON_BASE64:
        #     icon_data = base64.b64decode(ICON_BASE64)
        #     icon_image = tk.PhotoImage(data=icon_data)
        #     self.root.iconphoto(True, icon_image)

    def create_variables(self):
        """Initialize all variables used in the application."""
        # Queues for thread communication
        self.progress_queue = queue.Queue()
        self.status_queue = queue.Queue()
        
        # Common variables
        self.current_theme = tk.StringVar(value=DEFAULT_THEME)
        self.dark_mode = tk.BooleanVar(value=True)
        
        # Split PDF variables
        self.split_input_path = tk.StringVar()
        self.split_output_dir = tk.StringVar()
        self.split_method = tk.StringVar(value="ranges")
        self.split_ranges = tk.StringVar()
        self.split_prefix = tk.StringVar()
        
        # Merge PDF variables
        self.merge_input_paths = []
        self.merge_output_path = tk.StringVar()
        
        # Images to PDF variables
        self.img2pdf_input_paths = []
        self.img2pdf_output_path = tk.StringVar()
        self.img2pdf_dpi = tk.IntVar(value=300)
        self.img2pdf_page_size = tk.StringVar(value="A4")
        self.img2pdf_margin = tk.IntVar(value=50)
        
        # PDF to Images variables
        self.pdf2img_input_path = tk.StringVar()
        self.pdf2img_output_dir = tk.StringVar()
        self.pdf2img_format = tk.StringVar(value="png")
        self.pdf2img_prefix = tk.StringVar()
        
        # Operation tracking
        self.current_operation = None
        self.worker_thread = None
        self.operation_history = []

    def create_styles(self):
        """Create and configure ttk styles."""
        self.style = ttk.Style()
        
        # Define custom styles
        self.style.configure('Title.TLabel', font=('Helvetica', 16, 'bold'))
        self.style.configure('Subtitle.TLabel', font=('Helvetica', 12))
        self.style.configure('Success.TLabel', foreground='green')
        self.style.configure('Error.TLabel', foreground='red')
        self.style.configure('Warning.TLabel', foreground='orange')
        
        # Button styles
        self.style.configure('Primary.TButton', font=('Helvetica', 11, 'bold'))
        self.style.configure('Secondary.TButton', font=('Helvetica', 10))

    def create_main_layout(self):
        """Create the main layout of the application."""
        # Main container
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(expand=True, fill='both', padx=10, pady=10)
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(1, weight=1)
        
        # Header with tools
        self.header_frame = ttk.Frame(self.main_frame)
        self.header_frame.grid(row=0, column=0, sticky='ew', pady=(0, 10))
        self.header_frame.columnconfigure(0, weight=1)
        
        # App title
        title_frame = ttk.Frame(self.header_frame)
        title_frame.pack(side=tk.LEFT)
        ttk.Label(title_frame, text="Advanced PDF Toolkit", style='Title.TLabel').pack(side=tk.LEFT)
        
        # Theme selector in header
        theme_frame = ttk.Frame(self.header_frame)
        theme_frame.pack(side=tk.RIGHT, padx=5)
        
        ttk.Label(theme_frame, text="Theme:").pack(side=tk.LEFT, padx=(0, 5))
        theme_combobox = ttk.Combobox(theme_frame, textvariable=self.current_theme, 
                               values=DARK_THEMES + LIGHT_THEMES, width=12, state="readonly")
        theme_combobox.pack(side=tk.LEFT)
        theme_combobox.bind("<<ComboboxSelected>>", lambda e: self.apply_theme(self.current_theme.get()))
        
        ttk.Checkbutton(theme_frame, text="Dark Mode", variable=self.dark_mode, 
                      command=self.toggle_dark_mode).pack(side=tk.LEFT, padx=10)
        
        # Notebook for different operations
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.grid(row=1, column=0, sticky='nsew')
        
        # Create frames for each operation
        self.split_frame = ttk.Frame(self.notebook, padding=10)
        self.merge_frame = ttk.Frame(self.notebook, padding=10)
        self.img2pdf_frame = ttk.Frame(self.notebook, padding=10)
        self.pdf2img_frame = ttk.Frame(self.notebook, padding=10)
        
        # Add frames to notebook
        self.notebook.add(self.split_frame, text='Split PDF')
        self.notebook.add(self.merge_frame, text='Merge PDFs')
        self.notebook.add(self.img2pdf_frame, text='Images to PDF')
        self.notebook.add(self.pdf2img_frame, text='PDF to Images')

    def create_pages(self):
        """Create content for each notebook page."""
        self.create_split_page()
        self.create_merge_page()
        self.create_img2pdf_page()
        self.create_pdf2img_page()

    def create_split_page(self):
        """Create the Split PDF page."""
        frame = self.split_frame
        frame.columnconfigure(1, weight=1)
        
        # Section title
        ttk.Label(frame, text="Split PDF into Multiple Files", style='Title.TLabel').grid(
            row=0, column=0, columnspan=3, pady=(0, 15), sticky='w')
        
        # Input File
        ttk.Label(frame, text="Input PDF:").grid(row=1, column=0, padx=5, pady=5, sticky='w')
        ttk.Entry(frame, textvariable=self.split_input_path).grid(row=1, column=1, padx=5, pady=5, sticky='ew')
        ttk.Button(frame, text="Browse...", command=self.browse_split_input, style='Secondary.TButton').grid(
            row=1, column=2, padx=5, pady=5)
        
        # Output Directory
        ttk.Label(frame, text="Output Directory:").grid(row=2, column=0, padx=5, pady=5, sticky='w')
        ttk.Entry(frame, textvariable=self.split_output_dir).grid(row=2, column=1, padx=5, pady=5, sticky='ew')
        ttk.Button(frame, text="Browse...", command=self.browse_split_output, style='Secondary.TButton').grid(
            row=2, column=2, padx=5, pady=5)
        
        # Split Method Selection
        method_frame = ttk.LabelFrame(frame, text="Split Method", padding="10")
        method_frame.grid(row=3, column=0, columnspan=3, padx=5, pady=15, sticky='ew')
        method_frame.columnconfigure(0, weight=1)
        
        self.radio_ranges = ttk.Radiobutton(method_frame, text="Split by Page Ranges", variable=self.split_method,
                                          value="ranges", command=self.update_split_ui_state)
        self.radio_ranges.grid(row=0, column=0, padx=5, pady=5, sticky='w')
        
        # Help text for ranges
        help_frame = ttk.Frame(method_frame)
        help_frame.grid(row=0, column=1, padx=5, pady=5, sticky='w')
        ttk.Label(help_frame, text="Example: 1-3, 5, 8-10", font=('Helvetica', 9, 'italic'), 
                foreground='#555555').pack(side=tk.LEFT)
        
        self.radio_single = ttk.Radiobutton(method_frame, text="Split into Single Pages", variable=self.split_method,
                                          value="single", command=self.update_split_ui_state)
        self.radio_single.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky='w')
        
        # Page Ranges Input
        range_frame = ttk.Frame(frame)
        range_frame.grid(row=4, column=0, columnspan=3, padx=5, pady=5, sticky='ew')
        range_frame.columnconfigure(1, weight=1)
        
        ttk.Label(range_frame, text="Page Ranges:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.ranges_entry = ttk.Entry(range_frame, textvariable=self.split_ranges)
        self.ranges_entry.grid(row=0, column=1, padx=5, pady=5, sticky='ew')
        
        # Output Prefix
        prefix_frame = ttk.Frame(frame)
        prefix_frame.grid(row=5, column=0, columnspan=3, padx=5, pady=5, sticky='ew')
        prefix_frame.columnconfigure(1, weight=1)
        
        ttk.Label(prefix_frame, text="Output Prefix:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.prefix_entry = ttk.Entry(prefix_frame, textvariable=self.split_prefix)
        self.prefix_entry.grid(row=0, column=1, padx=5, pady=5, sticky='ew')
        ttk.Label(prefix_frame, text="(Optional)").grid(row=0, column=2, padx=5, pady=5, sticky='w')
        
        # Action Button
        button_frame = ttk.Frame(frame)
        button_frame.grid(row=6, column=0, columnspan=3, pady=20)
        
        self.split_button = ttk.Button(button_frame, text="Split PDF", 
                                     command=self.start_split_thread, style='Primary.TButton', width=20)
        self.split_button.pack(pady=10)
        
        # Initial UI state
        self.update_split_ui_state()

    def create_merge_page(self):
        """Create the Merge PDFs page."""
        frame = self.merge_frame
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)
        
        # Section title
        ttk.Label(frame, text="Merge Multiple PDFs into One", style='Title.TLabel').grid(
            row=0, column=0, columnspan=2, pady=(0, 15), sticky='w')
        
        # Files list section
        files_frame = ttk.LabelFrame(frame, text="Input PDFs (in order)", padding="10")
        files_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky='nsew')
        files_frame.columnconfigure(0, weight=1)
        files_frame.rowconfigure(0, weight=1)
        
        # Listbox with files
        self.merge_listbox = tk.Listbox(files_frame, selectmode=tk.EXTENDED, height=10)
        self.merge_listbox.grid(row=0, column=0, padx=5, pady=5, sticky='nsew')
        
        # Scrollbar for listbox
        listbox_scrollbar = ttk.Scrollbar(files_frame, orient=tk.VERTICAL, command=self.merge_listbox.yview)
        listbox_scrollbar.grid(row=0, column=1, sticky='ns')
        self.merge_listbox.configure(yscrollcommand=listbox_scrollbar.set)
        
        # Buttons for list management
        button_frame = ttk.Frame(files_frame)
        button_frame.grid(row=1, column=0, columnspan=2, pady=10)
        
        ttk.Button(button_frame, text="Add Files...", command=self.add_merge_files, style='Secondary.TButton').pack(
            side=tk.LEFT, padx=3)
        ttk.Button(button_frame, text="Remove Selected", command=self.remove_merge_files, style='Secondary.TButton').pack(
            side=tk.LEFT, padx=3)
        ttk.Button(button_frame, text="Clear List", command=self.clear_merge_list, style='Secondary.TButton').pack(
            side=tk.LEFT, padx=3)
        ttk.Button(button_frame, text="Move Up", command=lambda: self.move_merge_item(-1), style='Secondary.TButton').pack(
            side=tk.LEFT, padx=3)
        ttk.Button(button_frame, text="Move Down", command=lambda: self.move_merge_item(1), style='Secondary.TButton').pack(
            side=tk.LEFT, padx=3)
        
        # Output file
        output_frame = ttk.Frame(frame)
        output_frame.grid(row=2, column=0, columnspan=2, padx=5, pady=15, sticky='ew')
        output_frame.columnconfigure(1, weight=1)
        
        ttk.Label(output_frame, text="Output File:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        ttk.Entry(output_frame, textvariable=self.merge_output_path).grid(row=0, column=1, padx=5, pady=5, sticky='ew')
        ttk.Button(output_frame, text="Save As...", command=self.browse_merge_output, style='Secondary.TButton').grid(
            row=0, column=2, padx=5, pady=5)
        
        # Action Button
        button_frame = ttk.Frame(frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=10)
        
        self.merge_button = ttk.Button(button_frame, text="Merge PDFs", 
                                      command=self.start_merge_thread, style='Primary.TButton', width=20)
        self.merge_button.pack(pady=10)

    def create_img2pdf_page(self):
        """Create the Images to PDF page."""
        frame = self.img2pdf_frame
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)
        
        # Section title
        ttk.Label(frame, text="Convert Images to PDF", style='Title.TLabel').grid(
            row=0, column=0, columnspan=2, pady=(0, 15), sticky='w')
        
        # Files list section
        files_frame = ttk.LabelFrame(frame, text="Input Images (in order)", padding="10")
        files_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky='nsew')
        files_frame.columnconfigure(0, weight=1)
        files_frame.rowconfigure(0, weight=1)
        
        # Listbox with files
        self.img2pdf_listbox = tk.Listbox(files_frame, selectmode=tk.EXTENDED, height=10)
        self.img2pdf_listbox.grid(row=0, column=0, padx=5, pady=5, sticky='nsew')
        
        # Scrollbar for listbox
        listbox_scrollbar = ttk.Scrollbar(files_frame, orient=tk.VERTICAL, command=self.img2pdf_listbox.yview)
        listbox_scrollbar.grid(row=0, column=1, sticky='ns')
        self.img2pdf_listbox.configure(yscrollcommand=listbox_scrollbar.set)
        
        # Buttons for list management
        button_frame = ttk.Frame(files_frame)
        button_frame.grid(row=1, column=0, columnspan=2, pady=10)
        
        ttk.Button(button_frame, text="Add Images...", command=self.add_img2pdf_files, style='Secondary.TButton').pack(
            side=tk.LEFT, padx=3)
        ttk.Button(button_frame, text="Remove Selected", command=self.remove_img2pdf_files, style='Secondary.TButton').pack(
            side=tk.LEFT, padx=3)
        ttk.Button(button_frame, text="Clear List", command=self.clear_img2pdf_list, style='Secondary.TButton').pack(
            side=tk.LEFT, padx=3)
        ttk.Button(button_frame, text="Move Up", command=lambda: self.move_img2pdf_item(-1), style='Secondary.TButton').pack(
            side=tk.LEFT, padx=3)
        ttk.Button(button_frame, text="Move Down", command=lambda: self.move_img2pdf_item(1), style='Secondary.TButton').pack(
            side=tk.LEFT, padx=3)
        
        # Output file and options
        options_frame = ttk.LabelFrame(frame, text="Output Options", padding="10")
        options_frame.grid(row=2, column=0, columnspan=2, padx=5, pady=15, sticky='ew')
        options_frame.columnconfigure(1, weight=1)
        
        # Output file path
        ttk.Label(options_frame, text="Output PDF:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        ttk.Entry(options_frame, textvariable=self.img2pdf_output_path).grid(row=0, column=1, padx=5, pady=5, sticky='ew')
        ttk.Button(options_frame, text="Save As...", command=self.browse_img2pdf_output, style='Secondary.TButton').grid(
            row=0, column=2, padx=5, pady=5)
        
        # DPI setting
        ttk.Label(options_frame, text="Resolution (DPI):").grid(row=1, column=0, padx=5, pady=5, sticky='w')
        dpi_options = [100, 150, 200, 300, 600]
        dpi_combobox = ttk.Combobox(options_frame, textvariable=self.img2pdf_dpi, values=dpi_options, width=5)
        dpi_combobox.grid(row=1, column=1, padx=5, pady=5, sticky='w')
        
        # Page size
        ttk.Label(options_frame, text="Page Size:").grid(row=2, column=0, padx=5, pady=5, sticky='w')
        page_sizes = ["A4", "Letter", "Legal", "A3", "auto"]
        size_combobox = ttk.Combobox(options_frame, textvariable=self.img2pdf_page_size, values=page_sizes, width=10)
        size_combobox.grid(row=2, column=1, padx=5, pady=5, sticky='w')
        
        # Margin
        ttk.Label(options_frame, text="Margin (pixels):").grid(row=3, column=0, padx=5, pady=5, sticky='w')
        margin_spinbox = ttk.Spinbox(options_frame, from_=0, to=200, textvariable=self.img2pdf_margin, width=5)
        margin_spinbox.grid(row=3, column=1, padx=5, pady=5, sticky='w')
        
        # Action Button
        button_frame = ttk.Frame(frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=10)
        
        self.img2pdf_button = ttk.Button(button_frame, text="Create PDF from Images", 
                                       command=self.start_img2pdf_thread, style='Primary.TButton', width=25)
        self.img2pdf_button.pack(pady=10)

    def create_pdf2img_page(self):
        """Create the PDF to Images page."""
        frame = self.pdf2img_frame
        frame.columnconfigure(1, weight=1)
        
        # Section title
        ttk.Label(frame, text="Extract Images from PDF", style='Title.TLabel').grid(
            row=0, column=0, columnspan=3, pady=(0, 15), sticky='w')
        
        # Input PDF
        ttk.Label(frame, text="Input PDF:").grid(row=1, column=0, padx=5, pady=5, sticky='w')
        ttk.Entry(frame, textvariable=self.pdf2img_input_path).grid(row=1, column=1, padx=5, pady=5, sticky='ew')
        ttk.Button(frame, text="Browse...", command=self.browse_pdf2img_input, style='Secondary.TButton').grid(
            row=1, column=2, padx=5, pady=5)
        
        # Output Directory
        ttk.Label(frame, text="Output Directory:").grid(row=2, column=0, padx=5, pady=5, sticky='w')
        ttk.Entry(frame, textvariable=self.pdf2img_output_dir).grid(row=2, column=1, padx=5, pady=5, sticky='ew')
        ttk.Button(frame, text="Browse...", command=self.browse_pdf2img_output, style='Secondary.TButton').grid(
            row=2, column=2, padx=5, pady=5)
        
        # Options frame
        options_frame = ttk.LabelFrame(frame, text="Extraction Options", padding="10")
        options_frame.grid(row=3, column=0, columnspan=3, padx=5, pady=15, sticky='ew')
        options_frame.columnconfigure(1, weight=1)
        
        # Image format
        ttk.Label(options_frame, text="Image Format:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        formats = ["png", "jpg", "tiff", "bmp"]
        format_combobox = ttk.Combobox(options_frame, textvariable=self.pdf2img_format, values=formats, width=5)
        format_combobox.grid(row=0, column=1, padx=5, pady=5, sticky='w')
        
        # Output prefix
        ttk.Label(options_frame, text="Output Prefix:").grid(row=1, column=0, padx=5, pady=5, sticky='w')
        ttk.Entry(options_frame, textvariable=self.pdf2img_prefix).grid(row=1, column=1, padx=5, pady=5, sticky='ew')
        ttk.Label(options_frame, text="(Optional)").grid(row=1, column=2, padx=5, pady=5, sticky='w')
        
        # Action Button
        button_frame = ttk.Frame(frame)
        button_frame.grid(row=4, column=0, columnspan=3, pady=20)
        
        self.pdf2img_button = ttk.Button(button_frame, text="Extract Images", 
                                       command=self.start_pdf2img_thread, style='Primary.TButton', width=20)
        self.pdf2img_button.pack(pady=10)
        
        # Warning about quality
        warning_text = "Note: For best quality extraction, use specialized tools like pdf2image or Poppler."
        ttk.Label(button_frame, text=warning_text, font=('Helvetica', 9, 'italic'), 
                foreground='#555555').pack(pady=(5, 0))

    def create_status_bar(self):
        """Create the status bar and progress indicator."""
        self.status_frame = ttk.Frame(self.root, relief=tk.SUNKEN, padding=2)
        self.status_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
        
        # Status message with icon
        self.status_icon_label = ttk.Label(self.status_frame, text="✓", width=2)
        self.status_icon_label.pack(side=tk.LEFT, padx=(5, 0))
        
        self.status_label = ttk.Label(self.status_frame, text="Ready", anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Progress bar - hidden initially
        self.progress_bar = ttk.Progressbar(self.status_frame, orient=tk.HORIZONTAL, mode='determinate', length=200)
        # self.progress_bar.pack(side=tk.RIGHT, padx=5)
        # self.progress_bar.pack_forget()  # Hide initially

    def bind_events(self):
        """Bind events for the application."""
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Notebook tab change
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)

    # --- Theme Management ---

    def apply_theme(self, theme_name):
        """Apply the selected theme."""
        # For standard ttk
        self.style.theme_use(theme_name)
        self.current_theme.set(theme_name)
        
        # For ThemedTk if available
        if THEMED_TK_AVAILABLE and hasattr(self.root, 'set_theme'):
            self.root.set_theme(theme_name)
        
        # Update dark mode checkbox to match theme
        is_dark = theme_name in DARK_THEMES
        self.dark_mode.set(is_dark)

    def toggle_dark_mode(self):
        """Toggle between dark and light modes."""
        if self.dark_mode.get():
            # Switch to a dark theme
            self.apply_theme(DARK_THEMES[0])
        else:
            # Switch to a light theme
            self.apply_theme(LIGHT_THEMES[0])

    # --- Event Handlers ---

    def on_close(self):
        """Handle window close."""
        # Check if an operation is running
        if self.worker_thread and self.worker_thread.is_alive():
            if messagebox.askyesno("Confirm Exit", 
                                 "An operation is still running. Are you sure you want to exit?"):
                # We can't gracefully stop threads in Python, but we can cancel the queue checker
                if self.after_id:
                    self.root.after_cancel(self.after_id)
                self.root.destroy()
        else:
            self.root.destroy()

    def on_tab_changed(self, event):
        """Handle notebook tab changes."""
        tab_id = self.notebook.index(self.notebook.select())
        tab_name = self.notebook.tab(tab_id, "text")
        self.set_status(f"Ready to {tab_name}")

    def update_split_ui_state(self):
        """Enable/disable ranges entry based on radio button selection."""
        if self.split_method.get() == "ranges":
            self.ranges_entry.config(state=tk.NORMAL)
        else:
            self.ranges_entry.config(state=tk.DISABLED)

    # --- File Dialog Handlers ---

    def browse_split_input(self):
        """Browse for input PDF to split."""
        filepath = filedialog.askopenfilename(
            title="Select Input PDF",
            filetypes=[("PDF Files", "*.pdf"), ("All Files", "*.*")]
        )
        if filepath:
            self.split_input_path.set(filepath)
            # Set default output directory to same as input
            if not self.split_output_dir.get():
                self.split_output_dir.set(os.path.dirname(filepath))
            self.set_status(f"Selected input PDF: {os.path.basename(filepath)}")

    def browse_split_output(self):
        """Browse for output directory for split operation."""
        dirpath = filedialog.askdirectory(
            title="Select Output Directory",
            mustexist=False
        )
        if dirpath:
            self.split_output_dir.set(dirpath)
            self.set_status(f"Selected output directory: {dirpath}")

    def add_merge_files(self):
        """Add PDFs to the merge list."""
        filepaths = filedialog.askopenfilenames(
            title="Select Input PDFs to Merge",
            filetypes=[("PDF Files", "*.pdf"), ("All Files", "*.*")]
        )
        if filepaths:
            count = 0
            for fp in filepaths:
                if fp not in self.merge_input_paths:
                    self.merge_input_paths.append(fp)
                    self.merge_listbox.insert(tk.END, os.path.basename(fp))
                    count += 1
            self.set_status(f"Added {count} PDF file(s) to merge list")

    def remove_merge_files(self):
        """Remove selected PDFs from the merge list."""
        selected_indices = self.merge_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("Selection Error", "Please select file(s) to remove from the list.")
            return

        # Remove from listbox and internal list (iterate backwards for indices)
        for i in sorted(selected_indices, reverse=True):
            del self.merge_input_paths[i]
            self.merge_listbox.delete(i)
        
        self.set_status(f"Removed {len(selected_indices)} file(s) from merge list")

    def clear_merge_list(self):
        """Clear all PDFs from the merge list."""
        if not self.merge_input_paths:
            return
            
        if messagebox.askyesno("Confirm Clear", "Are you sure you want to clear the entire merge list?"):
            self.merge_listbox.delete(0, tk.END)
            self.merge_input_paths.clear()
            self.set_status("Merge list cleared")

    def move_merge_item(self, direction):
        """Move selected PDF up or down in the merge list."""
        selected_indices = self.merge_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("Selection Error", "Please select a file to move.")
            return
        if len(selected_indices) > 1:
            messagebox.showwarning("Selection Error", "Please select only one file to move at a time.")
            return

        index = selected_indices[0]
        new_index = index + direction

        # Check bounds
        if new_index < 0 or new_index >= self.merge_listbox.size():
            return  # Cannot move further

        # Swap in internal list
        self.merge_input_paths[index], self.merge_input_paths[new_index] = \
            self.merge_input_paths[new_index], self.merge_input_paths[index]

        # Update listbox
        text = self.merge_listbox.get(index)
        self.merge_listbox.delete(index)
        self.merge_listbox.insert(new_index, text)
        self.merge_listbox.selection_clear(0, tk.END)
        self.merge_listbox.selection_set(new_index)
        self.merge_listbox.activate(new_index)
        self.merge_listbox.see(new_index)  # Ensure visible

    def browse_merge_output(self):
        """Browse for output PDF for merge operation."""
        filepath = filedialog.asksaveasfilename(
            title="Save Merged PDF As...",
            defaultextension=".pdf",
            filetypes=[("PDF Files", "*.pdf"), ("All Files", "*.*")]
        )
        if filepath:
            # Ensure it ends with .pdf
            if not filepath.lower().endswith(".pdf"):
                filepath += ".pdf"
            self.merge_output_path.set(filepath)
            self.set_status(f"Selected output file: {os.path.basename(filepath)}")

    def add_img2pdf_files(self):
        """Add images to the image-to-PDF list."""
        filepaths = filedialog.askopenfilenames(
            title="Select Images to Convert",
            filetypes=[
                ("All Supported", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.gif"),
                ("JPEG", "*.jpg *.jpeg"),
                ("PNG", "*.png"),
                ("BMP", "*.bmp"),
                ("TIFF", "*.tiff *.tif"),
                ("GIF", "*.gif"),
                ("All Files", "*.*")
            ]
        )
        if filepaths:
            count = 0
            # Validate image files
            for fp in filepaths:
                try:
                    # Check if it's a valid image file
                    with Image.open(fp) as img:
                        # It's a valid image if we got here
                        if fp not in self.img2pdf_input_paths:
                            self.img2pdf_input_paths.append(fp)
                            self.img2pdf_listbox.insert(tk.END, os.path.basename(fp))
                            count += 1
                except (UnidentifiedImageError, IOError, FileNotFoundError):
                    messagebox.showwarning("Invalid Image", 
                                        f"The file '{os.path.basename(fp)}' is not a valid image and was skipped.")
            
            if count > 0:
                # Set a default output file name if none exists
                if not self.img2pdf_output_path.get() and self.img2pdf_input_paths:
                    first_img = os.path.basename(self.img2pdf_input_paths[0])
                    base_name = os.path.splitext(first_img)[0]
                    if count > 1:
                        base_name += f"_and_{count-1}_more"
                    self.img2pdf_output_path.set(os.path.join(os.path.dirname(self.img2pdf_input_paths[0]), 
                                                          f"{base_name}.pdf"))
                
                self.set_status(f"Added {count} image(s) to conversion list")

    def remove_img2pdf_files(self):
        """Remove selected images from the image-to-PDF list."""
        selected_indices = self.img2pdf_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("Selection Error", "Please select image(s) to remove from the list.")
            return

        # Remove from listbox and internal list (iterate backwards for indices)
        for i in sorted(selected_indices, reverse=True):
            del self.img2pdf_input_paths[i]
            self.img2pdf_listbox.delete(i)
        
        self.set_status(f"Removed {len(selected_indices)} image(s) from conversion list")

    def clear_img2pdf_list(self):
        """Clear all images from the image-to-PDF list."""
        if not self.img2pdf_input_paths:
            return
            
        if messagebox.askyesno("Confirm Clear", "Are you sure you want to clear the entire image list?"):
            self.img2pdf_listbox.delete(0, tk.END)
            self.img2pdf_input_paths.clear()
            self.set_status("Image list cleared")

    def move_img2pdf_item(self, direction):
        """Move selected image up or down in the image-to-PDF list."""
        selected_indices = self.img2pdf_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("Selection Error", "Please select an image to move.")
            return
        if len(selected_indices) > 1:
            messagebox.showwarning("Selection Error", "Please select only one image to move at a time.")
            return

        index = selected_indices[0]
        new_index = index + direction

        # Check bounds
        if new_index < 0 or new_index >= self.img2pdf_listbox.size():
            return  # Cannot move further

        # Swap in internal list
        self.img2pdf_input_paths[index], self.img2pdf_input_paths[new_index] = \
            self.img2pdf_input_paths[new_index], self.img2pdf_input_paths[index]

        # Update listbox
        text = self.img2pdf_listbox.get(index)
        self.img2pdf_listbox.delete(index)
        self.img2pdf_listbox.insert(new_index, text)
        self.img2pdf_listbox.selection_clear(0, tk.END)
        self.img2pdf_listbox.selection_set(new_index)
        self.img2pdf_listbox.activate(new_index)
        self.img2pdf_listbox.see(new_index)  # Ensure visible

    def browse_img2pdf_output(self):
        """Browse for output PDF for image-to-PDF operation."""
        filepath = filedialog.asksaveasfilename(
            title="Save PDF As...",
            defaultextension=".pdf",
            filetypes=[("PDF Files", "*.pdf"), ("All Files", "*.*")]
        )
        if filepath:
            # Ensure it ends with .pdf
            if not filepath.lower().endswith(".pdf"):
                filepath += ".pdf"
            self.img2pdf_output_path.set(filepath)
            self.set_status(f"Selected output file: {os.path.basename(filepath)}")

    def browse_pdf2img_input(self):
        """Browse for input PDF for PDF-to-image operation."""
        filepath = filedialog.askopenfilename(
            title="Select Input PDF",
            filetypes=[("PDF Files", "*.pdf"), ("All Files", "*.*")]
        )
        if filepath:
            self.pdf2img_input_path.set(filepath)
            # Set default output directory to same as input
            if not self.pdf2img_output_dir.get():
                output_dir = os.path.join(os.path.dirname(filepath), 
                                      os.path.splitext(os.path.basename(filepath))[0] + "_images")
                self.pdf2img_output_dir.set(output_dir)
            self.set_status(f"Selected input PDF: {os.path.basename(filepath)}")

    def browse_pdf2img_output(self):
        """Browse for output directory for PDF-to-image operation."""
        dirpath = filedialog.askdirectory(
            title="Select Output Directory for Images",
            mustexist=False
        )
        if dirpath:
            self.pdf2img_output_dir.set(dirpath)
            self.set_status(f"Selected output directory: {dirpath}")

    # --- Status and UI Management ---

    def set_status(self, message, level="info"):
        """Update the status bar with the given message and level."""
        self.status_label.config(text=message)
        
        # Set icon based on level
        if level == "error":
            self.status_icon_label.config(text="✗", foreground="red")
        elif level == "warning":
            self.status_icon_label.config(text="⚠", foreground="orange")
        elif level == "success":
            self.status_icon_label.config(text="✓", foreground="green")
        else:  # info
            self.status_icon_label.config(text="ℹ", foreground="blue")
        
        # Force update
        self.root.update_idletasks()

    def set_ui_state(self, enabled: bool):
        """Enable or disable UI elements during processing."""
        state = tk.NORMAL if enabled else tk.DISABLED
        
        # Disable notebook tabs
        for i in range(self.notebook.index("end")):
            self.notebook.tab(i, state=state)
        
        # Disable buttons
        for button in [self.split_button, self.merge_button, 
                      self.img2pdf_button, self.pdf2img_button]:
            button.config(state=state)
        
        # Re-enable range entry based on radio button if enabling UI
        if enabled:
            self.update_split_ui_state()

    def show_progress(self):
        """Make the progress bar visible."""
        self.progress_bar.pack(side=tk.RIGHT, padx=5)
        self.status_frame.update_idletasks()

    def hide_progress(self):
        """Hide the progress bar."""
        self.progress_bar.pack_forget()
        self.progress_bar['value'] = 0
        self.status_frame.update_idletasks()

    def check_queues(self):
        """Periodically check queues for updates from worker threads."""
        try:
            # Process status messages
            while not self.status_queue.empty():
                level, msg = self.status_queue.get_nowait()
                self.set_status(msg, level)
                
                # Add to history
                timestamp = datetime.now().strftime("%H:%M:%S")
                self.operation_history.append((timestamp, level, msg))
                
                # Log to console/file
                if level == "error":
                    logger.error(msg)
                elif level == "warning":
                    logger.warning(msg)
                elif level == "success":
                    logger.info(f"SUCCESS: {msg}")
                else:
                    logger.info(msg)

            # Process progress updates
            while not self.progress_queue.empty():
                progress_update = self.progress_queue.get_nowait()
                ptype = progress_update[0]

                if ptype == "start":
                    _, current, total = progress_update
                    self.progress_bar['maximum'] = total
                    self.progress_bar['value'] = current
                    self.show_progress()
                elif ptype == "update":
                    _, current, total = progress_update
                    self.progress_bar['maximum'] = total
                    self.progress_bar['value'] = current
                elif ptype == "done":
                    _, current, total = progress_update
                    self.progress_bar['maximum'] = total
                    self.progress_bar['value'] = current
                    self.set_ui_state(enabled=True)
                    self.hide_progress()
                elif ptype == "error":
                    self.set_ui_state(enabled=True)
                    self.hide_progress()

        except queue.Empty:
            pass
        except Exception as e:
            logger.exception(f"Error checking queues: {e}")

        # Reschedule the check
        self.after_id = self.root.after(100, self.check_queues)

    # --- Operation Threads ---

    def start_split_thread(self):
        """Validate inputs and start the split operation in a thread."""
        input_pdf = self.split_input_path.get()
        output_dir = self.split_output_dir.get()
        method = self.split_method.get()
        ranges = self.split_ranges.get()
        prefix = self.split_prefix.get()

        # Input Validation
        if not input_pdf or not os.path.exists(input_pdf):
            messagebox.showerror("Input Error", "Please select a valid input PDF file.")
            return
        if not os.path.isfile(input_pdf):
            messagebox.showerror("Input Error", "The selected input path is not a file.")
            return
        if not output_dir:
            messagebox.showerror("Input Error", "Please select an output directory.")
            return

        if method == "ranges" and not ranges:
            messagebox.showerror("Input Error", "Please enter the page ranges to split by.")
            return
        if method == "ranges":
            # Basic validation for range format characters
            if not re.fullmatch(r"[\d\s,-]+", ranges):
                messagebox.showerror("Input Error", 
                                   "Invalid characters in page ranges.\nUse numbers, commas, hyphens, and spaces only.")
                return

        # Start Processing
        self.set_ui_state(enabled=False)
        self.set_status("Starting PDF split...", "info")

        # Clear queues before starting
        while not self.status_queue.empty(): self.status_queue.get()
        while not self.progress_queue.empty(): self.progress_queue.get()

        self.current_operation = "split"
        self.worker_thread = threading.Thread(
            target=perform_split,
            args=(input_pdf, output_dir, ranges if method == "ranges" else None,
                  method == "single", prefix, self.progress_queue, self.status_queue),
            daemon=True
        )
        self.worker_thread.start()

    def start_merge_thread(self):
        """Validate inputs and start the merge operation in a thread."""
        input_files = self.merge_input_paths
        output_file = self.merge_output_path.get()

        # Input Validation
        if not input_files:
            messagebox.showerror("Input Error", "Please add at least one PDF file to merge.")
            return
        if len(input_files) < 2:
            messagebox.showwarning("Input Warning", 
                                 "Merging requires at least two files. Continuing will essentially copy the single selected file.")

        if not output_file:
            messagebox.showerror("Input Error", "Please select an output file path.")
            return

        # Basic check if output is same as any input
        abs_output_path = os.path.abspath(output_file)
        for in_path in input_files:
            if os.path.abspath(in_path) == abs_output_path:
                messagebox.showerror("Input Error", 
                                   f"Output file cannot be the same as an input file:\n{os.path.basename(in_path)}")
                return

        # Start Processing
        self.set_ui_state(enabled=False)
        self.set_status("Starting PDF merge...", "info")

        # Clear queues before starting
        while not self.status_queue.empty(): self.status_queue.get()
        while not self.progress_queue.empty(): self.progress_queue.get()

        self.current_operation = "merge"
        self.worker_thread = threading.Thread(
            target=perform_merge,
            args=(list(input_files), output_file, self.progress_queue, self.status_queue),
            daemon=True
        )
        self.worker_thread.start()

    def start_img2pdf_thread(self):
        """Validate inputs and start the image-to-PDF operation in a thread."""
        input_images = self.img2pdf_input_paths
        output_file = self.img2pdf_output_path.get()
        dpi = self.img2pdf_dpi.get()
        page_size = self.img2pdf_page_size.get()
        margin = self.img2pdf_margin.get()

        # Input Validation
        if not input_images:
            messagebox.showerror("Input Error", "Please add at least one image to convert.")
            return
        if not output_file:
            messagebox.showerror("Input Error", "Please select an output PDF file path.")
            return
        
        # Validate DPI
        try:
            dpi = int(dpi)
            if dpi < 72 or dpi > 1200:
                messagebox.showwarning("Input Warning", "DPI value should be between 72 and 1200. Using 300 DPI.")
                dpi = 300
        except ValueError:
            messagebox.showwarning("Input Warning", "Invalid DPI value. Using 300 DPI.")
            dpi = 300
            
        # Validate margin
        try:
            margin = int(margin)
            if margin < 0 or margin > 500:
                messagebox.showwarning("Input Warning", "Margin should be between 0 and 500 pixels. Using 50 pixels.")
                margin = 50
        except ValueError:
            messagebox.showwarning("Input Warning", "Invalid margin value. Using 50 pixels.")
            margin = 50

        # Start Processing
        self.set_ui_state(enabled=False)
        self.set_status("Starting image to PDF conversion...", "info")

        # Clear queues before starting
        while not self.status_queue.empty(): self.status_queue.get()
        while not self.progress_queue.empty(): self.progress_queue.get()

        self.current_operation = "img2pdf"
        self.worker_thread = threading.Thread(
            target=convert_images_to_pdf,
            args=(list(input_images), output_file, dpi, page_size, margin, 
                  self.progress_queue, self.status_queue),
            daemon=True
        )
        self.worker_thread.start()

    def start_pdf2img_thread(self):
        """Validate inputs and start the PDF-to-image operation in a thread."""
        input_pdf = self.pdf2img_input_path.get()
        output_dir = self.pdf2img_output_dir.get()
        format = self.pdf2img_format.get()
        prefix = self.pdf2img_prefix.get()

        # Input Validation
        if not input_pdf or not os.path.exists(input_pdf):
            messagebox.showerror("Input Error", "Please select a valid input PDF file.")
            return
        if not os.path.isfile(input_pdf):
            messagebox.showerror("Input Error", "The selected input path is not a file.")
            return
        if not output_dir:
            messagebox.showerror("Input Error", "Please select an output directory for images.")
            return

        # Start Processing
        self.set_ui_state(enabled=False)
        self.set_status("Starting PDF to image extraction...", "info")

        # Clear queues before starting
        while not self.status_queue.empty(): self.status_queue.get()
        while not self.progress_queue.empty(): self.progress_queue.get()

        self.current_operation = "pdf2img"
        self.worker_thread = threading.Thread(
            target=extract_images_from_pdf,
            args=(input_pdf, output_dir, format, prefix, 
                  self.progress_queue, self.status_queue),
            daemon=True
        )
        self.worker_thread.start()


# --- Main Application Entry Point ---

def main():
    """Main entry point for the application."""
    # Check for missing dependencies
    if MISSING_DEPENDENCIES:
        missing_deps = "\n".join([f"  - {dep}" for dep in MISSING_DEPENDENCIES])
        print(f"Error: Missing dependencies:\n{missing_deps}")
        
        try:
            # Try to show a GUI error message
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror("Missing Dependencies", 
                               f"The following dependencies are missing:\n{missing_deps}\n\nPlease install them and try again.")
            root.destroy()
        except:
            pass
        
        sys.exit(1)
    
    # Create and start the application
    if THEMED_TK_AVAILABLE:
        root = ThemedTk(theme=DEFAULT_THEME)
    else:
        root = tk.Tk()
    
    app = AdvancedPdfToolkit(root)
    root.mainloop()


if __name__ == "__main__":
    main()
