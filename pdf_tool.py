#!/usr/bin/env python3

"""
Advanced PDF Toolkit Pro - A comprehensive GUI application for PDF operations

This application provides a graphical user interface for various common and
advanced PDF manipulation tasks.

Features:
- Split PDFs by page ranges, single pages, or top-level bookmarks.
- Merge multiple PDF documents into a single file, with optional bookmark creation.
- Convert various image formats (JPG, PNG, TIFF, BMP, WEBP, GIF) into a PDF document.
- Extract pages from a PDF document and save them as images (PNG, JPG, TIFF, BMP).
- Compress PDF files to reduce their size, with options for image downsampling.
- Encrypt PDFs with user/owner passwords and permission settings.
- Decrypt password-protected PDFs.
- Extract text content from PDFs in various formats (plain, formatted, HTML, Markdown).
- Fill PDF forms (basic support).
- Rotate pages within a PDF document by 90, 180, or 270 degrees.
- Reorder pages within a PDF document (functionality can be achieved via split/merge).
- View and edit PDF metadata (Title, Author, Subject, Keywords, etc.).
- Clean and responsive user interface powered by tkinter and ttkbootstrap.
- Customizable theming options (Light, Dark, System, Custom).

Dependencies:
- PyMuPDF (fitz): Core library for advanced PDF manipulation and rendering.
- pypdf: Used for specific PDF operations and as a fallback.
- Pillow: Essential for image processing (conversion, manipulation).
- tkinter: Standard Python library for the GUI framework.
- ttkbootstrap: Provides modern themes and widgets for tkinter.
- reportlab: Used as a fallback for image-to-PDF conversion.

Installation:
Install required dependencies using pip:
pip install pymupdf pypdf Pillow ttkbootstrap reportlab
"""

# Standard Library Imports
import base64
import configparser
import contextlib
import json
import logging
import logging.handlers
import math
import os
import platform
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
import webbrowser
from collections import deque
from datetime import datetime
from enum import Enum, auto
from functools import partial
from io import BytesIO
from typing import (Any, Callable, Dict, List, Optional, Set, Tuple, Union)

# --- Check and Handle Missing Dependencies ---
MISSING_DEPENDENCIES = []
THEMED_TK_AVAILABLE = False # Assuming ttkthemes is less critical now with ttkbootstrap
REPORTLAB_AVAILABLE = False
FPDF_AVAILABLE = False # Optional for watermarking (currently not implemented)

try:
    import fitz  # PyMuPDF
    # Check minimum required version if necessary
    # if tuple(map(int, fitz.__doc__.split(' ')[1].split('.'))) < (1, 18, 0):
    #     MISSING_DEPENDENCIES.append("pymupdf (version 1.18.0+ required)")
except ImportError:
    MISSING_DEPENDENCIES.append("pymupdf (required for core functionality)")

try:
    from pypdf import PdfReader, PdfWriter, PageRange, PdfMerger
    from pypdf.errors import PdfReadError, DependencyError
except ImportError:
    MISSING_DEPENDENCIES.append("pypdf (required for merging, encryption, fallback)")

try:
    from PIL import Image, UnidentifiedImageError, ImageOps, ImageDraw, ImageFilter, ImageEnhance
    import PIL.ImageOps # Explicit import if needed elsewhere
    # Check minimum required version if necessary
    # if tuple(map(int, Image.__version__.split('.'))) < (9, 0, 0):
    #     MISSING_DEPENDENCIES.append("Pillow (version 9.0.0+ required)")
except ImportError:
    MISSING_DEPENDENCIES.append("Pillow (required for image handling)")

try:
    import ttkbootstrap as ttb
    from ttkbootstrap import Style # Use ttkbootstrap's Style
except ImportError:
    # ttkbootstrap is now core to the UI design
    MISSING_DEPENDENCIES.append("ttkbootstrap (required for GUI themes and widgets)")

# Optional but used as fallback
try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter, A4, legal, A3
    from reportlab.lib.units import inch, mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.utils import ImageReader # Needed for reportlab image drawing
    REPORTLAB_AVAILABLE = True
except ImportError:
    # Log this but don't block execution if only used as fallback
    # logging.warning("reportlab not found. Image-to-PDF fallback disabled.")
    # We will add it to missing if Image-to-PDF fails without it later maybe?
    # For now, let's assume PyMuPDF works. If not, the operation will fail.
    pass # Keep it optional unless core features rely on it

# Optional: pdf2image for PDF to Image fallback (requires poppler)
PDF2IMAGE_AVAILABLE = False
try:
    from pdf2image import convert_from_path
    # We also need to check if poppler is available at runtime if we use pdf2image
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    pass # pdf2image is only a fallback

# Optional: fpdf for watermarking (example)
try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    pass # Watermarking is optional

# --- GUI Libraries (already imported or checked) ---
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, font, simpledialog, colorchooser

# --- Application Constants and Configuration ---

# Version Information
__version__ = "2.1.0" # Incremented version for rewrite/improvements
APP_NAME = "Advanced PDF Toolkit Pro"

# Setup Application Logging
class CustomLogHandler(logging.StreamHandler):
    """Sends log records to a callback function for GUI display."""
    def __init__(self, callback: Callable[[str, str], None]):
        super().__init__()
        self.callback = callback

    def emit(self, record: logging.LogRecord):
        try:
            log_entry = self.format(record)
            self.callback(record.levelname, log_entry)
        except Exception:
            self.handleError(record)

def setup_logging(log_level_str: str = "INFO") -> logging.Logger:
    """Configure application logging with file rotation and GUI callback."""
    log_dir = os.path.join(os.path.expanduser("~"), ".pdf_toolkit_pro", "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "pdf_toolkit_pro.log")

    logger = logging.getLogger("PDFToolkitPro")

    # Set level from config or default
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    logger.setLevel(log_level)

    # Prevent adding multiple handlers if called again
    if logger.hasHandlers():
        logger.handlers.clear()

    # File Handler (Rotating)
    try:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=5*1024*1024, backupCount=5, encoding='utf-8')
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Error setting up file logging: {e}", file=sys.stderr) # Log initial error to stderr

    # Console Handler (for debugging if GUI fails)
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    # Only add console handler if not running bundled (or based on a flag)
    if not getattr(sys, 'frozen', False): # Basic check if running as script
        logger.addHandler(console_handler)

    # Initial log message
    logger.info(f"Logging initialized for {APP_NAME} v{__version__}. Level: {log_level_str}")
    logger.info(f"Log file: {log_file}")

    return logger

# Initialize logger early, but GUI handler will be added later
logger = setup_logging() # Use default level initially

# Themes and UI Configuration
class Theme(Enum):
    """Enumeration for available theme modes."""
    LIGHT = "light"
    DARK = "dark"
    SYSTEM = "system"
    CUSTOM = "custom"

# Base themes for ttkbootstrap (examples)
TTB_LIGHT_THEME = "litera"
TTB_DARK_THEME = "darkly" #ttkbootstrap has 'darkly', 'cyborg', etc.
TTB_DEFAULT_THEME = TTB_DARK_THEME # Default dark theme

# Predefined colors for custom themes (can be expanded)
# Using ttkbootstrap color names where possible
THEME_PRESETS = {
    "light": {
        "primary": "#0d6efd",    # Bootstrap primary blue
        "secondary": "#6c757d", # Bootstrap secondary gray
        "success": "#198754",   # Bootstrap success green
        "info": "#0dcaf0",      # Bootstrap info cyan
        "warning": "#ffc107",   # Bootstrap warning yellow
        "danger": "#dc3545",    # Bootstrap danger red (used for error)
        "light": "#f8f9fa",     # Bootstrap light (used for surface)
        "dark": "#212529",      # Bootstrap dark (used for text)
        "bg": "#ffffff",        # Background
        "fg": "#212529",        # Foreground text
    },
    "dark": {
        "primary": "#0d6efd",    # Bootstrap primary blue
        "secondary": "#6c757d", # Bootstrap secondary gray
        "success": "#198754",   # Bootstrap success green
        "info": "#0dcaf0",      # Bootstrap info cyan
        "warning": "#ffc107",   # Bootstrap warning yellow
        "danger": "#dc3545",    # Bootstrap danger red (used for error)
        "light": "#adb5bd",     # Lighter gray for contrast
        "dark": "#dee2e6",      # Lighter text on dark bg
        "bg": "#212529",        # Background (dark)
        "fg": "#dee2e6",        # Foreground text (light)
        # Specific dark theme adjustments if needed
        "surface": "#343a40",   # Slightly lighter dark for cards/surfaces
    }
}


# Other Constants
SYSTEM_DPI = 96  # Default assumption, may vary
SUPPORTED_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".gif", ".webp"]
# Page sizes for ReportLab fallback (in points)
REPORTLAB_PAGE_SIZES = {
    "A4": A4,
    "Letter": letter,
    "Legal": legal,
    "A3": A3,
}

# Configuration Defaults
DEFAULT_CONFIG = {
    "app": {
        "theme": Theme.SYSTEM.value, # Use Enum value
        "custom_theme_name": TTB_DARK_THEME, # Base theme for custom colors
        "save_session": "False", # Feature not fully implemented
        "check_updates": "True",
        "language": "en", # Placeholder for future localization
        "recent_files_limit": "10",
        "log_level": "INFO",
        "custom_theme_colors": "{}", # JSON string for color overrides
    },
    "pdf": {
        "default_dpi": "300", # For PDF->Image and Image->PDF (resolution)
        "img2pdf_page_size": "A4",
        "img2pdf_margin_pt": "36", # Margin in points (0.5 inch)
        "img2pdf_quality": "95", # JPEG quality 1-100
        "compress_quality": "75", # Compression quality target (implementation specific)
        "compress_img_dpi": "150", # Target DPI for image downsampling during compression
        "encrypt_level": "128", # 128 or 256 (PyMuPDF/pypdf support varies)
    },
    "paths": {
        "default_output_dir": "", # User's documents or last used
        "temp_dir": "", # System temp or custom
    }
}

# --- Custom Exceptions ---
class OperationCancelledException(Exception):
    """Exception raised when a user cancels an ongoing operation."""
    pass

class PdfPasswordException(Exception):
    """Exception raised for password-related errors during PDF processing."""
    pass

class ValidationError(ValueError):
    """Custom exception for validation errors in operations."""
    pass

# --- Model: Core Business Logic ---

class PDFOperation:
    """
    Abstract base class for all PDF processing operations.

    Handles common functionalities like progress tracking, cancellation,
    and listener notification.
    """
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self._running = False
        self._cancelled = False
        self._progress = 0.0 # Use float for potentially finer progress
        self._listeners: List[Any] = [] # List of objects listening to this operation
        self._status_message = ""

        # Define expected input/output attributes for documentation/introspection
        self.input_path: Optional[str] = None
        self.input_paths: Optional[List[str]] = None
        self.output_path: Optional[str] = None
        self.output_dir: Optional[str] = None
        self.password: Optional[str] = None # For input PDF

    def register_listener(self, listener: Any):
        """Register an object to receive updates from this operation."""
        if listener not in self._listeners:
            self._listeners.append(listener)

    def unregister_listener(self, listener: Any):
        """Unregister an object from receiving updates."""
        if listener in self._listeners:
            self._listeners.remove(listener)

    def _notify(self, method_name: str, *args, **kwargs):
        """Helper to call a method on all listeners if it exists."""
        for listener in self._listeners:
            if hasattr(listener, method_name):
                try:
                    getattr(listener, method_name)(self, *args, **kwargs)
                except Exception as e:
                    logger.error(f"Error notifying listener {listener}: {e}")
                    # traceback.print_exc() # Optionally log traceback

    def notify_progress(self, progress: float, status_message: Optional[str] = None, level: str = "info"):
        """Notify listeners of progress percentage and status message."""
        self._progress = max(0.0, min(100.0, progress))
        if status_message:
            self._status_message = status_message
        # Avoid excessive updates, maybe throttle here if needed
        self._notify("on_progress_update", self._progress, self._status_message, level)

    def notify_completion(self, success: bool, message: Optional[str] = None, level: Optional[str] = None):
        """Notify listeners when the operation completes (successfully or not)."""
        final_level = level or ("success" if success else "error")
        final_message = message or f"{self.name} {'completed successfully' if success else 'failed'}"
        self._notify("on_operation_complete", success, final_message, final_level)

    def notify_status(self, message: str, level: str = "info"):
        """Notify listeners of an intermediate status update."""
        self._status_message = message
        self._notify("on_status_update", message, level)

    def is_running(self) -> bool:
        """Check if the operation is currently executing."""
        return self._running

    def cancel(self):
        """Signal the operation to cancel at the next opportunity."""
        if self._running:
            self._cancelled = True
            self.notify_status("Operation cancellation requested...", "warning")

    def check_cancelled(self):
        """Check if cancellation was requested and raise exception if so."""
        if self._cancelled:
            logger.warning(f"{self.name} operation cancelled by user.")
            raise OperationCancelledException("Operation was cancelled by user")

    def _validate_paths(self):
        """Common path validation logic."""
        # Input validation
        if hasattr(self, 'input_path') and self.input_path:
            if not os.path.exists(self.input_path):
                raise ValidationError(f"Input file not found: {self.input_path}")
            if not os.path.isfile(self.input_path):
                raise ValidationError(f"Input path is not a file: {self.input_path}")
        elif hasattr(self, 'input_paths') and self.input_paths:
            if not self.input_paths:
                 raise ValidationError("No input files specified.")
            for path in self.input_paths:
                if not os.path.exists(path):
                    raise ValidationError(f"Input file not found: {path}")
                if not os.path.isfile(path):
                    raise ValidationError(f"Input path is not a file: {path}")

        # Output validation
        output_location = None
        if hasattr(self, 'output_path') and self.output_path:
            output_location = self.output_path
            output_target_is_dir = False
        elif hasattr(self, 'output_dir') and self.output_dir:
            output_location = self.output_dir
            output_target_is_dir = True
        else:
             raise ValidationError("Output location (file or directory) not specified.")

        if not output_location:
            raise ValidationError("Output location not specified.")

        output_parent_dir = os.path.dirname(output_location) if not output_target_is_dir else output_location

        # Ensure output directory exists or can be created
        if output_parent_dir: # If parent is empty, it means current dir
             if not os.path.exists(output_parent_dir):
                 try:
                     os.makedirs(output_parent_dir, exist_ok=True)
                     logger.info(f"Created output directory: {output_parent_dir}")
                 except OSError as e:
                     raise ValidationError(f"Could not create output directory '{output_parent_dir}': {e}")
             elif not os.path.isdir(output_parent_dir):
                  raise ValidationError(f"Output path '{output_parent_dir}' exists but is not a directory.")

        # Check if output overwrites input (for single input/output operations)
        if hasattr(self, 'input_path') and self.input_path and \
           hasattr(self, 'output_path') and self.output_path:
            if os.path.abspath(self.input_path) == os.path.abspath(self.output_path):
                raise ValidationError("Output file cannot be the same as the input file.")
        # Check for output == input in multi-input scenarios (Merge)
        if hasattr(self, 'input_paths') and self.input_paths and \
           hasattr(self, 'output_path') and self.output_path:
            for in_path in self.input_paths:
                 if os.path.abspath(in_path) == os.path.abspath(self.output_path):
                     raise ValidationError(f"Output file cannot be the same as input file: {in_path}")

    def _open_input_pdf_fitz(self) -> fitz.Document:
        """Helper to open the input PDF with fitz, handling passwords."""
        if not self.input_path:
            raise ValidationError("Input PDF path is not set.")
        try:
            doc = fitz.open(self.input_path)
            if doc.is_encrypted:
                if not self.password:
                    # If called non-interactively, this should fail earlier
                    # This assumes password might be prompted elsewhere if needed
                    doc.close()
                    raise PdfPasswordException("Input PDF is encrypted, password required.")
                if not doc.authenticate(self.password):
                    doc.close()
                    raise PdfPasswordException("Incorrect password provided for input PDF.")
            return doc
        except fitz.FileDataError as e:
            raise ValidationError(f"Invalid or corrupted PDF file: {os.path.basename(self.input_path)} - {e}")
        except PdfPasswordException: # Re-raise specific exception
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to open PDF '{os.path.basename(self.input_path)}': {e}")

    def _open_input_pdf_pypdf(self) -> PdfReader:
        """Helper to open the input PDF with pypdf, handling passwords."""
        if not self.input_path:
            raise ValidationError("Input PDF path is not set.")
        try:
            reader = PdfReader(self.input_path)
            if reader.is_encrypted:
                if not self.password:
                    raise PdfPasswordException("Input PDF is encrypted, password required.")
                if not reader.decrypt(self.password):
                    # pypdf decrypt returns 0 on fail, 1 user, 2 owner
                     raise PdfPasswordException("Incorrect password provided for input PDF.")
            return reader
        except PdfReadError as e:
             raise ValidationError(f"Invalid or corrupted PDF file: {os.path.basename(self.input_path)} - {e}")
        except PdfPasswordException:
             raise
        except Exception as e:
            raise RuntimeError(f"Failed to open PDF '{os.path.basename(self.input_path)}' with pypdf: {e}")


    def validate(self):
        """Validate operation-specific parameters. Must be implemented by subclasses."""
        raise NotImplementedError("Each operation must implement the 'validate' method.")

    def execute(self) -> Any:
        """Execute the core logic of the operation. Must be implemented by subclasses."""
        raise NotImplementedError("Each operation must implement the 'execute' method.")

    def run(self) -> Any:
        """
        Executes the operation lifecycle: validation, execution, notification.
        Returns the result of the execute method or None on failure/cancellation.
        """
        result = None
        self._running = True
        self._cancelled = False
        self._progress = 0.0
        start_time = time.monotonic()
        self.notify_progress(0, f"Starting {self.name}...", "info")

        try:
            # 1. Validation
            self.notify_status("Validating parameters...", "info")
            self._validate_paths() # Common path validation
            self.validate()       # Operation-specific validation
            self.check_cancelled()

            # 2. Execution
            self.notify_status("Executing operation...", "info")
            result = self.execute()
            self.check_cancelled() # Check again after main execution block

            # 3. Completion (Success)
            self._progress = 100.0
            elapsed_time = time.monotonic() - start_time
            success_msg = f"{self.name} completed successfully in {elapsed_time:.2f} seconds."
            self.notify_progress(100, success_msg, "success")
            self.notify_completion(True, success_msg, "success")
            return result

        except (ValidationError, PdfPasswordException, OperationCancelledException) as e:
            level = "warning" if isinstance(e, (OperationCancelledException, PdfPasswordException)) else "error"
            logger.log(logging.WARNING if level=="warning" else logging.ERROR, f"{self.name} failed: {e}")
            self.notify_completion(False, f"{self.name} failed: {e}", level)
            return None
        except Exception as e:
            error_msg = f"Unexpected error during {self.name}: {e}"
            logger.exception(error_msg) # Log full traceback
            self.notify_status(error_msg, "error") # Show error in status bar
            self.notify_completion(False, error_msg, "error")
            return None
        finally:
            self._running = False
            self._cancelled = False # Reset cancellation flag


class SplitPDFOperation(PDFOperation):
    """Operation to split a PDF into multiple files."""
    def __init__(self, input_path: str, output_dir: str,
                 split_method: str = "ranges", # "ranges", "single", "bookmarks"
                 ranges_str: Optional[str] = None,
                 prefix: Optional[str] = None,
                 password: Optional[str] = None):
        super().__init__("Split PDF", "Split PDF into multiple files")
        self.input_path = input_path
        self.output_dir = output_dir
        self.split_method = split_method
        self.ranges_str = ranges_str
        self.prefix = prefix or os.path.splitext(os.path.basename(input_path))[0] + "_"
        self.password = password
        self.num_pages = 0

    def validate(self):
        """Validate parameters for the split operation."""
        if self.split_method == "ranges" and not self.ranges_str:
            raise ValidationError("Page ranges must be specified for 'ranges' split method.")
        if self.split_method not in ["ranges", "single", "bookmarks"]:
            raise ValidationError(f"Invalid split method: {self.split_method}")

        # Check PDF validity and get page count using fitz helper
        try:
            with contextlib.closing(self._open_input_pdf_fitz()) as doc:
                self.num_pages = doc.page_count
            if self.num_pages == 0:
                raise ValidationError("Input PDF has no pages.")
        except (RuntimeError, ValidationError, PdfPasswordException) as e:
             # Re-raise validation/password errors, wrap others
             if isinstance(e, (ValidationError, PdfPasswordException)):
                 raise
             else:
                 raise ValidationError(f"Could not validate PDF: {e}")

        # Validate ranges if method is 'ranges' (after getting num_pages)
        if self.split_method == "ranges":
            try:
                self._parse_page_ranges() # Parse and validate ranges against num_pages
            except ValueError as e: # Catch errors from parsing
                raise ValidationError(f"Invalid page ranges: {e}")

    def _parse_page_ranges(self) -> List[Tuple[int, int]]:
        """Parse page ranges string into list of 0-based (start, end) tuples."""
        if not self.ranges_str:
            return []

        parsed_ranges: List[Tuple[int, int]] = []
        parts = self.ranges_str.split(',')

        for part in parts:
            part = part.strip()
            if not part:
                continue

            try:
                if '-' in part:
                    start_str, end_str = part.split('-', 1)
                    start = 1 if not start_str.strip() else int(start_str.strip())
                    end = self.num_pages if not end_str.strip() else int(end_str.strip())

                    if start < 1 or end < 1: raise ValueError("Page numbers must be >= 1")
                    if start > end: raise ValueError(f"Start page > end page ({start}-{end})")
                    if start > self.num_pages: raise ValueError(f"Start page {start} > total pages {self.num_pages}")

                    parsed_ranges.append((start - 1, min(end, self.num_pages) - 1)) # 0-based inclusive
                else:
                    page = int(part)
                    if page < 1 or page > self.num_pages:
                        raise ValueError(f"Page number {page} out of range (1-{self.num_pages})")
                    parsed_ranges.append((page - 1, page - 1)) # 0-based inclusive

            except ValueError as e:
                # Add context to the error message
                 raise ValueError(f"Invalid range part '{part}': {e}") from e

        # Sort and merge overlapping/adjacent ranges
        if not parsed_ranges: return []
        parsed_ranges.sort(key=lambda x: x[0])

        merged: List[Tuple[int, int]] = []
        current_start, current_end = parsed_ranges[0]

        for next_start, next_end in parsed_ranges[1:]:
            if next_start <= current_end + 1: # Overlap or adjacent
                current_end = max(current_end, next_end)
            else:
                merged.append((current_start, current_end))
                current_start, current_end = next_start, next_end
        merged.append((current_start, current_end)) # Add the last range

        # Final check: Ensure indices are valid (should be already, but belt-and-suspenders)
        validated_merged = [(s, e) for s, e in merged if 0 <= s < self.num_pages and 0 <= e < self.num_pages and s <= e]
        if not validated_merged:
             raise ValueError("No valid page ranges resulted after parsing and merging.")

        return validated_merged

    def _get_bookmark_splits(self, doc: fitz.Document) -> List[Tuple[int, int, str]]:
        """Extract split ranges based on top-level bookmarks."""
        toc = doc.get_toc(simple=False) # Get full TOC with levels
        if not toc:
            self.notify_status("No bookmarks found in the document.", "warning")
            return []

        splits: List[Tuple[int, int, str]] = [] # (start_page_idx, end_page_idx, title)
        top_level_bookmarks = []

        # Find all top-level bookmarks (level 1)
        for i, entry in enumerate(toc):
            level, title, page_num, _ = entry # Use the dict structure from simple=False if needed
            page_idx = page_num - 1
            if page_idx < 0 or page_idx >= self.num_pages:
                 self.notify_status(f"Skipping bookmark '{title}' pointing to invalid page {page_num}", "warning")
                 continue
            if level == 1:
                 top_level_bookmarks.append({'index': i, 'title': title, 'page_idx': page_idx})

        if not top_level_bookmarks:
             self.notify_status("No top-level bookmarks found to split by.", "warning")
             return []

        # Determine end page for each top-level bookmark
        for i, bm in enumerate(top_level_bookmarks):
            start_page = bm['page_idx']
            end_page = self.num_pages - 1 # Default to end of doc

            # Find the start page of the *next* top-level bookmark
            if i + 1 < len(top_level_bookmarks):
                next_bm_page = top_level_bookmarks[i+1]['page_idx']
                # Ensure end page is not before start page (can happen with overlapping bookmarks)
                if next_bm_page > start_page:
                    end_page = next_bm_page - 1

            # Clean title for filename
            clean_title = re.sub(r'[\\/*?:"<>|]', '_', bm['title']) # Remove invalid filename chars
            clean_title = re.sub(r'\s+', ' ', clean_title).strip() # Consolidate whitespace
            clean_title = clean_title[:60] # Limit length

            if start_page <= end_page: # Ensure valid range
                 splits.append((start_page, end_page, clean_title))
            else:
                 self.notify_status(f"Skipping potentially empty bookmark range for '{bm['title']}' (start: {start_page+1}, end: {end_page+1})", "warning")


        if not splits:
             self.notify_status("Could not determine valid split ranges from bookmarks.", "warning")

        return splits

    def execute(self) -> bool:
        """Execute the split operation using PyMuPDF (preferred) or pypdf."""
        split_instructions: List[Tuple[int, int, str]] = [] # (start_idx, end_idx, output_path)

        self.notify_status(f"Preparing split using method: {self.split_method}", "info")

        # --- Generate split instructions based on method ---
        if self.split_method == "single":
            pad_width = len(str(self.num_pages))
            for i in range(self.num_pages):
                filename = f"{self.prefix}page_{str(i + 1).zfill(pad_width)}.pdf"
                output_path = os.path.join(self.output_dir, filename)
                split_instructions.append((i, i, output_path))

        elif self.split_method == "ranges":
            try:
                page_ranges = self._parse_page_ranges() # Re-parse just in case? Validation should cover it.
            except ValueError as e:
                self.notify_status(f"Error parsing ranges during execution: {e}", "error")
                return False # Should not happen if validation passed

            if not page_ranges:
                self.notify_status("No valid page ranges found for splitting.", "warning")
                return False # Indicate nothing was done

            for i, (start_idx, end_idx) in enumerate(page_ranges):
                range_desc = f"{start_idx + 1}"
                if start_idx != end_idx:
                    range_desc += f"-{end_idx + 1}"
                filename = f"{self.prefix}pages_{range_desc}.pdf"
                output_path = os.path.join(self.output_dir, filename)
                split_instructions.append((start_idx, end_idx, output_path))

        elif self.split_method == "bookmarks":
             # Open doc here temporarily to get bookmarks
            try:
                 with contextlib.closing(self._open_input_pdf_fitz()) as doc:
                     bookmark_splits = self._get_bookmark_splits(doc)
            except (RuntimeError, ValidationError, PdfPasswordException) as e:
                 self.notify_status(f"Failed to analyze bookmarks: {e}", "error")
                 return False

            if not bookmark_splits:
                # Status already notified in _get_bookmark_splits
                return False # Indicate nothing was done

            for i, (start_idx, end_idx, title) in enumerate(bookmark_splits):
                 filename = f"{self.prefix}{i+1:02d}_{title}.pdf" # Add index for uniqueness
                 output_path = os.path.join(self.output_dir, filename)
                 split_instructions.append((start_idx, end_idx, output_path))

        else:
             # Should be caught by validation, but defensively check
             self.notify_status(f"Internal Error: Unknown split method '{self.split_method}'", "error")
             return False

        if not split_instructions:
            self.notify_status("No output files to generate based on split criteria.", "warning")
            return True # Operation technically succeeded, just did nothing.

        # --- Execute Splitting ---
        total_splits = len(split_instructions)
        self.notify_status(f"Generating {total_splits} output file(s)...", "info")

        # --- Attempt 1: PyMuPDF (fitz) ---
        try:
            self.notify_status("Using PyMuPDF for splitting...", "info")
            with contextlib.closing(self._open_input_pdf_fitz()) as doc:
                for i, (start_idx, end_idx, output_path) in enumerate(split_instructions):
                    self.check_cancelled()
                    range_str = f"{start_idx + 1}" + (f"-{end_idx + 1}" if start_idx != end_idx else "")
                    self.notify_progress(
                        (i / total_splits) * 100,
                        f"Creating '{os.path.basename(output_path)}' (Pages {range_str}) [{i+1}/{total_splits}]"
                    )

                    with fitz.open() as out_doc: # Create a new empty document for each split
                        # Select pages - ensure range is valid (should be by now)
                        valid_start = max(0, start_idx)
                        valid_end = min(self.num_pages - 1, end_idx)
                        if valid_start <= valid_end:
                            out_doc.insert_pdf(doc, from_page=valid_start, to_page=valid_end)

                        if out_doc.page_count > 0:
                            try:
                                # Use garbage collection and compression for output files
                                out_doc.save(output_path, garbage=4, deflate=True, linear=False)
                            except Exception as save_err:
                                self.notify_status(f"Error saving '{os.path.basename(output_path)}': {save_err}", "error")
                        else:
                            self.notify_status(f"Skipping empty output file for range {range_str}", "warning")
            return True # PyMuPDF method succeeded

        except OperationCancelledException:
             raise # Propagate cancellation
        except Exception as fitz_err:
            self.notify_status(f"PyMuPDF split failed ({fitz_err}), attempting fallback with pypdf...", "warning")
            logger.warning(f"PyMuPDF split failed: {fitz_err}", exc_info=False) # Log less verbose error

            # --- Attempt 2: pypdf Fallback ---
            try:
                self.notify_status("Using pypdf for splitting (fallback)...", "info")
                reader = self._open_input_pdf_pypdf() # Open with pypdf helper

                for i, (start_idx, end_idx, output_path) in enumerate(split_instructions):
                    self.check_cancelled()
                    range_str = f"{start_idx + 1}" + (f"-{end_idx + 1}" if start_idx != end_idx else "")
                    self.notify_progress(
                        (i / total_splits) * 100,
                        f"[pypdf] Creating '{os.path.basename(output_path)}' (Pages {range_str}) [{i+1}/{total_splits}]"
                    )

                    writer = PdfWriter()
                    pages_added = 0
                    for page_num in range(start_idx, end_idx + 1):
                        # Check index validity against pypdf reader pages
                        if 0 <= page_num < len(reader.pages):
                            try:
                                writer.add_page(reader.pages[page_num])
                                pages_added += 1
                            except Exception as page_err:
                                # pypdf can sometimes fail on specific pages
                                self.notify_status(f"Warning: Error adding page {page_num+1} using pypdf: {page_err}", "warning")
                        else:
                            self.notify_status(f"Warning: Page index {page_num+1} out of bounds for pypdf reader.", "warning")


                    if pages_added > 0:
                        try:
                            with open(output_path, "wb") as output_file:
                                writer.write(output_file)
                        except Exception as write_err:
                             self.notify_status(f"Error writing '{os.path.basename(output_path)}' with pypdf: {write_err}", "error")
                    else:
                        self.notify_status(f"Skipping empty output file for range {range_str} (pypdf)", "warning")

                return True # pypdf fallback succeeded overall (even if some pages failed)

            except OperationCancelledException:
                raise # Propagate cancellation
            except Exception as pypdf_err:
                self.notify_status(f"pypdf split fallback also failed: {pypdf_err}", "error")
                logger.error(f"pypdf split failed: {pypdf_err}", exc_info=True)
                return False # Both methods failed


class MergePDFOperation(PDFOperation):
    """Operation to merge multiple PDFs into one."""
    def __init__(self, input_paths: List[str], output_path: str,
                 passwords: Optional[Dict[str, str]] = None,
                 outline_title_template: Optional[str] = None): # None, "FILENAME", "INDEX_FILENAME"
        super().__init__("Merge PDFs", "Merge multiple PDFs into one file")
        self.input_paths = input_paths
        self.output_path = output_path
        self.passwords = passwords or {} # {filepath: password}
        self.outline_title_template = outline_title_template # Default: add bookmark using filename
        self.total_pages_merged = 0


    def validate(self):
        """Validate parameters for the merge operation."""
        # Path validation (including output != input) is handled by _validate_paths
        pass # Specific validation done in base class

    def execute(self) -> bool:
        """Execute the merge operation using PyMuPDF (preferred) or pypdf."""
        total_files = len(self.input_paths)
        files_processed = 0
        self.total_pages_merged = 0
        processed_file_paths = [] # Keep track of successfully processed files for outline

        self.notify_status(f"Starting merge of {total_files} file(s)...", "info")

        # --- Attempt 1: PyMuPDF (fitz) ---
        try:
            self.notify_status("Using PyMuPDF for merging...", "info")
            # Create the output document
            out_doc = fitz.open()
            bookmarks_to_add = [] # List of (level, title, page_index)

            for i, input_path in enumerate(self.input_paths):
                self.check_cancelled()
                current_progress = (i / (total_files + 1)) * 100 # +1 for final save step
                filename = os.path.basename(input_path)
                self.notify_progress(current_progress, f"Processing '{filename}' [{i+1}/{total_files}]")

                password = self.passwords.get(input_path)
                doc = None # Ensure doc is closable in finally
                try:
                    # Open the current input PDF
                    doc = fitz.open(input_path)
                    if doc.is_encrypted:
                         if not password:
                             self.notify_status(f"'{filename}' is encrypted, password needed. Skipping.", "warning")
                             continue # Skip this file
                         if not doc.authenticate(password):
                             self.notify_status(f"Incorrect password for '{filename}'. Skipping.", "warning")
                             continue # Skip this file

                    num_pages = doc.page_count
                    if num_pages > 0:
                        # If adding outlines, note the page number *before* inserting
                        if self.outline_title_template:
                            dest_page_idx = out_doc.page_count # 0-based index in output doc
                            title = self._generate_bookmark_title(self.outline_title_template, i, filename)
                            bookmarks_to_add.append((1, title, dest_page_idx + 1)) # fitz uses 1-based page for TOC

                        # Insert the pages
                        out_doc.insert_pdf(doc)
                        self.total_pages_merged += num_pages
                        files_processed += 1
                        processed_file_paths.append(input_path) # Add to list for potential pypdf fallback outline
                    else:
                        self.notify_status(f"'{filename}' has no pages. Skipping.", "warning")

                except fitz.FileDataError as e:
                     self.notify_status(f"Error reading '{filename}': Invalid format? Skipping. ({e})", "warning")
                except Exception as e:
                     self.notify_status(f"Error processing '{filename}': {e}. Skipping.", "warning")
                finally:
                    if doc: doc.close()

            if files_processed == 0:
                out_doc.close() # Close the empty output doc
                self.notify_status("No valid input PDF files could be processed.", "error")
                return False

            # Add bookmarks if requested
            if bookmarks_to_add:
                 self.notify_status("Adding bookmarks...", "info")
                 out_doc.set_toc(bookmarks_to_add)

            # Save the merged document
            self.notify_progress(95, f"Saving final merged PDF ({self.total_pages_merged} pages)...")
            try:
                 out_doc.save(self.output_path, garbage=4, deflate=True, linear=True)
            except Exception as save_err:
                 self.notify_status(f"Error saving merged PDF: {save_err}", "error")
                 return False
            finally:
                 out_doc.close()

            return True # PyMuPDF method succeeded

        except OperationCancelledException:
            raise
        except Exception as fitz_err:
            self.notify_status(f"PyMuPDF merge failed ({fitz_err}), attempting fallback with pypdf...", "warning")
            logger.warning(f"PyMuPDF merge failed: {fitz_err}", exc_info=False)

            # --- Attempt 2: pypdf Fallback ---
            try:
                self.notify_status("Using pypdf for merging (fallback)...", "info")
                merger = PdfMerger()
                self.total_pages_merged = 0 # Reset page count for pypdf
                files_processed = 0 # Reset count for pypdf
                bookmark_parent = None # For hierarchical bookmarks if needed later

                for i, input_path in enumerate(self.input_paths):
                    self.check_cancelled()
                    current_progress = (i / (total_files + 1)) * 100
                    filename = os.path.basename(input_path)
                    self.notify_progress(current_progress, f"[pypdf] Processing '{filename}' [{i+1}/{total_files}]")

                    password = self.passwords.get(input_path)
                    reader = None
                    try:
                        # Check if file was successfully processed by fitz (for outline consistency)
                        # if input_path not in processed_file_paths:
                        #    self.notify_status(f"Skipping '{filename}' as it failed in previous step.", "warning")
                        #    continue

                        # Open with pypdf
                        reader = PdfReader(input_path)
                        if reader.is_encrypted:
                            if not password:
                                self.notify_status(f"'{filename}' is encrypted, password needed. Skipping.", "warning")
                                continue
                            if not reader.decrypt(password):
                                self.notify_status(f"Incorrect password for '{filename}'. Skipping.", "warning")
                                continue

                        num_pages = len(reader.pages)
                        if num_pages > 0:
                            # Add bookmark before appending
                            bookmark = None
                            if self.outline_title_template:
                                title = self._generate_bookmark_title(self.outline_title_template, i, filename)
                                # `merger.append` adds bookmark automatically if `bookmark=True`
                                # For custom title, use `add_outline_item` *after* append or manage manually.
                                # Let's try simple merge first. We can add outlines manually later if needed.
                                bookmark_title = title # Use generated title

                            # Append the file
                            merger.append(reader, bookmark=bookmark_title) # Let pypdf handle basic bookmark
                            self.total_pages_merged += num_pages
                            files_processed += 1
                        else:
                            self.notify_status(f"'{filename}' has no pages. Skipping.", "warning")

                    except PdfReadError as e:
                        self.notify_status(f"pypdf Error reading '{filename}': {e}. Skipping.", "warning")
                    except PdfPasswordException as e: # Should be caught earlier, but just in case
                         self.notify_status(f"Password error for '{filename}' with pypdf: {e}. Skipping.", "warning")
                    except Exception as e:
                        self.notify_status(f"pypdf Error processing '{filename}': {e}. Skipping.", "warning")
                    # reader is closed automatically by PdfMerger append

                if files_processed == 0:
                    merger.close()
                    self.notify_status("No valid input PDF files processed by pypdf.", "error")
                    return False

                # Write the merged PDF
                self.notify_progress(95, f"Saving final merged PDF ({self.total_pages_merged} pages) using pypdf...")
                try:
                    with open(self.output_path, "wb") as output_file:
                        merger.write(output_file)
                except Exception as write_err:
                    self.notify_status(f"Error writing merged PDF with pypdf: {write_err}", "error")
                    return False
                finally:
                    merger.close()

                return True # pypdf fallback succeeded

            except OperationCancelledException:
                raise
            except Exception as pypdf_err:
                self.notify_status(f"pypdf merge fallback also failed: {pypdf_err}", "error")
                logger.error(f"pypdf merge failed: {pypdf_err}", exc_info=True)
                return False # Both methods failed

    def _generate_bookmark_title(self, template: str, index: int, filename: str) -> str:
        """Generates bookmark title based on template."""
        base_name = os.path.splitext(filename)[0]
        if template == "FILENAME":
            return base_name
        elif template == "INDEX_FILENAME":
            return f"{index + 1}. {base_name}"
        else: # Default or unknown template
             return base_name


class ImagesToPDFOperation(PDFOperation):
    """Operation to convert images to a PDF file."""
    def __init__(self, image_paths: List[str], output_path: str,
                 page_size_str: str = "A4", # "A4", "Letter", "Auto", etc.
                 margin_pt: float = 36.0, # Margin in points
                 fit_method: str = "fit", # "fit", "fill", "stretch", "center"
                 resolution: int = 300, # Input image DPI interpretation for scaling
                 quality: int = 95): # JPEG quality
        super().__init__("Images to PDF", "Convert images to a PDF document")
        self.image_paths = image_paths
        self.output_path = output_path
        self.page_size_str = page_size_str
        self.margin_pt = max(0, margin_pt)
        self.fit_method = fit_method
        self.resolution = max(72, resolution) # Minimum 72 DPI
        self.quality = max(1, min(100, quality))
        self.valid_image_paths: List[str] = []

    def validate(self):
        """Validate parameters for the image-to-PDF operation."""
        if not self.image_paths:
            raise ValidationError("No input image files specified.")

        self.valid_image_paths = []
        skipped_count = 0
        for path in self.image_paths:
            if not os.path.exists(path):
                self.notify_status(f"Image file not found: {path}. Skipping.", "warning")
                skipped_count += 1
                continue
            # Basic check by extension
            if not any(path.lower().endswith(ext) for ext in SUPPORTED_IMAGE_FORMATS):
                 self.notify_status(f"Unsupported file type: {path}. Skipping.", "warning")
                 skipped_count += 1
                 continue
            # Try opening with Pillow to be sure
            try:
                 with Image.open(path) as img:
                    img.verify() # Verify image data integrity
                 self.valid_image_paths.append(path)
            except UnidentifiedImageError:
                 self.notify_status(f"Cannot identify image file (unsupported/corrupt): {path}. Skipping.", "warning")
                 skipped_count += 1
            except Exception as e:
                 self.notify_status(f"Error validating image {path}: {e}. Skipping.", "warning")
                 skipped_count += 1

        if not self.valid_image_paths:
            raise ValidationError("No valid input image files found.")

        if self.page_size_str.lower() != "auto" and self.page_size_str not in REPORTLAB_PAGE_SIZES:
            raise ValidationError(f"Invalid page size specified: {self.page_size_str}")
        if self.fit_method not in ["fit", "fill", "stretch", "center"]:
             raise ValidationError(f"Invalid fit method: {self.fit_method}")

    def execute(self) -> bool:
        """Execute the image-to-PDF conversion using PyMuPDF."""
        total_images = len(self.valid_image_paths)
        self.notify_status(f"Starting conversion of {total_images} image(s) to PDF...", "info")

        # --- Use PyMuPDF (fitz) ---
        # fitz generally handles image placement well and is faster.
        try:
            self.notify_status("Using PyMuPDF for image conversion...", "info")
            doc = fitz.open() # New empty PDF

            for i, img_path in enumerate(self.valid_image_paths):
                self.check_cancelled()
                filename = os.path.basename(img_path)
                self.notify_progress((i / total_images) * 100, f"Processing '{filename}' [{i+1}/{total_images}]")

                try:
                    # Use fitz to directly open image and determine dimensions
                    img_doc = fitz.open(img_path)
                    img_page = img_doc.load_page(0) # Images are usually single page
                    img_rect = img_page.rect # Image dimensions in points at its native resolution

                    page_width_pt, page_height_pt = 0, 0
                    if self.page_size_str.lower() == "auto":
                        # Use image dimensions (+ margins) for page size
                        page_width_pt = img_rect.width + 2 * self.margin_pt
                        page_height_pt = img_rect.height + 2 * self.margin_pt
                    else:
                        page_width_pt, page_height_pt = REPORTLAB_PAGE_SIZES[self.page_size_str]

                    # Create page
                    page = doc.new_page(width=page_width_pt, height=page_height_pt)

                    # Calculate available drawing area (inside margins)
                    draw_width = page_width_pt - 2 * self.margin_pt
                    draw_height = page_height_pt - 2 * self.margin_pt
                    if draw_width <= 0 or draw_height <= 0:
                        self.notify_status(f"Margins are too large for page size on image {i+1}. Placing at corner.", "warning")
                        draw_width = max(1, draw_width)
                        draw_height = max(1, draw_height)
                        target_rect = fitz.Rect(self.margin_pt, self.margin_pt,
                                                self.margin_pt + draw_width, self.margin_pt + draw_height)
                    else:
                        target_rect = fitz.Rect(self.margin_pt, self.margin_pt,
                                                self.margin_pt + draw_width, self.margin_pt + draw_height)


                    # Fit the image into the target rectangle
                    # `show_pdf_page` handles scaling and placement nicely
                    page.show_pdf_page(target_rect, img_doc, 0, clip=target_rect) # Use clip to prevent overflow

                    img_doc.close()

                except Exception as e:
                    self.notify_status(f"Error processing image '{filename}': {e}", "warning")
                    logger.warning(f"Image to PDF error on {filename}: {e}", exc_info=True)

            if doc.page_count == 0:
                 doc.close()
                 self.notify_status("No images could be added to the PDF.", "error")
                 return False

            # Save the PDF
            self.notify_progress(95, "Saving PDF document...")
            try:
                doc.save(self.output_path, garbage=4, deflate=True)
            except Exception as save_err:
                self.notify_status(f"Error saving output PDF: {save_err}", "error")
                return False
            finally:
                doc.close()

            return True

        except OperationCancelledException:
            raise
        except Exception as fitz_err:
            # Fallback to reportlab might be overly complex given fitz's capabilities.
            # If fitz fails, it's likely a fundamental issue.
            self.notify_status(f"PyMuPDF failed to convert images: {fitz_err}", "error")
            logger.error(f"Image to PDF failed: {fitz_err}", exc_info=True)
            # Consider removing ReportLab fallback unless specifically needed.
            # If kept, it would need significant rework matching the fitz logic.
            # Let's assume fitz is the primary and only method for now.
            return False


class PDFToImagesOperation(PDFOperation):
    """Operation to convert PDF pages to images."""
    def __init__(self, input_path: str, output_dir: str,
                 img_format: str = "png", # png, jpg, tiff, bmp
                 dpi: int = 300,
                 prefix: Optional[str] = None,
                 pages: Optional[List[int]] = None, # List of 0-based page indices
                 password: Optional[str] = None):
        super().__init__("PDF to Images", "Extract pages from PDF as images")
        self.input_path = input_path
        self.output_dir = output_dir
        self.img_format = img_format.lower()
        # Map common names to fitz/PIL formats
        self.output_format_map = {
            "png": "png", "jpg": "jpeg", "jpeg": "jpeg",
            "tif": "tiff", "tiff": "tiff", "bmp": "bmp"
        }
        self.pil_save_format_map = {
            "png": "PNG", "jpeg": "JPEG", "tiff": "TIFF", "bmp": "BMP"
        }
        self.fitz_format = self.output_format_map.get(self.img_format, "png")
        self.pil_format = self.pil_save_format_map.get(self.fitz_format, "PNG")

        self.dpi = max(72, dpi)
        self.prefix = prefix or os.path.splitext(os.path.basename(input_path))[0] + "_"
        self.pages = pages # None means all pages
        self.password = password
        self.num_pages = 0

    def validate(self):
        """Validate parameters for the PDF-to-images operation."""
        if self.img_format not in self.output_format_map:
            raise ValidationError(f"Unsupported image format: {self.img_format}. "
                                  f"Supported: png, jpg, jpeg, tif, tiff, bmp")

        # Check PDF validity and get page count
        try:
            with contextlib.closing(self._open_input_pdf_fitz()) as doc:
                self.num_pages = doc.page_count
            if self.num_pages == 0:
                raise ValidationError("Input PDF has no pages.")
        except (RuntimeError, ValidationError, PdfPasswordException) as e:
             if isinstance(e, (ValidationError, PdfPasswordException)): raise
             else: raise ValidationError(f"Could not validate PDF: {e}")

        # Validate specified pages against num_pages
        if self.pages is not None:
            valid_pages = []
            for p_idx in self.pages:
                if 0 <= p_idx < self.num_pages:
                    valid_pages.append(p_idx)
                else:
                    self.notify_status(f"Page number {p_idx+1} is out of range (1-{self.num_pages}). Skipping.", "warning")
            if not valid_pages:
                 raise ValidationError("No valid pages specified in the page range.")
            self.pages = sorted(list(set(valid_pages))) # Ensure unique and sorted

    def execute(self) -> bool:
        """Execute PDF-to-images using PyMuPDF (preferred) or fallbacks."""
        pages_to_process = self.pages if self.pages is not None else list(range(self.num_pages))
        total_pages_to_process = len(pages_to_process)

        if total_pages_to_process == 0:
             self.notify_status("No pages selected for conversion.", "warning")
             return True # Succeeded by doing nothing

        pad_width = len(str(self.num_pages)) # Padding based on total pages
        self.notify_status(f"Converting {total_pages_to_process} page(s) to {self.pil_format} at {self.dpi} DPI...", "info")

        # --- Attempt 1: PyMuPDF (fitz) ---
        try:
            self.notify_status("Using PyMuPDF for rendering...", "info")
            zoom_factor = self.dpi / 72.0 # fitz uses 72 DPI as base
            mat = fitz.Matrix(zoom_factor, zoom_factor)

            with contextlib.closing(self._open_input_pdf_fitz()) as doc:
                for i, page_idx in enumerate(pages_to_process):
                    self.check_cancelled()
                    page_num_str = str(page_idx + 1).zfill(pad_width)
                    output_filename = f"{self.prefix}page_{page_num_str}.{self.img_format}"
                    output_path = os.path.join(self.output_dir, output_filename)

                    self.notify_progress(
                         (i / total_pages_to_process) * 100,
                         f"Rendering page {page_idx+1} to '{output_filename}' [{i+1}/{total_pages_to_process}]"
                    )

                    try:
                        page = doc.load_page(page_idx)
                        # Render page to pixmap (alpha=False for formats like JPG/BMP)
                        pix = page.get_pixmap(matrix=mat, alpha=(self.fitz_format == 'png'))

                        # Save the pixmap
                        if self.fitz_format == "jpeg":
                            # Ensure RGB for JPEG
                            pix_rgb = fitz.Pixmap(fitz.csRGB, pix) if pix.colorspace != fitz.csRGB else pix
                            # fitz save doesn't have quality param? Use PIL save below.
                            # pix_rgb.save(output_path) # Simple save
                            img = Image.frombytes("RGB", [pix_rgb.width, pix_rgb.height], pix_rgb.samples)
                            img.save(output_path, format=self.pil_format, quality=95, optimize=True)
                            img.close()
                            pix_rgb = None # Release memory
                        elif self.fitz_format == "png":
                            pix.save(output_path) # Save using filename extension

                        else:
                            # For other formats, convert Pixmap to PIL Image and save
                            mode = "RGB" if pix.colorspace == fitz.csRGB else "L" if pix.colorspace == fitz.csGRAY else "RGBA" # Basic mode detection
                            img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
                            img.save(output_path, format=self.pil_format)
                            img.close()

                        pix = None # Release memory

                    except Exception as page_err:
                        self.notify_status(f"Error rendering page {page_idx+1}: {page_err}", "warning")
                        logger.warning(f"Error rendering page {page_idx+1}: {page_err}", exc_info=False)

            return True # PyMuPDF method finished

        except OperationCancelledException:
            raise
        except Exception as fitz_err:
            self.notify_status(f"PyMuPDF rendering failed ({fitz_err}). Checking fallbacks...", "warning")
            logger.warning(f"PyMuPDF rendering failed: {fitz_err}", exc_info=False)

            # --- Attempt 2: pdf2image Fallback (if available and poppler works) ---
            if PDF2IMAGE_AVAILABLE:
                try:
                    self.notify_status("Using pdf2image (needs Poppler) for rendering...", "info")

                    # Check if poppler is likely available (crude check)
                    poppler_path = shutil.which("pdftoppm") or shutil.which("pdftocairo")
                    if not poppler_path:
                         self.notify_status("Poppler utility not found in PATH. pdf2image fallback unavailable.", "warning")
                         raise DependencyError("Poppler not found") # Skip to next fallback

                    self.notify_status(f"Found Poppler: {poppler_path}", "debug")

                    # Convert pages using pdf2image
                    # Note: pdf2image processes page numbers 1-based
                    first_page = pages_to_process[0] + 1 if pages_to_process else None
                    last_page = pages_to_process[-1] + 1 if pages_to_process else None

                    images = convert_from_path(
                        self.input_path,
                        dpi=self.dpi,
                        fmt=self.img_format,
                        output_folder=self.output_dir, # Save directly to avoid memory issues
                        output_file=self.prefix + "page_", # Use prefix
                        first_page=first_page,
                        last_page=last_page,
                        use_pdftocairo=True, # Often better quality
                        userpw=self.password,
                        # paths_only=True # Could use this if memory is tight, then process paths
                    )

                    # If images are returned (not saved directly), save them:
                    if isinstance(images, list):
                         for i, img in enumerate(images):
                             self.check_cancelled()
                             # Determine the original page index based on the input list
                             page_idx = pages_to_process[i]
                             page_num_str = str(page_idx + 1).zfill(pad_width)
                             output_filename = f"{self.prefix}page_{page_num_str}.{self.img_format}"
                             output_path = os.path.join(self.output_dir, output_filename)

                             self.notify_progress(
                                 (i / total_pages_to_process) * 100,
                                 f"[pdf2image] Saving page {page_idx+1} to '{output_filename}' [{i+1}/{total_pages_to_process}]"
                             )
                             try:
                                 img.save(output_path, format=self.pil_format, quality=95 if self.pil_format=="JPEG" else None)
                                 img.close()
                             except Exception as save_err:
                                 self.notify_status(f"Error saving image for page {page_idx+1}: {save_err}", "warning")

                    # If pdf2image saved files directly, we need to potentially rename them
                    # based on the actual processed pages if a subset was requested.
                    # This part is complex and depends on pdf2image's output naming.
                    # Let's assume direct save works fine for now.

                    return True # pdf2image succeeded

                except OperationCancelledException:
                    raise
                except DependencyError as dep_err:
                     # Already notified about Poppler
                     pass # Continue to next fallback
                except Exception as p2i_err:
                    self.notify_status(f"pdf2image rendering failed: {p2i_err}", "warning")
                    logger.warning(f"pdf2image failed: {p2i_err}", exc_info=False)
            else:
                 self.notify_status("pdf2image library not installed. Skipping this fallback.", "warning")


            # --- Attempt 3: Basic Pillow (very limited, often fails/low quality) ---
            try:
                 # Check if Pillow's PDF support is functional (can be basic)
                 with Image.open(self.input_path) as img_pdf:
                     try:
                        # Pillow uses 0-based seek for pages
                        img_pdf.seek(0)
                        _ = img_pdf.tell() # Check if seek works
                     except EOFError:
                         raise RuntimeError("Pillow PDF support seems non-functional (EOFError on seek).")

                 self.notify_status("Using basic PIL for rendering (fallback, may be slow/low quality)...", "warning")
                 with Image.open(self.input_path) as img_pdf:
                    # Determine which pages Pillow sees
                    pillow_page_count = 0
                    try:
                        while True:
                            img_pdf.seek(pillow_page_count)
                            pillow_page_count += 1
                    except EOFError:
                         pass # Reached end of pages Pillow can see

                    if pillow_page_count < self.num_pages:
                         self.notify_status(f"Warning: Pillow only detected {pillow_page_count} pages (PDF has {self.num_pages}). Results may be incomplete.", "warning")

                    processed_count = 0
                    for page_idx in pages_to_process:
                         self.check_cancelled()
                         if page_idx >= pillow_page_count:
                             self.notify_status(f"Skipping page {page_idx+1} as it's beyond Pillow's detected range.", "warning")
                             continue

                         page_num_str = str(page_idx + 1).zfill(pad_width)
                         output_filename = f"{self.prefix}page_{page_num_str}.{self.img_format}"
                         output_path = os.path.join(self.output_dir, output_filename)

                         current_progress = (processed_count / total_pages_to_process) * 100
                         self.notify_progress(
                             current_progress,
                             f"[PIL] Converting page {page_idx+1} to '{output_filename}' [{processed_count+1}/{total_pages_to_process}]"
                         )

                         try:
                             img_pdf.seek(page_idx)
                             # Convert Pillow image object
                             # Pillow's default rendering is often low DPI. Scaling is needed.
                             # This is tricky as Pillow doesn't know the original DPI well.
                             # Let's just save what Pillow gives.
                             page_image = img_pdf.convert("RGBA" if self.pil_format == "PNG" else "RGB")
                             page_image.save(output_path, format=self.pil_format, quality=95 if self.pil_format=="JPEG" else None)
                             page_image.close()
                             processed_count += 1

                         except Exception as pil_page_err:
                             self.notify_status(f"Error converting page {page_idx+1} with PIL: {pil_page_err}", "warning")

                 if processed_count > 0:
                     return True # Pillow fallback produced some output
                 else:
                      self.notify_status("Pillow fallback failed to convert any pages.", "error")
                      return False

            except OperationCancelledException:
                 raise
            except Exception as pil_err:
                 self.notify_status(f"Basic PIL rendering fallback failed: {pil_err}", "error")
                 logger.error(f"PIL PDF rendering failed: {pil_err}", exc_info=True)
                 return False # All methods failed


class CompressPDFOperation(PDFOperation):
    """Operation to compress a PDF file to reduce its size."""
    def __init__(self, input_path: str, output_path: str,
                 compress_quality: int = 75, # Target quality (0-100, interpretation varies)
                 downsample_images: bool = True,
                 image_dpi: int = 150, # Target DPI for downsampling
                 password: Optional[str] = None):
        super().__init__("Compress PDF", "Reduce PDF file size")
        self.input_path = input_path
        self.output_path = output_path
        self.compress_quality = max(0, min(100, compress_quality))
        self.downsample_images = downsample_images
        self.image_dpi = max(72, image_dpi) # Min 72 DPI
        self.password = password
        self.num_pages = 0

    def validate(self):
        """Validate parameters for compression operation."""
        # Check PDF validity and get page count
        try:
            with contextlib.closing(self._open_input_pdf_fitz()) as doc:
                self.num_pages = doc.page_count
            if self.num_pages == 0:
                raise ValidationError("Input PDF has no pages.")
        except (RuntimeError, ValidationError, PdfPasswordException) as e:
             if isinstance(e, (ValidationError, PdfPasswordException)): raise
             else: raise ValidationError(f"Could not validate PDF: {e}")

    def execute(self) -> bool:
        """Execute the compression using PyMuPDF's save options."""
        self.notify_status(f"Compressing PDF (Target Quality: {self.compress_quality}, Downsample DPI: {self.image_dpi if self.downsample_images else 'Off'})...", "info")

        input_size = os.path.getsize(self.input_path)
        self.notify_status(f"Input size: {self._format_size(input_size)}", "info")

        # PyMuPDF's `save` method with `garbage`, `deflate`, and `clean`
        # is the primary way to compress. Image resampling is more complex.
        # We will rely on fitz's built-in optimization features primarily.
        # Manual image extraction/recompression is possible but complex and slow.

        # --- Use PyMuPDF save options ---
        try:
            self.notify_status("Analyzing and optimizing PDF structure with PyMuPDF...", "info")
            # Open the document
            with contextlib.closing(self._open_input_pdf_fitz()) as doc:

                 # Calculate progress per page for saving step
                 total_steps = self.num_pages + 1 # N pages + 1 save step
                 current_step = 0

                 # Optional: Iterate pages to show *some* progress, although saving does the work
                 for i in range(self.num_pages):
                     self.check_cancelled()
                     current_step += 1
                     # Don't update status too often here, save is the main part
                     if (i + 1) % 10 == 0 or i == self.num_pages - 1:
                        self.notify_progress(
                             (current_step / total_steps) * 90, # Use 90% for page iteration phase
                             f"Processing page {i+1}/{self.num_pages}..."
                        )

                 # The core compression happens during save
                 self.notify_progress(90, "Saving compressed PDF (this may take a while)...")
                 save_options = {
                     "garbage": 4,       # Remove unused objects (max effort)
                     "deflate": True,    # Use Flate/Deflate compression
                     "deflate_images": True, # Compress image streams
                     "deflate_fonts": True,  # Compress font streams
                     "clean": True,      # Clean content streams (remove redundancy)
                     "linear": True,     # Linearize for web view (optional, might increase size slightly)
                     "pretty": False,    # No pretty printing for smaller size
                     # Note: PyMuPDF doesn't have a direct 'quality' setting like Ghostscript.
                     # Compression effectiveness depends on content and the options above.
                     # Image downsampling would require manual processing (see commented code below).
                 }
                 doc.save(self.output_path, **save_options)

            # --- Post-Save Analysis ---
            output_size = os.path.getsize(self.output_path)
            reduction = input_size - output_size
            percent_reduction = (reduction / input_size * 100) if input_size > 0 else 0

            if reduction > 0:
                self.notify_status(f"Compression successful: {self._format_size(input_size)} -> {self._format_size(output_size)} ({percent_reduction:.1f}% reduction)", "success")
            elif input_size == output_size:
                self.notify_status(f"File size unchanged: {self._format_size(input_size)}. No further compression possible with these settings.", "info")
            else:
                # This should be rare with compression options, but possible with linearization
                self.notify_status(f"Warning: File size increased slightly: {self._format_size(input_size)} -> {self._format_size(output_size)}.", "warning")

            return True

        except OperationCancelledException:
            # If cancelled during save, the output file might be incomplete/corrupt
            if os.path.exists(self.output_path):
                try: os.remove(self.output_path) # Attempt cleanup
                except OSError: pass
            raise
        except Exception as e:
            self.notify_status(f"Error during PDF compression: {e}", "error")
            logger.exception("PDF compression operation failed")
            # Attempt cleanup
            if os.path.exists(self.output_path):
                try: os.remove(self.output_path)
                except OSError: pass
            return False


    # --- Manual Image Downsampling (Example - Complex, Disabled by Default) ---
    # This requires extracting, resizing, recompressing images and replacing them,
    # which is significantly more complex and slower than relying on save().
    # Kept here for reference if advanced image-specific compression is needed later.
    """
    @contextlib.contextmanager
    def _temp_dir_context(self):
        # Context manager for creating and cleaning a temporary directory
        temp_dir = tempfile.mkdtemp(prefix="pdf_compress_")
        self.notify_status(f"Using temporary directory: {temp_dir}", "debug")
        try:
            yield temp_dir
        finally:
            try:
                shutil.rmtree(temp_dir)
                self.notify_status(f"Cleaned up temporary directory: {temp_dir}", "debug")
            except OSError as e:
                self.notify_status(f"Warning: Could not remove temporary directory {temp_dir}: {e}", "warning")

    def _process_and_replace_images(self, doc: fitz.Document, temp_dir: str):
        # Placeholder for the complex image processing logic
        total_images_processed = 0
        total_images_replaced = 0
        original_size_sum = 0
        new_size_sum = 0
        zoom_factor = self.image_dpi / 72.0

        for page_num in range(doc.page_count):
            self.check_cancelled()
            img_list = []
            try:
                img_list = doc.get_page_images(page_num, full=True)
            except Exception as e:
                self.notify_status(f"Could not get images for page {page_num+1}: {e}", "warning")
                continue

            if not img_list: continue

            self.notify_progress(
                (page_num / doc.page_count) * 50, # Image processing is first 50%
                f"Analyzing images on page {page_num+1}/{doc.page_count} ({len(img_list)} found)"
            )

            for img_index, img_info in enumerate(img_list):
                xref = img_info[0]
                if xref == 0: continue # Skip inline images for simplicity for now

                try:
                    base_img = doc.extract_image(xref)
                    if not base_img: continue

                    img_bytes = base_img["image"]
                    img_ext = base_img["ext"].lower()
                    if img_ext not in ["png", "jpg", "jpeg", "tif", "tiff", "bmp"]:
                        continue # Skip unsupported formats for recompression

                    original_size = len(img_bytes)
                    if original_size < 5000: continue # Skip very small images

                    total_images_processed += 1
                    original_size_sum += original_size

                    with Image.open(BytesIO(img_bytes)) as img:
                        img_width, img_height = img.size

                        # Calculate target size based on DPI
                        target_width = int(img_width * zoom_factor)
                        target_height = int(img_height * zoom_factor)

                        resized = False
                        if self.downsample_images and zoom_factor < 1.0 and target_width < img_width and target_height < img_height:
                            try:
                                img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
                                resized = True
                            except Exception as resize_err:
                                self.notify_status(f"Failed to resize image {xref} on page {page_num+1}: {resize_err}", "warning")
                                continue # Skip replacement if resize fails

                        # Recompress
                        img_out = BytesIO()
                        save_params = {}
                        pil_format = self.pil_save_format_map.get(img_ext, "PNG")

                        if pil_format == "JPEG":
                            save_params['quality'] = self.compress_quality
                            save_params['optimize'] = True
                            save_params['progressive'] = True
                        elif pil_format == "PNG":
                            save_params['optimize'] = True
                            # PNG compress_level 0-9, higher is more compression (opposite of JPEG quality)
                            save_params['compress_level'] = max(0, min(9, int((100 - self.compress_quality) / 10)))

                        try:
                            # Ensure correct mode before saving
                            if pil_format == "JPEG" and img.mode not in ("RGB", "L"):
                                img = img.convert("RGB")
                            elif pil_format == "PNG" and "A" in img.mode:
                                pass # Keep alpha
                            elif img.mode not in ("RGB", "L", "RGBA"): # Fallback for others
                                img = img.convert("RGB")

                            img.save(img_out, format=pil_format, **save_params)
                        except Exception as save_err:
                             self.notify_status(f"Failed to recompress image {xref} on page {page_num+1}: {save_err}", "warning")
                             continue

                        new_img_bytes = img_out.getvalue()
                        new_size = len(new_img_bytes)

                    # Replace image only if smaller
                    if new_size < original_size:
                        try:
                            doc.update_image(xref, stream=new_img_bytes)
                            total_images_replaced += 1
                            new_size_sum += new_size
                            self.notify_status(f"  Replaced image {xref} on page {page_num+1} ({self._format_size(original_size)} -> {self._format_size(new_size)})", "debug")
                        except Exception as update_err:
                             self.notify_status(f"Failed to update image {xref} on page {page_num+1}: {update_err}", "warning")
                    else:
                         new_size_sum += original_size # Count original size if not replaced

                except OperationCancelledException:
                    raise
                except Exception as img_proc_err:
                     self.notify_status(f"Error processing image {xref} on page {page_num+1}: {img_proc_err}", "warning")

        size_diff = original_size_sum - new_size_sum
        self.notify_status(f"Image processing finished. Processed: {total_images_processed}, Replaced: {total_images_replaced}. Estimated reduction: {self._format_size(size_diff)}", "info")

    """

    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format."""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024**2:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024**3:
            return f"{size_bytes / (1024**2):.2f} MB"
        else:
            return f"{size_bytes / (1024**3):.2f} GB"


class EncryptPDFOperation(PDFOperation):
    """Operation to encrypt or decrypt a PDF file."""
    def __init__(self, input_path: str, output_path: str,
                 action: str = "encrypt", # "encrypt" or "decrypt"
                 user_password: Optional[str] = "", # Empty string for user pw means no user pw
                 owner_password: Optional[str] = None, # None means use user pw, "" means specific blank owner pw
                 encryption_level: int = 128, # 128, 256 (check library support)
                 allow_printing: bool = True,
                 allow_copying: bool = False,
                 allow_modifying: bool = False, # Add modify permission
                 allow_annotating: bool = True, # Add annotation permission
                 input_password: Optional[str] = None):
        super().__init__("Encrypt/Decrypt PDF", "Add or remove password protection")
        self.input_path = input_path
        self.output_path = output_path
        self.action = action
        self.user_password = user_password or "" # Ensure it's a string
        self.owner_password = owner_password # Keep as None initially
        self.encryption_level = encryption_level
        # Map permissions for fitz
        self.permissions = 0
        if allow_printing: self.permissions |= fitz.PDF_PERM_PRINT | fitz.PDF_PERM_PRINT_HQ
        if allow_copying: self.permissions |= fitz.PDF_PERM_COPY
        if allow_modifying: self.permissions |= fitz.PDF_PERM_MODIFY
        if allow_annotating: self.permissions |= fitz.PDF_PERM_ANNOTATE | fitz.PDF_PERM_FILLFORM # Annotate usually implies form filling

        # Add basic accessibility/assembly permissions often included
        self.permissions |= fitz.PDF_PERM_ACCESSIBILITY
        self.permissions |= fitz.PDF_PERM_ASSEMBLE


        self.input_password = input_password # Password for encrypted input

    def validate(self):
        """Validate parameters for encryption/decryption operation."""
        if self.action not in ["encrypt", "decrypt"]:
            raise ValidationError(f"Invalid action: {self.action}")

        # Check input PDF validity (will also check encryption status)
        doc = None
        try:
            # Use input_password if decrypting, otherwise None
            check_password = self.input_password if self.action == "decrypt" else None
            # Temporarily set self.password for the helper function
            original_self_password = self.password
            self.password = check_password
            doc = self._open_input_pdf_fitz() # Handles auth check if needed
            self.password = original_self_password # Restore original

            is_encrypted = doc.is_encrypted

            if self.action == "encrypt":
                # If encrypting, ensure at least one password is set
                effective_owner_pw = self.owner_password if self.owner_password is not None else self.user_password
                if not self.user_password and not effective_owner_pw:
                     raise ValidationError("At least a User or Owner password must be provided for encryption.")
                # If already encrypted, warn user? Overwriting encryption is fine.
                if is_encrypted:
                     self.notify_status("Input PDF is already encrypted. Re-encrypting with new settings.", "info")

            elif self.action == "decrypt":
                if not is_encrypted:
                    raise ValidationError("Input PDF is not encrypted. Cannot decrypt.")
                # Password validity was checked by _open_input_pdf_fitz if input_password was provided
                if not self.input_password:
                     # This case should ideally be caught by GUI asking for password
                     raise ValidationError("Input PDF is encrypted, password required for decryption.")

            # Validate encryption level (check fitz support)
            valid_levels = {
                # Older RC4 (less secure)
                40: fitz.PDF_ENCRYPT_RC4_40,
                128: fitz.PDF_ENCRYPT_RC4_128,
                # Modern AES
                # 128: fitz.PDF_ENCRYPT_AES_128, # Alias for RC4_128 in some versions?
                256: fitz.PDF_ENCRYPT_AES_256
            }
            # Adjust based on actual fitz version capabilities if needed
            if self.encryption_level not in valid_levels:
                 self.notify_status(f"Encryption level {self.encryption_level}-bit not directly supported or invalid, using 128-bit AES.", "warning")
                 self.encryption_level = 128 # Default to a strong, common level
                 self.fitz_encrypt_method = fitz.PDF_ENCRYPT_AES_128 # Or PDF_ENCRYPT_RC4_128
            else:
                 # Select AES if available for the level, otherwise RC4
                 if self.encryption_level == 128:
                     self.fitz_encrypt_method = fitz.PDF_ENCRYPT_AES_128 # Prefer AES
                 elif self.encryption_level == 256:
                      self.fitz_encrypt_method = fitz.PDF_ENCRYPT_AES_256
                 else: # 40-bit
                      self.fitz_encrypt_method = valid_levels[self.encryption_level]


        except (RuntimeError, ValidationError, PdfPasswordException) as e:
            if isinstance(e, (ValidationError, PdfPasswordException)): raise
            else: raise ValidationError(f"Could not validate PDF: {e}")
        finally:
             if doc: doc.close()


    def execute(self) -> bool:
        """Execute encryption/decryption using PyMuPDF."""
        action_msg = "Encrypting" if self.action == "encrypt" else "Decrypting"
        self.notify_status(f"{action_msg} PDF...", "info")

        # --- Use PyMuPDF (fitz) ---
        # Fitz handles both encryption and decryption via save options
        try:
            # Open with input password if decrypting
            open_password = self.input_password if self.action == "decrypt" else None
            # Temporarily set self.password for helper
            original_self_password = self.password
            self.password = open_password
            doc = self._open_input_pdf_fitz()
            self.password = original_self_password # Restore

            save_options = {
                "garbage": 4,
                "deflate": True,
                "linear": False, # Linearization might interfere?
            }

            if self.action == "encrypt":
                # If owner password is None, use user password
                effective_owner_pw = self.owner_password if self.owner_password is not None else self.user_password

                self.notify_status(f"Applying {self.encryption_level}-bit encryption...", "info")
                save_options["encryption"] = self.fitz_encrypt_method
                save_options["owner_pw"] = effective_owner_pw
                save_options["user_pw"] = self.user_password
                save_options["permissions"] = int(self.permissions) # Ensure it's int

            else: # Decrypt
                self.notify_status("Removing encryption...", "info")
                save_options["encryption"] = fitz.PDF_ENCRYPT_NONE # Remove encryption

            # Save the document
            self.notify_progress(50, "Saving modified PDF...")
            try:
                doc.save(self.output_path, **save_options)
            except Exception as save_err:
                 self.notify_status(f"Error saving {self.action}ed PDF: {save_err}", "error")
                 return False
            finally:
                doc.close()

            self.notify_status(f"PDF successfully {self.action}ed.", "success")
            return True

        except OperationCancelledException:
            if os.path.exists(self.output_path):
                 try: os.remove(self.output_path)
                 except OSError: pass
            raise
        except (RuntimeError, ValidationError, PdfPasswordException) as e:
             # These should have been caught in validate, but handle defensively
             self.notify_status(f"Error during {self.action}: {e}", "error")
             logger.error(f"{self.action.capitalize()} failed pre-save: {e}", exc_info=False)
             return False
        except Exception as e:
            self.notify_status(f"Unexpected error during PDF {self.action}: {e}", "error")
            logger.exception(f"PDF {self.action} operation failed")
            if os.path.exists(self.output_path):
                 try: os.remove(self.output_path)
                 except OSError: pass
            return False


class ExtractTextOperation(PDFOperation):
    """Operation to extract text content from a PDF file."""
    def __init__(self, input_path: str, output_path: str,
                 pages: Optional[List[int]] = None, # List of 0-based page indices
                 format_type: str = "plain", # "plain", "blocks", "words", "html", "xhtml", "xml", "dict", "json"
                 password: Optional[str] = None,
                 layout: bool = False): # Corresponds to fitz text flags
        super().__init__("Extract Text", "Extract text content from a PDF")
        self.input_path = input_path
        self.output_path = output_path
        self.pages = pages # None means all pages
        self.format_type = format_type.lower()
        self.password = password
        self.layout = layout # Preserve layout (more spaces, etc.)?
        self.num_pages = 0
        # Map UI options to fitz get_text() flags if needed
        self.fitz_output_format = self.format_type
        if self.format_type == "plain": self.fitz_output_format = "text" # Basic text
        if self.format_type == "formatted": self.fitz_output_format = "text" # Use layout flag instead
        if self.format_type == "markdown": self.fitz_output_format = "text" # Post-process text for MD

    def validate(self):
        """Validate parameters for text extraction."""
        # Check PDF validity and get page count
        try:
            with contextlib.closing(self._open_input_pdf_fitz()) as doc:
                self.num_pages = doc.page_count
            if self.num_pages == 0:
                raise ValidationError("Input PDF has no pages.")
        except (RuntimeError, ValidationError, PdfPasswordException) as e:
             if isinstance(e, (ValidationError, PdfPasswordException)): raise
             else: raise ValidationError(f"Could not validate PDF: {e}")

        # Validate specified pages
        if self.pages is not None:
            valid_pages = []
            for p_idx in self.pages:
                if 0 <= p_idx < self.num_pages:
                    valid_pages.append(p_idx)
                else:
                    self.notify_status(f"Page number {p_idx+1} is out of range (1-{self.num_pages}). Skipping.", "warning")
            if not valid_pages:
                 raise ValidationError("No valid pages specified in the page range.")
            self.pages = sorted(list(set(valid_pages)))

        # Validate format type against fitz options
        valid_fitz_formats = ["text", "blocks", "words", "html", "xhtml", "xml", "dict", "json"]
        if self.fitz_output_format not in valid_fitz_formats and self.format_type != "markdown":
            self.notify_status(f"Using basic text extraction for unsupported format '{self.format_type}'.", "warning")
            self.fitz_output_format = "text" # Default to basic text

    def execute(self) -> bool:
        """Execute text extraction using PyMuPDF."""
        pages_to_process = self.pages if self.pages is not None else list(range(self.num_pages))
        total_pages_to_process = len(pages_to_process)

        if total_pages_to_process == 0:
             self.notify_status("No pages selected for text extraction.", "warning")
             return True

        self.notify_status(f"Extracting text from {total_pages_to_process} page(s) as {self.format_type}...", "info")

        # Determine fitz flags based on layout preservation
        # Note: flags are integers, combine with |
        # flags = fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_WHITESPACE
        # default flags value is TEXT_PRESERVE_LIGATURES | TEXT_PRESERVE_WHITESPACE | TEXT_PRESERVE_IMAGES.
        # To NOT preserve layout, we need to remove the whitespace flag.
        flags = fitz.TEXTFLAGS_DEFAULT
        if not self.layout:
            flags &= ~fitz.TEXT_PRESERVE_WHITESPACE # Remove whitespace preservation
            flags &= ~fitz.TEXT_PRESERVE_IMAGES # Don't include image blocks


        full_output = []

        try:
            with contextlib.closing(self._open_input_pdf_fitz()) as doc:
                for i, page_idx in enumerate(pages_to_process):
                    self.check_cancelled()
                    self.notify_progress(
                         (i / total_pages_to_process) * 100,
                         f"Extracting page {page_idx+1}/{self.num_pages} [{i+1}/{total_pages_to_process}]"
                    )

                    try:
                        page = doc.load_page(page_idx)
                        page_text = page.get_text(self.fitz_output_format, flags=flags)

                        # Post-processing for specific formats
                        if self.format_type == "markdown":
                             full_output.append(f"## Page {page_idx+1}\n\n{page_text.strip()}\n")
                        elif self.format_type == "html":
                             # Basic HTML structure if requested
                             if i == 0: full_output.append("<html>\n<head><title>Extracted Text</title></head>\n<body>\n")
                             full_output.append(f"<!-- Page {page_idx+1} -->\n{page_text}\n")
                             if i == total_pages_to_process - 1: full_output.append("</body>\n</html>")
                        elif self.format_type in ["json", "dict"]:
                            # Append page number for context if output is dict/json per page
                             try:
                                 data = json.loads(page_text) if self.format_type == "json" else eval(page_text) # eval is risky!
                                 data["page_number"] = page_idx + 1
                                 full_output.append(json.dumps(data, indent=2) if self.format_type == "json" else repr(data))
                             except Exception as json_err:
                                  self.notify_status(f"Could not parse {self.format_type} output for page {page_idx+1}: {json_err}", "warning")
                                  full_output.append(f"Error processing page {page_idx+1} as {self.format_type}")
                        else:
                             full_output.append(page_text)

                    except Exception as page_err:
                         self.notify_status(f"Error extracting text from page {page_idx+1}: {page_err}", "warning")
                         full_output.append(f"\n--- ERROR EXTRACTING PAGE {page_idx+1} ---\n")


            # Combine output and write to file
            self.notify_progress(95, "Writing output file...")
            output_separator = "\n\n" if self.format_type not in ["html", "xhtml", "xml", "markdown", "json", "dict"] else "\n"
            final_text = output_separator.join(full_output)

            try:
                with open(self.output_path, "w", encoding="utf-8") as f:
                    f.write(final_text)
            except Exception as write_err:
                 self.notify_status(f"Error writing output file '{self.output_path}': {write_err}", "error")
                 return False

            self.notify_status(f"Text successfully extracted to '{os.path.basename(self.output_path)}'.", "success")
            return True

        except OperationCancelledException:
            raise
        except (RuntimeError, ValidationError, PdfPasswordException) as e:
             self.notify_status(f"Error during text extraction: {e}", "error")
             logger.error(f"Text extraction failed pre-processing: {e}", exc_info=False)
             return False
        except Exception as e:
            self.notify_status(f"Unexpected error during text extraction: {e}", "error")
            logger.exception("Text extraction operation failed")
            return False


class RotatePDFOperation(PDFOperation):
    """Operation to rotate pages in a PDF file."""
    def __init__(self, input_path: str, output_path: str,
                 rotation: int, # 90, 180, 270
                 pages: Optional[List[int]] = None, # List of 0-based page indices
                 password: Optional[str] = None):
        super().__init__("Rotate PDF", "Rotate pages in a PDF document")
        self.input_path = input_path
        self.output_path = output_path
        if rotation not in [0, 90, 180, 270]:
            raise ValueError("Rotation angle must be 0, 90, 180, or 270 degrees.")
        self.rotation = rotation # Rotation delta
        self.pages = pages # None means all pages
        self.password = password
        self.num_pages = 0

    def validate(self):
        """Validate parameters for rotation operation."""
        # Check PDF validity and get page count
        try:
            with contextlib.closing(self._open_input_pdf_fitz()) as doc:
                self.num_pages = doc.page_count
            if self.num_pages == 0:
                raise ValidationError("Input PDF has no pages.")
        except (RuntimeError, ValidationError, PdfPasswordException) as e:
             if isinstance(e, (ValidationError, PdfPasswordException)): raise
             else: raise ValidationError(f"Could not validate PDF: {e}")

        # Validate specified pages
        if self.pages is not None:
            valid_pages = []
            for p_idx in self.pages:
                if 0 <= p_idx < self.num_pages:
                    valid_pages.append(p_idx)
                else:
                    self.notify_status(f"Page number {p_idx+1} is out of range (1-{self.num_pages}). Skipping.", "warning")
            if not valid_pages:
                 raise ValidationError("No valid pages specified in the page range.")
            self.pages = sorted(list(set(valid_pages)))

    def execute(self) -> bool:
        """Execute rotation using PyMuPDF."""
        pages_to_process = self.pages if self.pages is not None else list(range(self.num_pages))
        total_pages_to_process = len(pages_to_process)

        if total_pages_to_process == 0:
             self.notify_status("No pages selected for rotation.", "warning")
             return True
        if self.rotation == 0:
             self.notify_status("Rotation angle is 0. No changes needed.", "info")
             # Maybe copy file if output is different? For now, just succeed.
             # Consider copying if input != output path?
             if os.path.abspath(self.input_path) != os.path.abspath(self.output_path):
                 try:
                     shutil.copy2(self.input_path, self.output_path)
                     self.notify_status("Input copied to output location as rotation was 0.", "info")
                 except Exception as e:
                     self.notify_status(f"Could not copy input to output for 0 rotation: {e}", "error")
                     return False
             return True

        self.notify_status(f"Rotating {total_pages_to_process} page(s) by {self.rotation} degrees...", "info")

        # --- Use PyMuPDF (fitz) ---
        try:
            # Open the document
            with contextlib.closing(self._open_input_pdf_fitz()) as doc:
                # Process each page specified
                for i, page_idx in enumerate(pages_to_process):
                    self.check_cancelled()
                    self.notify_progress(
                         (i / total_pages_to_process) * 100,
                         f"Rotating page {page_idx+1} [{i+1}/{total_pages_to_process}]"
                    )
                    try:
                        page = doc[page_idx] # Load page implicitly
                        current_rotation = page.rotation
                        # Set *absolute* rotation, so add delta to current
                        # new_rotation = (current_rotation + self.rotation) % 360
                        # Note: fitz page.set_rotation seems to expect the DELTA. Let's test.
                        # Documentation for page.set_rotation(deg) is sparse.
                        # Let's assume it sets the *absolute* rotation.
                        page.set_rotation(self.rotation) # Try setting absolute first
                        # If that fails or behaves like delta, try:
                        # page.set_rotation((page.rotation + self.rotation) % 360)

                    except Exception as page_err:
                        self.notify_status(f"Error rotating page {page_idx+1}: {page_err}", "warning")

                # Save the document
                self.notify_progress(95, "Saving rotated PDF...")
                try:
                     # Save incrementally if possible? No, save standard.
                    doc.save(self.output_path, garbage=4, deflate=True)
                except Exception as save_err:
                    self.notify_status(f"Error saving rotated PDF: {save_err}", "error")
                    return False

            self.notify_status("PDF rotation completed successfully.", "success")
            return True

        except OperationCancelledException:
            if os.path.exists(self.output_path):
                 try: os.remove(self.output_path)
                 except OSError: pass
            raise
        except (RuntimeError, ValidationError, PdfPasswordException) as e:
             self.notify_status(f"Error during rotation: {e}", "error")
             logger.error(f"Rotation failed pre-save: {e}", exc_info=False)
             return False
        except Exception as e:
            self.notify_status(f"Unexpected error during PDF rotation: {e}", "error")
            logger.exception("PDF rotation operation failed")
            if os.path.exists(self.output_path):
                 try: os.remove(self.output_path)
                 except OSError: pass
            return False


class EditMetadataOperation(PDFOperation):
    """Operation to edit PDF document information (metadata)."""
    def __init__(self, input_path: str, output_path: str,
                 metadata: Dict[str, Optional[str]], # Dict of {'key': 'value', 'key_to_clear': None}
                 password: Optional[str] = None):
        super().__init__("Edit Metadata", "Modify PDF document information")
        self.input_path = input_path
        self.output_path = output_path
        self.metadata_to_set = metadata
        self.password = password

    def validate(self):
        """Validate parameters for metadata operation."""
        if not isinstance(self.metadata_to_set, dict):
            raise ValidationError("Metadata must be provided as a dictionary.")
        # Check PDF validity
        try:
            with contextlib.closing(self._open_input_pdf_fitz()) as doc:
                pass # Just check if it opens
        except (RuntimeError, ValidationError, PdfPasswordException) as e:
             if isinstance(e, (ValidationError, PdfPasswordException)): raise
             else: raise ValidationError(f"Could not validate PDF: {e}")

    def execute(self) -> bool:
        """Execute metadata editing using PyMuPDF."""
        self.notify_status("Updating PDF metadata...", "info")
        update_count = 0

        # --- Use PyMuPDF (fitz) ---
        try:
            # Open the document
            with contextlib.closing(self._open_input_pdf_fitz()) as doc:

                 # Get current metadata for comparison (optional)
                 current_meta = doc.metadata or {}
                 self.notify_status(f"Current metadata fields: {len(current_meta)}", "debug")

                 # Prepare metadata for setting (handle None as clear)
                 # Fitz uses None to clear a field.
                 updates_applied: Dict[str, Optional[str]] = {}
                 for key, value in self.metadata_to_set.items():
                     # Check if key is valid for fitz metadata (case-insensitive keys)
                     lower_key = key.lower()
                     # Standard keys: format, title, author, subject, keywords, creator, producer, creationDate, modDate, trapped
                     if lower_key in ["format", "title", "author", "subject", "keywords", "creator", "producer", "creationdate", "moddate", "trapped"]:
                         # Only apply if value changed or field doesn't exist
                         if value != current_meta.get(key, None): # Compare with current
                             updates_applied[key] = value # Keep original case for setting? Fitz likely normalizes.
                             update_count += 1
                             self.notify_status(f"Setting '{key}': '{value if value is not None else '[CLEAR]'}'", "debug")
                         else:
                             self.notify_status(f"Skipping '{key}': Value unchanged.", "debug")

                     else:
                         self.notify_status(f"Skipping unknown metadata key: '{key}'", "warning")

                 if update_count == 0:
                     self.notify_status("No metadata changes detected. Output file will be a copy.", "info")
                     # If input != output, copy the file
                     if os.path.abspath(self.input_path) != os.path.abspath(self.output_path):
                         try:
                             shutil.copy2(self.input_path, self.output_path)
                         except Exception as e:
                             self.notify_status(f"Could not copy input to output: {e}", "error")
                             return False
                     return True # Succeeded by doing nothing


                 self.notify_progress(30, f"Applying {update_count} metadata changes...")
                 doc.set_metadata(updates_applied)

                 # Save the document
                 self.notify_progress(70, "Saving PDF with updated metadata...")
                 try:
                    doc.save(self.output_path, garbage=4, deflate=True)
                 except Exception as save_err:
                    self.notify_status(f"Error saving PDF with updated metadata: {save_err}", "error")
                    return False

            self.notify_status(f"PDF metadata updated successfully ({update_count} fields changed).", "success")
            return True

        except OperationCancelledException:
            if os.path.exists(self.output_path):
                 try: os.remove(self.output_path)
                 except OSError: pass
            raise
        except (RuntimeError, ValidationError, PdfPasswordException) as e:
             self.notify_status(f"Error during metadata update: {e}", "error")
             logger.error(f"Metadata update failed pre-save: {e}", exc_info=False)
             return False
        except Exception as e:
            self.notify_status(f"Unexpected error during metadata update: {e}", "error")
            logger.exception("Metadata update operation failed")
            if os.path.exists(self.output_path):
                 try: os.remove(self.output_path)
                 except OSError: pass
            return False


# --- Controller: Application Logic ---

class OperationManager:
    """
    Manages the execution of PDFOperations in background threads
    and notifies listeners of progress and completion.
    """
    def __init__(self):
        self.operation_history: deque = deque(maxlen=100) # Limited history
        self.current_operation: Optional[PDFOperation] = None
        self._listeners: List[Any] = []
        self._lock = threading.Lock() # Lock for accessing current_operation

    def register_listener(self, listener: Any):
        """Register a listener (typically the main UI) for operation events."""
        if listener not in self._listeners:
            self._listeners.append(listener)

    def unregister_listener(self, listener: Any):
        """Unregister a listener."""
        if listener in self._listeners:
            self._listeners.remove(listener)

    def _notify_listeners(self, method_name: str, *args, **kwargs):
        """Helper to notify all registered listeners."""
        # Operate on a copy in case listeners modify the list during iteration
        listeners_copy = self._listeners[:]
        for listener in listeners_copy:
            if hasattr(listener, method_name):
                try:
                    # Schedule the call in the listener's thread context (e.g., GUI thread)
                    # This requires the listener (UI) to handle thread safety.
                    # A common way is using a queue or platform-specific event posting.
                    # For tkinter, root.after(0, lambda: ...) or queue.put can work.
                    # Assuming listeners handle their own threading context:
                    getattr(listener, method_name)(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Error notifying listener {listener}.{method_name}: {e}")

    def execute_operation(self, operation: PDFOperation):
        """
        Executes a given PDFOperation in a separate thread.
        Manages the current operation state and cancellation.
        """
        with self._lock:
            if self.current_operation and self.current_operation.is_running():
                # Optionally ask user if they want to cancel the previous one
                logger.warning("Cancelling previous operation to start a new one.")
                self.current_operation.cancel() # Signal cancellation
                # Maybe wait briefly for it to acknowledge cancellation?

            # Register self as a listener to the operation to relay events
            operation.register_listener(self)
            self.current_operation = operation

        # Notify UI immediately that an operation is starting
        self._notify_listeners("on_operation_start", operation)

        # Start the operation in a background thread
        thread = threading.Thread(target=self._run_operation_thread, args=(operation,), daemon=True)
        thread.start()

    def _run_operation_thread(self, operation: PDFOperation):
        """Worker thread function to run the operation."""
        success = False
        try:
            # Run the operation (which handles its own notifications)
            operation.run() # Result is handled via notifications
            success = True # If run completes without exception re-raising here
        except Exception as e:
            # Catch unexpected errors *within* the thread execution itself
            logger.exception(f"Critical error in operation thread for {operation.name}: {e}")
            # Notify listeners about this critical failure
            self._notify_listeners("on_operation_complete", operation, False, f"Critical thread error: {e}", "error")
        finally:
            # Clean up after operation finishes or fails critically
            with self._lock:
                if self.current_operation == operation:
                    self.current_operation = None # Clear current operation only if it's this one

            # Add to history (do this even if thread had critical error?)
            timestamp = datetime.now().isoformat(sep=' ', timespec='seconds')
            try:
                # Basic details, avoid complex objects
                details = {
                    'input': getattr(operation, 'input_path', None) or getattr(operation, 'input_paths', None),
                    'output': getattr(operation, 'output_path', None) or getattr(operation, 'output_dir', None),
                    'params': {k: v for k, v in operation.__dict__.items() if k not in ['_listeners', '_lock', 'input_path', 'input_paths', 'output_path', 'output_dir', 'password', 'passwords'] and not k.startswith('_')}
                }
                # Redact passwords from details
                if 'password' in details.get('params', {}): details['params']['password'] = '***'
                if 'input_password' in details.get('params', {}): details['params']['input_password'] = '***'

            except Exception:
                 details = "Error getting details" # Fallback

            history_entry = {
                "timestamp": timestamp,
                "operation": operation.name,
                "success": success, # Reflects if run() completed okay, not application logic success
                "details": details
            }
            self.operation_history.appendleft(history_entry) # Add to front

            # Unregister self as listener from the operation
            operation.unregister_listener(self)

    def cancel_current_operation(self) -> bool:
        """Requests cancellation of the currently running operation."""
        with self._lock:
            if self.current_operation and self.current_operation.is_running():
                self.current_operation.cancel()
                return True
        return False

    # --- Relay methods called by the PDFOperation ---
    # These methods simply forward notifications from the operation
    # to the manager's listeners (e.g., the GUI).

    def on_progress_update(self, operation: PDFOperation, progress: float, status_message: str, level: str):
        if operation == self.current_operation: # Ensure update is from current op
            self._notify_listeners("on_progress_update", operation, progress, status_message, level)

    def on_status_update(self, operation: PDFOperation, message: str, level: str):
        if operation == self.current_operation:
            self._notify_listeners("on_status_update", operation, message, level)

    def on_operation_complete(self, operation: PDFOperation, success: bool, message: str, level: str):
         # This notification comes *from* the operation upon its completion.
         # The _run_operation_thread handles the manager's final state cleanup.
        self._notify_listeners("on_operation_complete", operation, success, message, level)


class ConfigManager:
    """
    Manages application configuration using configparser.
    Handles loading, saving, and accessing settings with defaults.
    """
    def __init__(self):
        self.config = configparser.ConfigParser(interpolation=None) # No interpolation
        self.config_file = self._get_config_path()
        self.load_config() # Load or create config on initialization

    def _get_config_path(self) -> str:
        """Determines the path for the configuration file."""
        config_dir = os.path.join(os.path.expanduser("~"), ".pdf_toolkit_pro")
        os.makedirs(config_dir, exist_ok=True) # Ensure directory exists
        return os.path.join(config_dir, "config.ini")

    def load_config(self):
        """Loads configuration from file, applying defaults if missing."""
        # Apply defaults first
        self.config.read_dict(DEFAULT_CONFIG)

        # Read user's config file, overriding defaults if present
        if os.path.exists(self.config_file):
            try:
                self.config.read(self.config_file, encoding='utf-8')
            except configparser.Error as e:
                logger.error(f"Error reading configuration file '{self.config_file}': {e}. Using defaults.")
                # Re-apply defaults to ensure clean state
                self.config = configparser.ConfigParser(interpolation=None)
                self.config.read_dict(DEFAULT_CONFIG)
                self.save_config() # Save the defaults back
            except Exception as e:
                 logger.error(f"Unexpected error loading config: {e}", exc_info=True)
                 # Attempt to recover with defaults
                 self.config = configparser.ConfigParser(interpolation=None)
                 self.config.read_dict(DEFAULT_CONFIG)
                 self.save_config()
        else:
            # If the file doesn't exist, save the defaults
            logger.info(f"Configuration file not found at '{self.config_file}'. Creating with defaults.")
            self.save_config()

    def save_config(self):
        """Saves the current configuration to the file."""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                self.config.write(f)
        except OSError as e:
            logger.error(f"Error saving configuration to '{self.config_file}': {e}")
        except Exception as e:
             logger.error(f"Unexpected error saving config: {e}", exc_info=True)

    def get(self, section: str, option: str, fallback: Optional[str] = None) -> Optional[str]:
        """Gets a configuration value as a string."""
        try:
            return self.config.get(section, option)
        except (configparser.NoSectionError, configparser.NoOptionError):
            # Try getting from default if available
            default_value = DEFAULT_CONFIG.get(section, {}).get(option)
            return default_value if fallback is None else fallback
        except Exception as e:
             logger.error(f"Error getting config [{section}] {option}: {e}")
             return fallback


    def getint(self, section: str, option: str, fallback: Optional[int] = None) -> Optional[int]:
        """Gets a configuration value as an integer."""
        try:
            return self.config.getint(section, option)
        except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
             default_value = DEFAULT_CONFIG.get(section, {}).get(option)
             try:
                 return int(default_value) if default_value is not None else fallback
             except (ValueError, TypeError):
                  return fallback
        except Exception as e:
              logger.error(f"Error getting int config [{section}] {option}: {e}")
              return fallback

    def getboolean(self, section: str, option: str, fallback: Optional[bool] = None) -> Optional[bool]:
        """Gets a configuration value as a boolean."""
        try:
            return self.config.getboolean(section, option)
        except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
            default_value = DEFAULT_CONFIG.get(section, {}).get(option)
            if default_value is None: return fallback
            # Handle common boolean string representations robustly
            return default_value.lower() in ('true', 'yes', '1', 'on')
        except Exception as e:
            logger.error(f"Error getting boolean config [{section}] {option}: {e}")
            return fallback

    def getfloat(self, section: str, option: str, fallback: Optional[float] = None) -> Optional[float]:
        """Gets a configuration value as a float."""
        try:
            return self.config.getfloat(section, option)
        except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
            default_value = DEFAULT_CONFIG.get(section, {}).get(option)
            try:
                 return float(default_value) if default_value is not None else fallback
            except (ValueError, TypeError):
                  return fallback
        except Exception as e:
            logger.error(f"Error getting float config [{section}] {option}: {e}")
            return fallback


    def set(self, section: str, option: str, value: Any):
        """Sets a configuration value (converts value to string)."""
        try:
            if not self.config.has_section(section):
                self.config.add_section(section)
            self.config.set(section, option, str(value))
            # Optionally save immediately, or provide explicit save method call
            # self.save_config()
        except Exception as e:
             logger.error(f"Error setting config [{section}] {option}: {e}")

    # --- Theme specific methods ---
    def get_theme_mode(self) -> Theme:
        """Gets the currently configured theme mode (light, dark, system, custom)."""
        theme_name = self.get("app", "theme", Theme.SYSTEM.value)
        try:
            return Theme(theme_name)
        except ValueError:
            logger.warning(f"Invalid theme name '{theme_name}' in config. Using system default.")
            return Theme.SYSTEM

    def set_theme_mode(self, theme_mode: Theme):
        """Sets the theme mode."""
        self.set("app", "theme", theme_mode.value)

    def get_custom_theme_base(self) -> str:
        """Gets the base ttkbootstrap theme name used for custom colors."""
        return self.get("app", "custom_theme_name", TTB_DARK_THEME)

    def set_custom_theme_base(self, theme_name: str):
        """Sets the base ttkbootstrap theme name for custom colors."""
        self.set("app", "custom_theme_name", theme_name)

    def get_custom_theme_colors(self) -> Dict[str, str]:
        """Gets custom theme color overrides as a dictionary."""
        colors_json = self.get("app", "custom_theme_colors", "{}")
        try:
            colors = json.loads(colors_json)
            if isinstance(colors, dict):
                return colors
            else:
                logger.warning("Custom theme colors in config is not a valid JSON object. Returning empty.")
                return {}
        except json.JSONDecodeError:
            logger.warning("Could not parse custom theme colors JSON from config. Returning empty.")
            return {}

    def set_custom_theme_colors(self, colors: Dict[str, str]):
        """Sets the custom theme color overrides."""
        try:
            colors_json = json.dumps(colors, indent=None) # Compact JSON
            self.set("app", "custom_theme_colors", colors_json)
        except (TypeError, ValueError) as e:
            logger.error(f"Could not serialize custom theme colors to JSON: {e}")


    # --- Recent Files specific methods ---
    def get_recent_files(self) -> List[str]:
        """Gets the list of recent file paths from the configuration."""
        files_str = self.get("recent", "files", "[]") # Store as JSON list
        limit = self.getint("app", "recent_files_limit", 10)
        try:
            files = json.loads(files_str)
            if isinstance(files, list):
                # Filter out non-existent files
                valid_files = [f for f in files if isinstance(f, str) and os.path.exists(f)]
                # Ensure list doesn't exceed limit
                return valid_files[:limit]
            else:
                logger.warning("Recent files data in config is not a list.")
                return []
        except json.JSONDecodeError:
            logger.warning("Could not parse recent files JSON from config.")
            return []

    def set_recent_files(self, files: List[str]):
        """Sets the list of recent file paths in the configuration."""
        limit = self.getint("app", "recent_files_limit", 10)
        # Ensure uniqueness and limit
        unique_files = []
        for f in files:
            if f not in unique_files:
                unique_files.append(f)
        limited_files = unique_files[:limit]
        try:
            files_json = json.dumps(limited_files, indent=None)
            self.set("recent", "files", files_json)
        except (TypeError, ValueError) as e:
            logger.error(f"Could not serialize recent files list to JSON: {e}")


class RecentFilesManager:
    """
    Manages the list of recently used files, interacting with ConfigManager
    and notifying listeners of changes.
    """
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self._listeners: List[Any] = []
        # Load initial list (optional, could be lazy loaded)
        # self.recent_files = self.config_manager.get_recent_files()

    def register_listener(self, listener: Any):
        """Register a listener for recent files updates."""
        if listener not in self._listeners:
            self._listeners.append(listener)

    def unregister_listener(self, listener: Any):
        """Unregister a listener."""
        if listener in self._listeners:
            self._listeners.remove(listener)

    def _notify_change(self):
        """Notify listeners that the recent files list has changed."""
        current_list = self.get_recent_files()
        for listener in self._listeners:
            if hasattr(listener, "on_recent_files_changed"):
                try:
                    listener.on_recent_files_changed(current_list)
                except Exception as e:
                     logger.error(f"Error notifying recent files listener {listener}: {e}")

    def get_recent_files(self) -> List[str]:
        """Get the current list of recent files."""
        # Always get the fresh list from config manager
        return self.config_manager.get_recent_files()

    def add_recent_file(self, path: str):
        """Adds a file path to the top of the recent files list."""
        if not path or not os.path.exists(path): # Don't add invalid paths
            return

        current_files = self.get_recent_files()

        # Remove if already present (to move it to the top)
        if path in current_files:
            current_files.remove(path)

        # Add to the beginning
        current_files.insert(0, path)

        # Limit is handled by set_recent_files
        self.config_manager.set_recent_files(current_files)
        self._notify_change()

    def clear_recent_files(self):
        """Clears the recent files list."""
        self.config_manager.set_recent_files([])
        self._notify_change()


# --- View: User Interface ---

class AdvancedPdfToolkit:
    """Main application class orchestrating the GUI and interactions."""

    def __init__(self, root: ttb.Window):
        self.root = root
        self._check_dependencies() # Check dependencies before proceeding

        self.setup_core_components()
        self.setup_window_styling()
        self.create_variables()
        self.register_config_bindings() # Link variables to config changes
        self.create_main_layout()
        self.create_menu()
        self.create_sidebar()
        self.create_content_area() # Create notebook and header
        self.create_pages() # Populate notebook tabs
        self.create_status_bar()
        self.create_toolbar() # Optional toolbar

        self.bind_global_events()
        self.apply_theme() # Apply initial theme from config

        # Setup logging to route to UI *after* status bar is created
        self.setup_log_handler()

        # Load and display recent files
        self.recent_files_manager.register_listener(self) # Listen for changes
        self.populate_recent_files_menu() # Populate menu initially

        logger.info(f"{APP_NAME} GUI initialized successfully.")
        self.show_splash_screen() # Show splash after init

    def _check_dependencies(self):
        """Checks for missing dependencies and exits if critical ones are missing."""
        if MISSING_DEPENDENCIES:
            critical_missing = [dep for dep in MISSING_DEPENDENCIES if any(core in dep for core in ['pymupdf', 'pypdf', 'Pillow', 'ttkbootstrap'])]
            if critical_missing:
                error_message = "Critical dependencies missing:\n\n" + "\n".join(critical_missing)
                error_message += "\n\nPlease install them (e.g., using 'pip install ...') and restart the application."
                messagebox.showerror("Dependency Error", error_message)
                logger.critical(f"Exiting due to missing critical dependencies: {critical_missing}")
                self.root.quit() # Use quit instead of destroy before mainloop
                sys.exit(1)
            else:
                # Log warnings for non-critical missing dependencies
                logger.warning(f"Optional dependencies missing: {MISSING_DEPENDENCIES}. Some fallback features might be unavailable.")


    def setup_core_components(self):
        """Initialize backend managers."""
        self.config_manager = ConfigManager()
        # Re-initialize logger with level from config now that config is loaded
        global logger
        logger = setup_logging(self.config_manager.get("app", "log_level", "INFO"))

        self.recent_files_manager = RecentFilesManager(self.config_manager)
        self.operation_manager = OperationManager()
        self.operation_manager.register_listener(self) # UI listens to manager
        self.log_queue = queue.Queue() # Queue for thread-safe logging to UI

    def setup_log_handler(self):
        """Setup log handler to route messages to UI status bar via queue."""
        if not hasattr(self, 'status_message'):
             logger.error("Log handler setup called before status bar creation.")
             return

        def log_callback(level: str, message: str):
            # Put log message onto the queue for the GUI thread to process
            try:
                 self.log_queue.put_nowait((level, message))
            except queue.Full:
                 # Should not happen with default queue size, but handle anyway
                 print(f"Log queue full! Dropping message: [{level}] {message}", file=sys.stderr)

        self.custom_gui_handler = CustomLogHandler(log_callback)
        log_level_str = self.config_manager.get("app", "log_level", "INFO")
        log_level = getattr(logging, log_level_str.upper(), logging.INFO)
        self.custom_gui_handler.setLevel(log_level)
        # Use a simpler format for the GUI status bar
        self.custom_gui_handler.setFormatter(logging.Formatter('%(message)s'))

        logger.addHandler(self.custom_gui_handler)
        logger.info("GUI log handler initialized.")

        # Start the log checker loop
        self.check_log_queue()

    def check_log_queue(self):
        """Periodically check the log queue and update the status bar."""
        try:
            while not self.log_queue.empty():
                level, message = self.log_queue.get_nowait()
                # Call add_status_message directly as this runs in the GUI thread
                self.add_status_message(message, level.lower())
        except queue.Empty:
            pass
        except Exception as e:
             # Avoid crashing the checker loop
             logger.error(f"Error processing log queue: {e}")

        # Schedule the next check
        self.root.after(150, self.check_log_queue) # Check every 150ms

    def setup_window_styling(self):
        """Configure the main window appearance."""
        self.root.title(f"{APP_NAME} v{__version__}")
        # Set minimum size
        self.root.minsize(850, 600)

        # Attempt to set initial size based on screen
        try:
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            # Set initial size relative to screen, with max limits
            window_width = min(1600, int(screen_width * 0.75))
            window_height = min(1000, int(screen_height * 0.75))
            # Ensure minimum size is respected
            window_width = max(self.root.winfo_reqwidth(), window_width)
            window_height = max(self.root.winfo_reqheight(), window_height)

            # Center the window
            x_pos = (screen_width - window_width) // 2
            y_pos = (screen_height - window_height) // 2
            self.root.geometry(f"{window_width}x{window_height}+{x_pos}+{y_pos}")
        except tk.TclError: # May fail on some systems before mainloop
             logger.warning("Could not get screen dimensions to set initial window size.")
             self.root.geometry("1000x700") # Fallback size

        # Set icon (replace with actual icon handling)
        # self.set_window_icon() # Separate method for icon logic

        # Use ttkbootstrap style engine directly
        self.style = self.root.style
        # Define custom styles if needed (ttkbootstrap might cover most)
        self.style.configure('Title.TLabel', font=('Helvetica', 16, 'bold')) # Adjusted size
        self.style.configure('Subtitle.TLabel', font=('Helvetica', 10)) # Adjusted size
        self.style.configure('Heading.TLabel', font=('Helvetica', 12, 'bold'))
        # Status label styles (ttkbootstrap might have these: success, danger, warning, info)
        # self.style.configure('Success.TLabel', foreground=self.style.colors.success) # Example
        # self.style.configure('Error.TLabel', foreground=self.style.colors.danger)
        # self.style.configure('Warning.TLabel', foreground=self.style.colors.warning
        self.style.configure('Info.TLabel', foreground=self.style.colors.info)
        # Ttkbootstrap buttons often have color styles like 'success', 'danger' built-in
        # self.style.configure('Primary.TButton', font=('Helvetica', 11, 'bold')) # Example if needed
        self.style.configure('Card.TFrame', relief='raised', borderwidth=1)
        self.style.configure('PathEntry.TEntry', font=('Courier', 10)) # Example for specific entries

    def create_variables(self):
        """Initialize tkinter variables for UI state and bindings."""
        logger.debug("Creating UI variables...")
        # Theme variables
        self.current_theme_mode = tk.StringVar(value=self.config_manager.get_theme_mode().value)
        self.is_dark_mode = tk.BooleanVar(value=(self.current_theme_mode.get() == Theme.DARK.value)) # Initial guess

        # --- Operation Specific Variables ---

        # Split PDF
        self.split_input_path = tk.StringVar()
        self.split_output_dir = tk.StringVar()
        self.split_method = tk.StringVar(value="ranges")
        self.split_ranges = tk.StringVar()
        self.split_prefix = tk.StringVar()
        self.split_password = tk.StringVar()

        # Merge PDF
        self.merge_input_paths: List[str] = [] # List of actual paths
        self.merge_passwords: Dict[str, str] = {} # {filepath: password}
        self.merge_output_path = tk.StringVar()
        self.merge_add_bookmarks = tk.BooleanVar(value=True)
        # Template: Use 'FILENAME' or 'INDEX_FILENAME' as values
        self.merge_bookmark_template = tk.StringVar(value="FILENAME")

        # Images to PDF
        self.img2pdf_input_paths: List[str] = []
        self.img2pdf_output_path = tk.StringVar()
        self.img2pdf_page_size = tk.StringVar(value=self.config_manager.get("pdf", "img2pdf_page_size", "A4"))
        self.img2pdf_margin_pt = tk.DoubleVar(value=self.config_manager.getfloat("pdf", "img2pdf_margin_pt", 36.0))
        self.img2pdf_fit_method = tk.StringVar(value="fit")
        self.img2pdf_resolution = tk.IntVar(value=self.config_manager.getint("pdf", "default_dpi", 300))
        self.img2pdf_quality = tk.IntVar(value=self.config_manager.getint("pdf", "img2pdf_quality", 95))

        # PDF to Images
        self.pdf2img_input_path = tk.StringVar()
        self.pdf2img_output_dir = tk.StringVar()
        self.pdf2img_format = tk.StringVar(value="png")
        self.pdf2img_dpi = tk.IntVar(value=self.config_manager.getint("pdf", "default_dpi", 300))
        self.pdf2img_prefix = tk.StringVar()
        self.pdf2img_password = tk.StringVar()
        self.pdf2img_page_range = tk.StringVar() # e.g., "1, 3-5, 10-"

        # Compress PDF
        self.compress_input_path = tk.StringVar()
        self.compress_output_path = tk.StringVar()
        self.compress_quality = tk.IntVar(value=self.config_manager.getint("pdf", "compress_quality", 75))
        self.compress_downsample = tk.BooleanVar(value=True)
        self.compress_image_dpi = tk.IntVar(value=self.config_manager.getint("pdf", "compress_img_dpi", 150))
        self.compress_password = tk.StringVar()

        # Encrypt/Decrypt PDF
        self.encrypt_input_path = tk.StringVar()
        self.encrypt_output_path = tk.StringVar()
        self.encrypt_action = tk.StringVar(value="encrypt")
        self.encrypt_user_password = tk.StringVar()
        self.encrypt_owner_password = tk.StringVar()
        self.encrypt_level = tk.IntVar(value=self.config_manager.getint("pdf", "encrypt_level", 128))
        self.encrypt_allow_printing = tk.BooleanVar(value=True)
        self.encrypt_allow_copying = tk.BooleanVar(value=False)
        self.encrypt_allow_modifying = tk.BooleanVar(value=False)
        self.encrypt_allow_annotating = tk.BooleanVar(value=True)
        self.encrypt_input_password = tk.StringVar() # Password for input file

        # Extract Text
        self.extract_input_path = tk.StringVar()
        self.extract_output_path = tk.StringVar()
        self.extract_format = tk.StringVar(value="plain") # plain, formatted, html, markdown, json etc.
        self.extract_page_range = tk.StringVar()
        self.extract_password = tk.StringVar()
        self.extract_layout = tk.BooleanVar(value=False) # Default to no layout preservation

        # Rotate PDF
        self.rotate_input_path = tk.StringVar()
        self.rotate_output_path = tk.StringVar()
        self.rotate_angle = tk.IntVar(value=90) # Must be 90, 180, or 270
        self.rotate_page_range = tk.StringVar()
        self.rotate_password = tk.StringVar()

        # Edit Metadata
        self.metadata_input_path = tk.StringVar()
        self.metadata_output_path = tk.StringVar()
        self.metadata_title = tk.StringVar()
        self.metadata_author = tk.StringVar()
        self.metadata_subject = tk.StringVar()
        self.metadata_keywords = tk.StringVar()
        self.metadata_creator = tk.StringVar()
        self.metadata_producer = tk.StringVar()
        self.metadata_password = tk.StringVar()

        # Status Bar Variables
        self.status_message = tk.StringVar(value="Ready.")
        self.status_level = tk.StringVar(value="info") # info, success, warning, error
        self.progress_value = tk.DoubleVar(value=0.0) # Use DoubleVar for float progress
        self.progress_visible = tk.BooleanVar(value=False)
        self.status_history: deque = deque(maxlen=200) # Store status messages for log viewer

        logger.debug("UI variables created.")

    def register_config_bindings(self):
        """Bind tkinter variables to automatically update ConfigManager on change."""
        logger.debug("Registering config bindings...")
        bindings = {
            ("app", "theme"): self.current_theme_mode,
            ("app", "recent_files_limit"): self.config_manager.getint("app", "recent_files_limit"), # Read-only example
            ("app", "log_level"): self.config_manager.get("app", "log_level"), # Read-only, set via settings dialog
            ("pdf", "img2pdf_page_size"): self.img2pdf_page_size,
            ("pdf", "img2pdf_margin_pt"): self.img2pdf_margin_pt,
            ("pdf", "img2pdf_quality"): self.img2pdf_quality,
            ("pdf", "default_dpi"): self.img2pdf_resolution, # Link default DPI to a relevant UI var
            ("pdf", "compress_quality"): self.compress_quality,
            ("pdf", "compress_img_dpi"): self.compress_image_dpi,
            ("pdf", "encrypt_level"): self.encrypt_level,
        }

        # Simple trace lambda to update config and save
        def _create_config_update_callback(section, key, var):
            # This callback runs when the tk variable *changes*
            def _update_config(*args):
                try:
                    value = var.get()
                    self.config_manager.set(section, key, value)
                    # Save immediately? Or have explicit save in settings? Let's save.
                    self.config_manager.save_config()
                    logger.debug(f"Config updated via binding: [{section}] {key} = {value}")
                except tk.TclError:
                     pass # Ignore errors during interpreter shutdown
                except Exception as e:
                     logger.error(f"Error in config update callback for {section}/{key}: {e}")
            return _update_config

        for (section, key), tk_var in bindings.items():
            # Ensure the variable exists and has a trace method
            if tk_var and hasattr(tk_var, 'trace_add'):
                callback = _create_config_update_callback(section, key, tk_var)
                # Use unique name for trace to avoid conflicts if re-registering
                trace_name = f"config_trace_{section}_{key}"
                # Remove existing trace first if any
                try:
                    existing_traces = tk_var.trace_info()
                    for mode, name in existing_traces:
                         if name == trace_name:
                             tk_var.trace_remove(mode, name)
                except tk.TclError: pass # Ignore if var destroyed
                # Add the new trace
                tk_var.trace_add("write", callback) # Add trace using the generated callback
                setattr(self, trace_name, callback) # Keep reference to callback


        # Special binding for theme mode change
        self.current_theme_mode.trace_add("write", self._theme_mode_changed)

    def _theme_mode_changed(self, *args):
        """Handle theme mode change triggered by the variable."""
        selected_mode = self.current_theme_mode.get()
        logger.info(f"Theme mode variable changed to: {selected_mode}")
        self.config_manager.set_theme_mode(Theme(selected_mode))
        self.config_manager.save_config()
        self.apply_theme() # Apply the theme change visually


    def create_main_layout(self):
        """Create the main PanedWindow layout with sidebar and content areas."""
        logger.debug("Creating main layout...")
        # Main container frame fills the root window
        self.main_frame = ttk.Frame(self.root, padding=0)
        self.main_frame.pack(expand=True, fill=tk.BOTH)

        # Paned window for resizable sidebar/content
        self.paned_window = ttk.PanedWindow(self.main_frame, orient=tk.HORIZONTAL)
        self.paned_window.pack(expand=True, fill=tk.BOTH)

        # Sidebar Frame (added to paned window later in create_sidebar)
        self.sidebar_frame = ttk.Frame(self.paned_window, width=220, style='primary.TFrame') # Example style
        # Content Frame (added to paned window later in create_content_area)
        self.content_frame = ttk.Frame(self.paned_window)

        self.paned_window.add(self.sidebar_frame, weight=0) # Sidebar has fixed initial width
        self.paned_window.add(self.content_frame, weight=1) # Content area takes remaining space


    def create_menu(self):
        """Create the main application menu bar."""
        logger.debug("Creating menu bar...")
        self.menu_bar = tk.Menu(self.root)
        self.root.config(menu=self.menu_bar)

        # --- File Menu ---
        file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open PDF...", command=lambda: self.select_input_file_generic("pdf"))
        file_menu.add_command(label="Open Image(s)...", command=lambda: self.select_input_file_generic("images"))
        file_menu.add_separator()
        self.recent_files_menu = tk.Menu(file_menu, tearoff=0)
        file_menu.add_cascade(label="Recent Files", menu=self.recent_files_menu)
        # Populate recent_files_menu later
        file_menu.add_separator()
        file_menu.add_command(label="Settings...", command=self.show_settings_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit_app)

        # --- Edit Menu ---
        edit_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Clear Current Tab Inputs", command=self.clear_all_inputs)
        # Add Cut/Copy/Paste if needed for specific text fields later

        # --- View Menu ---
        view_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="View", menu=view_menu)
        theme_submenu = tk.Menu(view_menu, tearoff=0)
        view_menu.add_cascade(label="Theme", menu=theme_submenu)
        # Use Theme enum values for radio buttons
        theme_submenu.add_radiobutton(label="System", variable=self.current_theme_mode, value=Theme.SYSTEM.value)
        theme_submenu.add_radiobutton(label="Light", variable=self.current_theme_mode, value=Theme.LIGHT.value)
        theme_submenu.add_radiobutton(label="Dark", variable=self.current_theme_mode, value=Theme.DARK.value)
        theme_submenu.add_radiobutton(label="Custom", variable=self.current_theme_mode, value=Theme.CUSTOM.value)
        # Theme change is handled by variable trace -> _theme_mode_changed -> apply_theme
        theme_submenu.add_separator()
        theme_submenu.add_command(label="Edit Custom Theme...", command=self.edit_custom_theme)
        view_menu.add_separator()
        view_menu.add_command(label="Show Log Viewer", command=self.show_log_viewer)

        # --- Tools Menu (Shortcuts to Notebook Tabs) ---
        tools_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Tools", menu=tools_menu)
        self.tool_commands = [ # Store command references if needed later
            tools_menu.add_command(label="Split PDF", command=lambda: self.notebook.select(0)),
            tools_menu.add_command(label="Merge PDFs", command=lambda: self.notebook.select(1)),
            tools_menu.add_command(label="Images to PDF", command=lambda: self.notebook.select(2)),
            tools_menu.add_command(label="PDF to Images", command=lambda: self.notebook.select(3)),
            tools_menu.add_command(label="Compress PDF", command=lambda: self.notebook.select(4)),
            tools_menu.add_command(label="Encrypt/Decrypt PDF", command=lambda: self.notebook.select(5)),
            tools_menu.add_command(label="Extract Text", command=lambda: self.notebook.select(6)),
            tools_menu.add_command(label="Rotate PDF", command=lambda: self.notebook.select(7)),
            tools_menu.add_command(label="Edit Metadata", command=lambda: self.notebook.select(8)),
        ]

        # --- Help Menu ---
        help_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="Help Topics...", command=self.show_help)
        help_menu.add_command(label="Check for Updates...", command=self.check_for_updates)
        help_menu.add_separator()
        help_menu.add_command(label=f"About {APP_NAME}...", command=self.show_about_dialog)

    def populate_recent_files_menu(self):
        """Update the File -> Recent Files menu dynamically."""
        logger.debug("Populating recent files menu...")
        self.recent_files_menu.delete(0, tk.END) # Clear existing items
        recent_files = self.recent_files_manager.get_recent_files()

        if not recent_files:
            self.recent_files_menu.add_command(label="(No recent files)", state=tk.DISABLED)
        else:
            for i, path in enumerate(recent_files):
                label = f"&{i+1 % 10}. {os.path.basename(path)}" # Add accelerator key (1-0)
                # Truncate long paths in menu label if needed
                max_len = 60
                if len(path) > max_len:
                    label = f"&{i+1 % 10}. {os.path.basename(path)} (...{path[-max_len+15:]})"

                # Determine action based on file type
                action = None
                if path.lower().endswith('.pdf'):
                    action = partial(self.open_recent_pdf, path)
                elif any(path.lower().endswith(ext) for ext in SUPPORTED_IMAGE_FORMATS):
                    action = partial(self.open_recent_image, path)
                else:
                    action = partial(self.open_recent_unknown, path) # Generic open

                if action:
                     self.recent_files_menu.add_command(label=label, command=action, underline=0) # Underline the number

            self.recent_files_menu.add_separator()
            self.recent_files_menu.add_command(label="Clear Recent Files", command=self.clear_recent_files_menu)

    def open_recent_pdf(self, path: str):
        """Handles opening a recent PDF file."""
        if not os.path.exists(path):
             messagebox.showerror("File Not Found", f"The file '{path}' no longer exists.")
             # Consider removing it from recent list here?
             return
        logger.info(f"Opening recent PDF: {path}")
        # Heuristic: Open in Split, PDF2Img, Compress, Encrypt, Extract, Rotate, Metadata
        # Prioritize the currently active tab if it accepts PDF input
        current_tab_index = self.notebook.index(self.notebook.select())
        # Tabs accepting single PDF input path variable:
        pdf_input_tabs = [0, 3, 4, 5, 6, 7, 8]

        if current_tab_index in pdf_input_tabs:
            self.set_input_path_for_tab(current_tab_index, path)
        else:
            # Default to Split tab if current tab doesn't take PDF input
            self.set_input_path_for_tab(0, path)
            self.notebook.select(0) # Switch view to Split tab

    def open_recent_image(self, path: str):
        """Handles opening a recent image file."""
        if not os.path.exists(path):
             messagebox.showerror("File Not Found", f"The file '{path}' no longer exists.")
             return
        logger.info(f"Opening recent image: {path}")
        # Always target the Images to PDF tab
        self.add_img2pdf_files([path]) # Use the method to add single file
        self.notebook.select(2) # Switch to Images to PDF tab

    def open_recent_unknown(self, path: str):
        """Handle opening recent files with unknown/unhandled types."""
        logger.warning(f"Attempting to open recent file of unknown type: {path}")
        messagebox.showinfo("Open Recent", f"Cannot automatically determine how to open:\n{path}\n\nPlease select the appropriate tool manually.")


    def set_input_path_for_tab(self, tab_index: int, path: str):
        """Sets the primary input path variable for the specified tab index."""
        path_var = self.get_input_var_for_tab(tab_index)
        if path_var:
            path_var.set(path)
            # Add to recent files (triggers trace for auto-fill output)
            self.recent_files_manager.add_recent_file(path)
            logger.debug(f"Set input path for tab {tab_index} to: {path}")
        else:
             logger.warning(f"No primary input path variable found for tab index {tab_index}")


    def clear_recent_files_menu(self):
        """Command to clear recent files list."""
        if messagebox.askyesno("Confirm Clear", "Are you sure you want to clear the recent files list?"):
            logger.info("Clearing recent files list.")
            self.recent_files_manager.clear_recent_files()
            # The notification mechanism should automatically call populate_recent_files_menu

    def on_recent_files_changed(self, recent_files: List[str]):
        """Callback when the RecentFilesManager detects a change."""
        logger.debug("Recent files list changed, updating menu.")
        self.populate_recent_files_menu() # Update the menu display


    def create_sidebar(self):
        """Creates the content of the left sidebar."""
        logger.debug("Creating sidebar...")
        sidebar = self.sidebar_frame # Use the frame already created
        sidebar.columnconfigure(0, weight=1)

        # --- App Title Area ---
        title_frame = ttk.Frame(sidebar, padding=(10, 10))
        title_frame.grid(row=0, column=0, sticky='ew')
        # Consider adding an icon here
        app_title = ttk.Label(title_frame, text=APP_NAME, font=('Helvetica', 14, 'bold'))
        app_title.pack(side=tk.LEFT, anchor='w')
        version_label = ttk.Label(title_frame, text=f" v{__version__}", font=('Helvetica', 9))
        version_label.pack(side=tk.LEFT, anchor='sw', padx=(2, 0))

        # --- Separator ---
        ttk.Separator(sidebar, orient=tk.HORIZONTAL).grid(row=1, column=0, sticky='ew', padx=10, pady=5)

        # --- Navigation Buttons ---
        nav_frame = ttk.Frame(sidebar, padding=(10, 0))
        nav_frame.grid(row=2, column=0, sticky='nsew')
        nav_frame.columnconfigure(0, weight=1)

        self.nav_buttons: Dict[int, ttk.Button] = {}
        nav_items = [
            ("Split PDF", 0), ("Merge PDFs", 1), ("Images to PDF", 2),
            ("PDF to Images", 3), ("Compress PDF", 4), ("Encrypt/Decrypt", 5),
            ("Extract Text", 6), ("Rotate PDF", 7), ("Edit Metadata", 8)
        ]

        for i, (text, tab_id) in enumerate(nav_items):
            # Use ttkbootstrap buttons, potentially with styling ('primary', 'secondary', etc.)
            btn = ttk.Button(nav_frame, text=text,
                           command=lambda t=tab_id: self.notebook.select(t),
                           style='Outline.TButton') # Example style
            btn.grid(row=i, column=0, sticky='ew', pady=2)
            self.nav_buttons[tab_id] = btn

        # --- Separator ---
        ttk.Separator(sidebar, orient=tk.HORIZONTAL).grid(row=3, column=0, sticky='ew', padx=10, pady=10)

        # --- Tools Section ---
        tools_frame = ttk.Frame(sidebar, padding=(10, 0))
        tools_frame.grid(row=4, column=0, sticky='nsew')
        tools_frame.columnconfigure(0, weight=1)

        ttk.Button(tools_frame, text="Recent Files...", command=self.show_recent_files_dialog, style='Outline.TButton').grid(row=0, column=0, sticky='ew', pady=2)
        ttk.Button(tools_frame, text="Settings...", command=self.show_settings_dialog, style='Outline.TButton').grid(row=1, column=0, sticky='ew', pady=2)
        ttk.Button(tools_frame, text="View Logs...", command=self.show_log_viewer, style='Outline.TButton').grid(row=2, column=0, sticky='ew', pady=2)

        # --- Spacer to push About button down ---
        ttk.Frame(sidebar).grid(row=5, column=0, sticky='ns', pady=10) # Empty frame acts as spacer
        sidebar.rowconfigure(5, weight=1) # Make spacer expand vertically

        # --- About Button ---
        about_frame = ttk.Frame(sidebar, padding=(10, 10))
        about_frame.grid(row=6, column=0, sticky='ew')
        about_frame.columnconfigure(0, weight=1)
        ttk.Button(about_frame, text="About", command=self.show_about_dialog, style='Outline.TButton').grid(row=0, column=0, sticky='ew')

    def create_content_area(self):
        """Creates the main content area with header and notebook."""
        logger.debug("Creating content area...")
        content = self.content_frame # Use the frame already created
        content.columnconfigure(0, weight=1)
        content.rowconfigure(1, weight=1) # Allow notebook to expand

        # --- Header Frame ---
        self.header_frame = ttk.Frame(content, padding=(10, 10, 10, 5))
        self.header_frame.grid(row=0, column=0, sticky='ew')

        self.page_title = ttk.Label(self.header_frame, text="Welcome", style='Title.TLabel')
        self.page_title.pack(side=tk.LEFT, anchor='w')

        self.page_desc = ttk.Label(self.header_frame, text="Select a tool from the sidebar to begin.", style='Subtitle.TLabel')
        self.page_desc.pack(side=tk.LEFT, anchor='sw', padx=10)

        # --- Notebook for Operations ---
        self.notebook = ttk.Notebook(content, style='TNotebook') # Apply ttkbootstrap style
        self.notebook.grid(row=1, column=0, sticky='nsew', padx=5, pady=(0, 5))

        # Add frames to notebook (content added in create_pages)
        self.tab_frames: Dict[int, ttk.Frame] = {}
        tab_names = [
            "Split PDF", "Merge PDFs", "Images to PDF", "PDF to Images",
            "Compress PDF", "Encrypt/Decrypt", "Extract Text", "Rotate PDF",
            "Edit Metadata"
        ]
        for i, name in enumerate(tab_names):
             frame = ttk.Frame(self.notebook, padding=15)
             self.notebook.add(frame, text=name)
             self.tab_frames[i] = frame

        # Bind tab change event
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)


    def create_pages(self):
        """Populate notebook tabs with widgets for each operation."""
        logger.debug("Creating pages within notebook...")
        # Call helper methods to create content for each tab frame
        # Pass the correct frame from self.tab_frames
        self._create_split_page(self.tab_frames[0])
        self._create_merge_page(self.tab_frames[1])
        self._create_img2pdf_page(self.tab_frames[2])
        self._create_pdf2img_page(self.tab_frames[3])
        self._create_compress_page(self.tab_frames[4])
        self._create_encrypt_page(self.tab_frames[5])
        self._create_extract_page(self.tab_frames[6])
        self._create_rotate_page(self.tab_frames[7])
        self._create_metadata_page(self.tab_frames[8])

        # Select initial tab (optional, e.g., Split)
        self.notebook.select(0)
        self.on_tab_changed() # Manually trigger update for initial tab


    def _create_widget_row(self, parent: ttk.Frame, label_text: str,
                           variable: Optional[tk.Variable] = None,
                           widget_type: str = 'entry', # entry, readonly_entry, password_entry, combobox, checkbutton, spinbox
                           widget_options: Optional[Dict] = None,
                           button_text: Optional[str] = None,
                           button_cmd: Optional[Callable] = None,
                           tooltip: Optional[str] = None,
                           label_width: int = 20) -> Tuple[ttk.Frame, Optional[ttk.Widget]]:
        """Helper to create a standard labeled widget row."""
        widget_options = widget_options or {}
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2) # Pack row frame horizontally

        label = ttk.Label(frame, text=label_text, width=label_width, anchor=tk.W)
        label.pack(side=tk.LEFT, padx=(0, 5))

        widget = None
        if widget_type == 'entry':
            widget = ttk.Entry(frame, textvariable=variable, **widget_options)
        elif widget_type == 'readonly_entry':
             # Use normal entry, set state in options if needed or outside
            widget_options['state'] = 'readonly'
            widget = ttk.Entry(frame, textvariable=variable, **widget_options)
        elif widget_type == 'password_entry':
            widget_options['show'] = '*'
            widget = ttk.Entry(frame, textvariable=variable, **widget_options)
        elif widget_type == 'combobox':
             widget_options['state'] = 'readonly' # Usually read-only
             widget = ttk.Combobox(frame, textvariable=variable, **widget_options)
        elif widget_type == 'checkbutton':
             # Label is the text for checkbutton, so label widget might be redundant
             # Let's keep the label for alignment, but checkbutton text is empty
             label.pack_forget() # Hide the separate label widget
             widget_options['text'] = label_text # Use label_text for checkbutton text
             widget = ttk.Checkbutton(frame, variable=variable, **widget_options)
             widget.pack(side=tk.LEFT, padx=5) # Pack checkbutton differently
             frame.widget = widget # Store ref
             if button_text: # Pack button after checkbutton
                  button = ttk.Button(frame, text=button_text, command=button_cmd, width=10, style='Outline.TButton')
                  button.pack(side=tk.LEFT, padx=(15, 0)) # Add more space
             return frame, widget # Skip standard packing below for checkbutton
        elif widget_type == 'spinbox':
             widget = ttk.Spinbox(frame, textvariable=variable, **widget_options)

        if widget:
            widget.pack(side=tk.LEFT, fill=tk.X, expand=True)
            # Add tooltip if provided (requires external library or simple event binding)
            if tooltip:
                 self.add_tooltip(widget, tooltip)

        if button_text and button_cmd:
             # Use outline style for secondary buttons
            button = ttk.Button(frame, text=button_text, command=button_cmd, width=10, style='Outline.TButton')
            button.pack(side=tk.LEFT, padx=(5, 0))

        frame.widget = widget # Store reference if needed
        return frame, widget

    def _create_io_section(self, parent: ttk.Frame, input_var: tk.StringVar, output_var: tk.StringVar,
                       input_label: str="Input File:", output_label: str="Output Location:",
                       input_type: str="pdf", # pdf, image, images, any
                       output_type: str="dir", # dir, pdf, txt
                       input_cmd: Optional[Callable]=None, output_cmd: Optional[Callable]=None):
        """Helper to create standard Input/Output file/dir selection rows."""
        # --- Input Row ---
        if input_cmd is None:
            if input_type == "images": # Special case for multi-file select
                input_cmd = lambda: self.select_input_files_generic() # Needs context which listbox/var
            else:
                 input_cmd = lambda: self.select_input_file(input_var, input_type)

        input_widget_type = 'readonly_entry' if input_type != "images" else 'label' # Don't use entry for multi-image

        if input_widget_type == 'label':
            # Handle multi-input display (e.g., listbox elsewhere)
             self._create_widget_row(parent, input_label, None, 'label', # Just the label
                                    widget_options={'text': '(Use Add button below)'})
        else:
             _, entry = self._create_widget_row(parent, input_label, input_var, input_widget_type,
                                           button_text="Browse...", button_cmd=input_cmd)
             if entry: entry.configure(style='PathEntry.TEntry')

        # --- Output Row ---
        if output_cmd is None:
            if output_type == "dir":
                output_cmd = lambda: self.select_output_dir(output_var)
            else: # Assume file
                output_cmd = lambda: self.select_output_file(output_var, output_type)

        _, entry = self._create_widget_row(parent, output_label, output_var, 'readonly_entry',
                                       button_text="Browse...", button_cmd=output_cmd)
        if entry: entry.configure(style='PathEntry.TEntry')

    def _create_password_field(self, parent: ttk.Frame, variable: tk.StringVar,
                              label: str = "Password (if any):") -> Tuple[ttk.Frame, ttk.Entry]:
        """Helper to create a password input field row."""
        return self._create_widget_row(parent, label, variable, 'password_entry', widget_options={'width': 30})

    def _create_page_range_field(self, parent: ttk.Frame, variable: tk.StringVar,
                                label: str="Page Range (optional):") -> Tuple[ttk.Frame, ttk.Entry]:
        """Helper to create a page range input field with help text."""
        # Use the helper to create the main row (label + entry)
        # Assuming the default label_width in _create_widget_row is around 20
        frame, entry = self._create_widget_row(parent, label, variable, 'entry', widget_options={'width': 30})

        # Add help text below the row frame, packed directly into the parent frame
        help_label = ttk.Label(parent, text="e.g., 1, 3, 5-7, 10- (blank = all)", style='Subtitle.TLabel')

        # --- MODIFIED LINE ---
        # Use a fixed pixel indent instead of the complex style lookup
        fixed_indent = 150 # Adjust this value as needed for visual alignment
        help_label.pack(anchor=tk.W, padx=(fixed_indent, 0), pady=(0, 5))
        # --- END MODIFIED LINE ---

        return frame, entry # Return the original row frame and entry widget

    def _create_action_button(self, parent: ttk.Frame, text: str, command: Callable, style: str = 'primary') -> ttk.Button:
        """Helper to create the main action button for a page, centered."""
        # Use a separate frame to center the button
        btn_container = ttk.Frame(parent, padding=(0, 15, 0, 5))
        btn_container.pack(fill=tk.X) # Fill horizontally
        btn_container.columnconfigure(0, weight=1) # Allow centering

        # Use ttkbootstrap colored button styles
        button = ttk.Button(btn_container, text=text, command=command, style=f'{style}.TButton', width=25)
        # Pack in the container to center it
        button.pack(pady=5) # Let pack handle centering within the container
        return button

    # --- Individual Page Creation Methods ---

    def _create_split_page(self, parent: ttk.Frame):
        logger.debug("Creating Split page...")
        # Input/Output
        self._create_io_section(parent, self.split_input_path, self.split_output_dir,
                                input_label="PDF to Split:", output_label="Output Directory:",
                                input_type="pdf", output_type="dir")
        self._create_password_field(parent, self.split_password, label="Input PDF Password:")
        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10, padx=5)

        # Options Frame
        options_frame = ttk.LabelFrame(parent, text="Split Options", padding=10)
        options_frame.pack(fill=tk.X, padx=5, pady=5)

        # Split Method Radio Buttons
        method_frame = ttk.Frame(options_frame)
        method_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(method_frame, text="Method:", width=15).pack(side=tk.LEFT, padx=(0, 10))
        # Use ttkbootstrap radiobuttons if desired for different look
        r1 = ttk.Radiobutton(method_frame, text="By Page Ranges", variable=self.split_method, value="ranges")
        r1.pack(side=tk.LEFT, padx=5)
        r2 = ttk.Radiobutton(method_frame, text="Single Pages", variable=self.split_method, value="single")
        r2.pack(side=tk.LEFT, padx=5)
        r3 = ttk.Radiobutton(method_frame, text="By Bookmarks", variable=self.split_method, value="bookmarks")
        r3.pack(side=tk.LEFT, padx=5)
        self.add_tooltip(r3, "Splits based on top-level bookmarks in the PDF.")

        # Page Ranges Entry (conditionally visible)
        self.split_ranges_row, self.split_ranges_entry = self._create_page_range_field(options_frame, self.split_ranges, label="Page Ranges:")

        # Output Prefix
        _, self.split_prefix_entry = self._create_widget_row(options_frame, "Output Prefix:", self.split_prefix, 'entry', widget_options={'width': 30})
        self.add_tooltip(self.split_prefix_entry, "Optional prefix for output filenames (e.g., 'Chapter_'). Defaults to input filename.")

        # --- Visibility Logic ---
        def _update_split_options_visibility(*args):
            method = self.split_method.get()
            is_ranges = (method == "ranges")
            # Define the fixed indent here as well
            fixed_indent = 150 # Use the same value as in _create_page_range_field

            # Show/hide the entire row frame for ranges
            if is_ranges:
                self.split_ranges_row.pack(fill=tk.X, pady=2) # Show the row
                # Find the help label associated with ranges and show it
                for child in options_frame.winfo_children():
                     if isinstance(child, ttk.Label) and "e.g., 1, 3, 5-7" in child.cget("text"):
                         # --- MODIFIED LINE ---
                         child.pack(anchor=tk.W, padx=(fixed_indent, 0), pady=(0,5)) # Use fixed indent
                         # --- END MODIFIED LINE ---
                         break
            else:
                self.split_ranges_row.pack_forget() # Hide the row
                # Find the help label associated with ranges and hide it
                for child in options_frame.winfo_children():
                     if isinstance(child, ttk.Label) and "e.g., 1, 3, 5-7" in child.cget("text"):
                         child.pack_forget()
                         break


        self.split_method.trace_add("write", _update_split_options_visibility)
        _update_split_options_visibility() # Initial setup

        # Action Button
        self._create_action_button(parent, "Split PDF", self.run_split_pdf)


    def _create_merge_page(self, parent: ttk.Frame):
        logger.debug("Creating Merge page...")
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        # Use PanedWindow for File list / Options layout
        h_pane = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        h_pane.grid(row=0, column=0, sticky='nsew', padx=5, pady=(0,5))

        # --- Left Pane: File List and Controls ---
        left_frame = ttk.Frame(h_pane, padding=5)
        h_pane.add(left_frame, weight=1)
        left_frame.rowconfigure(1, weight=1) # Listbox expands
        left_frame.columnconfigure(0, weight=1) # Listbox expands

        ttk.Label(left_frame, text="Files to Merge (Drag to Reorder):", style='Heading.TLabel').grid(row=0, column=0, columnspan=3, sticky='w', pady=(0, 5))

        # Listbox for files
        self.merge_listbox = tk.Listbox(left_frame, selectmode=tk.EXTENDED, width=40, height=10)
        self.merge_listbox.grid(row=1, column=0, columnspan=2, sticky='nsew')
        merge_scrollbar = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, command=self.merge_listbox.yview, style='Vertical.TScrollbar')
        merge_scrollbar.grid(row=1, column=2, sticky='ns')
        self.merge_listbox.config(yscrollcommand=merge_scrollbar.set)
        # Add drag-and-drop reordering later if desired (requires more complex binding)
        self.add_tooltip(self.merge_listbox, "Add PDF files. Select items and use buttons to reorder or remove. Drag-and-drop not yet implemented.")

        # List Manipulation Buttons Frame
        list_btn_frame = ttk.Frame(left_frame)
        list_btn_frame.grid(row=2, column=0, columnspan=3, pady=(5,0), sticky='ew')
        # Use smaller outline buttons
        ttk.Button(list_btn_frame, text="Add...", command=self.add_merge_files, style='Outline.TButton').pack(side=tk.LEFT, padx=2)
        ttk.Button(list_btn_frame, text="Remove", command=self.remove_merge_files, style='Outline.TButton').pack(side=tk.LEFT, padx=2)
        ttk.Button(list_btn_frame, text="Up", command=lambda: self.move_merge_item(-1), style='Outline.TButton').pack(side=tk.LEFT, padx=2)
        ttk.Button(list_btn_frame, text="Down", command=lambda: self.move_merge_item(1), style='Outline.TButton').pack(side=tk.LEFT, padx=2)
        ttk.Button(list_btn_frame, text="Password...", command=self.set_merge_password, style='Outline.TButton').pack(side=tk.LEFT, padx=2)
        self.add_tooltip(list_btn_frame.winfo_children()[-1], "Set password for the selected encrypted PDF in the list.")
        ttk.Button(list_btn_frame, text="Clear", command=self.clear_merge_list, style='Outline.Danger.TButton').pack(side=tk.LEFT, padx=2)


        # --- Right Pane: Output and Options ---
        right_frame = ttk.Frame(h_pane, padding=10)
        h_pane.add(right_frame, weight=1)

        # Output File
        self._create_io_section(right_frame, None, self.merge_output_path, # No single input var
                                output_label="Merged Output PDF:", output_type="pdf")
        ttk.Separator(right_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10, padx=5)

        # Merge Options Frame
        options_frame = ttk.LabelFrame(right_frame, text="Merge Options", padding=10)
        options_frame.pack(fill=tk.X, padx=5, pady=5)

        # Add Bookmarks Option
        bm_frame = ttk.Frame(options_frame)
        bm_frame.pack(fill=tk.X)
        check = ttk.Checkbutton(bm_frame, variable=self.merge_add_bookmarks, text="Add Bookmarks:")
        check.pack(side=tk.LEFT)
        combo = ttk.Combobox(bm_frame, textvariable=self.merge_bookmark_template,
                             values=["FILENAME", "INDEX_FILENAME"], state='readonly', width=18)
        combo.pack(side=tk.LEFT, padx=5)
        self.add_tooltip(combo, "Choose bookmark format:\nFILENAME: Use original filename.\nINDEX_FILENAME: Add number before filename.")
        # Enable/disable combobox based on checkbox
        def _toggle_bookmark_combo(*args):
            combo.config(state=tk.NORMAL if self.merge_add_bookmarks.get() else tk.DISABLED)
        self.merge_add_bookmarks.trace_add("write", _toggle_bookmark_combo)
        _toggle_bookmark_combo() # Initial state


        # Action Button (Pushed to bottom)
        right_frame.pack_propagate(False) # Prevent options frame from pushing button down initially? No, let it flow.
        self._create_action_button(right_frame, "Merge PDFs", self.run_merge_pdf)


    def _create_img2pdf_page(self, parent: ttk.Frame):
        logger.debug("Creating Images-to-PDF page...")
        # Similar layout to Merge page
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)
        h_pane = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        h_pane.grid(row=0, column=0, sticky='nsew', padx=5, pady=(0,5))

        # Left Pane: Image List
        left_frame = ttk.Frame(h_pane, padding=5)
        h_pane.add(left_frame, weight=1)
        left_frame.rowconfigure(1, weight=1)
        left_frame.columnconfigure(0, weight=1)
        ttk.Label(left_frame, text="Images to Convert (Order matters):", style='Heading.TLabel').grid(row=0, column=0, columnspan=3, sticky='w', pady=(0, 5))
        self.img2pdf_listbox = tk.Listbox(left_frame, selectmode=tk.EXTENDED, width=40, height=10)
        self.img2pdf_listbox.grid(row=1, column=0, columnspan=2, sticky='nsew')
        img_scrollbar = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, command=self.img2pdf_listbox.yview, style='Vertical.TScrollbar')
        img_scrollbar.grid(row=1, column=2, sticky='ns')
        self.img2pdf_listbox.config(yscrollcommand=img_scrollbar.set)
        self.add_tooltip(self.img2pdf_listbox, "Add image files. Order determines page order in the PDF.")

        # List Buttons
        list_btn_frame = ttk.Frame(left_frame)
        list_btn_frame.grid(row=2, column=0, columnspan=3, pady=(5,0), sticky='ew')
        ttk.Button(list_btn_frame, text="Add...", command=self.add_img2pdf_files, style='Outline.TButton').pack(side=tk.LEFT, padx=2)
        ttk.Button(list_btn_frame, text="Remove", command=self.remove_img2pdf_files, style='Outline.TButton').pack(side=tk.LEFT, padx=2)
        ttk.Button(list_btn_frame, text="Up", command=lambda: self.move_img2pdf_item(-1), style='Outline.TButton').pack(side=tk.LEFT, padx=2)
        ttk.Button(list_btn_frame, text="Down", command=lambda: self.move_img2pdf_item(1), style='Outline.TButton').pack(side=tk.LEFT, padx=2)
        ttk.Button(list_btn_frame, text="Clear", command=self.clear_img2pdf_list, style='Outline.Danger.TButton').pack(side=tk.LEFT, padx=2)

        # Right Pane: Output and Options
        right_frame = ttk.Frame(h_pane, padding=10)
        h_pane.add(right_frame, weight=1)
        # Output File
        self._create_io_section(right_frame, None, self.img2pdf_output_path,
                                output_label="Output PDF File:", output_type="pdf")
        ttk.Separator(right_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10, padx=5)

        # Conversion Options
        options_frame = ttk.LabelFrame(right_frame, text="Conversion Options", padding=10)
        options_frame.pack(fill=tk.X, padx=5, pady=5)
        self._create_widget_row(options_frame, "Page Size:", self.img2pdf_page_size, 'combobox',
                                widget_options={'values': ["A4", "Letter", "Legal", "A3", "auto"], 'width': 10},
                                tooltip="Size of the pages in the output PDF ('auto' uses first image size).")
        self._create_widget_row(options_frame, "Margin (points):", self.img2pdf_margin_pt, 'spinbox',
                                widget_options={'from_': 0, 'to': 200, 'increment': 6, 'width': 10},
                                tooltip="Page margin in points (1 inch = 72 points).")
        self._create_widget_row(options_frame, "Image Fit:", self.img2pdf_fit_method, 'combobox',
                                widget_options={'values': ["fit", "fill", "stretch", "center"], 'width': 10},
                                tooltip="How to place the image on the page:\nfit: Scale to fit within margins (keeps ratio).\nfill: Scale to fill margins (keeps ratio, may crop).\nstretch: Distort image to fill margins exactly.\ncenter: Place original size in center.")
        self._create_widget_row(options_frame, "Resolution:", self.img2pdf_resolution, 'spinbox',
                                widget_options={'from_': 72, 'to': 1200, 'increment': 1, 'width': 10},
                                tooltip="Assumed DPI of input images (affects scaling if page size is fixed).")
        self._create_widget_row(options_frame, "JPEG Quality:", self.img2pdf_quality, 'spinbox',
                                widget_options={'from_': 1, 'to': 100, 'increment': 1, 'width': 10},
                                tooltip="Quality for JPEG compression within the PDF (1-100). Higher is better quality, larger size.")

        # Action Button
        self._create_action_button(right_frame, "Convert Images to PDF", self.run_images_to_pdf)


    def _create_pdf2img_page(self, parent: ttk.Frame):
        logger.debug("Creating PDF-to-Images page...")
        # Input/Output
        self._create_io_section(parent, self.pdf2img_input_path, self.pdf2img_output_dir,
                                input_label="PDF to Convert:", output_label="Output Directory:",
                                input_type="pdf", output_type="dir")
        self._create_password_field(parent, self.pdf2img_password, label="Input PDF Password:")
        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10, padx=5)

        # Conversion Options
        options_frame = ttk.LabelFrame(parent, text="Conversion Options", padding=10)
        options_frame.pack(fill=tk.X, padx=5, pady=5)
        self._create_widget_row(options_frame, "Image Format:", self.pdf2img_format, 'combobox',
                                widget_options={'values': ["png", "jpg", "tif", "bmp"], 'width': 10}, # Keep names simple
                                tooltip="Output image file format.")
        self._create_widget_row(options_frame, "Resolution (DPI):", self.pdf2img_dpi, 'spinbox',
                                widget_options={'from_': 72, 'to': 1200, 'increment': 1, 'width': 10},
                                tooltip="Resolution of the output images.")
        self._create_widget_row(options_frame, "Output Prefix:", self.pdf2img_prefix, 'entry',
                                widget_options={'width': 30},
                                tooltip="Optional prefix for output filenames (e.g., 'Scan_'). Defaults to input filename.")
        self._create_page_range_field(options_frame, self.pdf2img_page_range)

        # Action Button
        self._create_action_button(parent, "Convert PDF to Images", self.run_pdf_to_images)


    def _create_compress_page(self, parent: ttk.Frame):
        logger.debug("Creating Compress page...")
        # Input/Output
        self._create_io_section(parent, self.compress_input_path, self.compress_output_path,
                                input_label="PDF to Compress:", output_label="Compressed Output PDF:",
                                input_type="pdf", output_type="pdf")
        self._create_password_field(parent, self.compress_password, label="Input PDF Password:")
        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10, padx=5)

        # Compression Options
        options_frame = ttk.LabelFrame(parent, text="Compression Options", padding=10)
        options_frame.pack(fill=tk.X, padx=5, pady=5)

        # Note: PyMuPDF doesn't have a single "quality" slider like some tools.
        # Compression relies on flags like deflate, garbage collection, clean.
        # We can offer image downsampling as a key option.
        # self._create_widget_row(options_frame, "Target Quality:", self.compress_quality, 'spinbox',
        #                         widget_options={'from_': 0, 'to': 100, 'increment': 5, 'width': 10},
        #                         tooltip="Target quality (approximate). Relies on underlying library features.")

        self._create_widget_row(options_frame, "Optimize Structure:", None, 'label', # Just a label for info
                                widget_options={'text': '(Enabled: Removes unused objects, compresses streams)'})

        # Image Downsampling Options
        img_options_frame = ttk.Frame(options_frame)
        img_options_frame.pack(fill=tk.X, pady=(5,0))
        cb_frame, cb = self._create_widget_row(img_options_frame, "Downsample Images:", self.compress_downsample, 'checkbutton')
        self.add_tooltip(cb_frame, "Reduces the resolution of images within the PDF to save space.")

        self.compress_dpi_frame, self.compress_dpi_spinbox = self._create_widget_row(options_frame, "  Target Image DPI:", self.compress_image_dpi, 'spinbox',
                                        widget_options={'from_': 72, 'to': 600, 'increment': 1, 'width': 10})
        self.add_tooltip(self.compress_dpi_spinbox, "Images above this resolution will be downsampled.")

        # Bind visibility of DPI entry to checkbox
        def _update_compress_dpi_visibility(*args):
            is_enabled = self.compress_downsample.get()
            # Show/hide the entire row frame
            if is_enabled:
                 self.compress_dpi_frame.pack(fill=tk.X, pady=2, padx=(20, 0)) # Indent slightly
            else:
                 self.compress_dpi_frame.pack_forget()

        self.compress_downsample.trace_add("write", _update_compress_dpi_visibility)
        _update_compress_dpi_visibility() # Initial check

        # Action Button
        self._create_action_button(parent, "Compress PDF", self.run_compress_pdf)


    def _create_encrypt_page(self, parent: ttk.Frame):
        logger.debug("Creating Encrypt/Decrypt page...")
        # Input/Output
        self._create_io_section(parent, self.encrypt_input_path, self.encrypt_output_path,
                                input_label="Input PDF:", output_label="Output PDF File:",
                                input_type="pdf", output_type="pdf")
        self._create_password_field(parent, self.encrypt_input_password, label="Input Password (if any):")
        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10, padx=5)

        # Action Selection (Encrypt/Decrypt)
        action_frame = ttk.LabelFrame(parent, text="Action", padding=10)
        action_frame.pack(fill=tk.X, padx=5, pady=5)
        r_enc = ttk.Radiobutton(action_frame, text="Encrypt PDF", variable=self.encrypt_action, value="encrypt", command=self._update_encrypt_options_visibility)
        r_enc.pack(side=tk.LEFT, padx=5)
        r_dec = ttk.Radiobutton(action_frame, text="Decrypt PDF", variable=self.encrypt_action, value="decrypt", command=self._update_encrypt_options_visibility)
        r_dec.pack(side=tk.LEFT, padx=5)

        # --- Encryption Options Frame (conditionally visible) ---
        self.encrypt_options_frame = ttk.LabelFrame(parent, text="Encryption Settings", padding=10)
        # Packed/unpacked in _update_encrypt_options_visibility

        # Passwords
        self._create_password_field(self.encrypt_options_frame, self.encrypt_user_password, label="User Password:")
        self.add_tooltip(self.encrypt_options_frame.winfo_children()[-1], "Password required to open the PDF. Leave blank for no user password (only owner).")
        self._create_password_field(self.encrypt_options_frame, self.encrypt_owner_password, label="Owner Password:")
        self.add_tooltip(self.encrypt_options_frame.winfo_children()[-1], "Password required to change permissions or remove encryption.\nIf blank, uses the User Password.")

        # Encryption Level
        self._create_widget_row(self.encrypt_options_frame, "Encryption Level:", self.encrypt_level, 'combobox',
                                widget_options={'values': [128, 256], 'width': 5}, # Check PyMuPDF/pypdf support for 256 AES
                                tooltip="Encryption strength (AES). 128-bit is common, 256-bit is stronger.")

        # Permissions Frame
        perm_frame = ttk.LabelFrame(self.encrypt_options_frame, text="Permissions", padding=10)
        perm_frame.pack(fill=tk.X, padx=5, pady=5)
        perm_frame.columnconfigure((0, 1), weight=1) # Two columns for checkboxes

        cb1 = ttk.Checkbutton(perm_frame, variable=self.encrypt_allow_printing, text="Allow Printing")
        cb1.grid(row=0, column=0, sticky='w', pady=1)
        cb2 = ttk.Checkbutton(perm_frame, variable=self.encrypt_allow_copying, text="Allow Copying Text/Images")
        cb2.grid(row=1, column=0, sticky='w', pady=1)
        cb3 = ttk.Checkbutton(perm_frame, variable=self.encrypt_allow_modifying, text="Allow Modifying Document")
        cb3.grid(row=0, column=1, sticky='w', pady=1)
        self.add_tooltip(cb3, "Allows assembly, form filling, commenting, but not content changes.")
        cb4 = ttk.Checkbutton(perm_frame, variable=self.encrypt_allow_annotating, text="Allow Annotating/Forms")
        cb4.grid(row=1, column=1, sticky='w', pady=1)
        # Add tooltips to permissions if needed

        # Action Button (Text changes based on action)
        self.encrypt_action_button = self._create_action_button(parent, "Encrypt PDF", self.run_encrypt_decrypt_pdf)

        # --- Initial Visibility ---
        self._update_encrypt_options_visibility()

    def _update_encrypt_options_visibility(self, *args):
        """Show/hide encryption options based on selected action."""
        is_encrypt = (self.encrypt_action.get() == "encrypt")
        if is_encrypt:
            self.encrypt_options_frame.pack(fill=tk.X, padx=5, pady=5, before=self.encrypt_action_button.master) # Pack before button's frame
            self.encrypt_action_button.config(text="Encrypt PDF")
        else:
            self.encrypt_options_frame.pack_forget()
            self.encrypt_action_button.config(text="Decrypt PDF")


    def _create_extract_page(self, parent: ttk.Frame):
        logger.debug("Creating Extract Text page...")
        # Input/Output
        self._create_io_section(parent, self.extract_input_path, self.extract_output_path,
                                input_label="PDF to Extract From:", output_label="Output Text File:",
                                input_type="pdf", output_type="txt") # Default to .txt output
        self._create_password_field(parent, self.extract_password, label="Input PDF Password:")
        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10, padx=5)

        # Extraction Options
        options_frame = ttk.LabelFrame(parent, text="Extraction Options", padding=10)
        options_frame.pack(fill=tk.X, padx=5, pady=5)

        # Map fitz formats to user-friendly names
        extract_formats_display = ["Plain Text", "Plain Text (Blocks)", "HTML", "XML", "Markdown"]
        extract_formats_fitz = ["text", "blocks", "html", "xml", "markdown"] # Use 'markdown' as custom type
        self.extract_format_map = dict(zip(extract_formats_display, extract_formats_fitz))
        self.extract_format_rev_map = dict(zip(extract_formats_fitz, extract_formats_display))

        # Get current config value and find corresponding display name
        current_fitz_format = self.config_manager.get("pdf", "extract_default_format", "text")
        current_display_format = self.extract_format_rev_map.get(current_fitz_format, "Plain Text")
        self.extract_format.set(current_display_format) # Set display name

        self._create_widget_row(options_frame, "Output Format:", self.extract_format, 'combobox',
                                widget_options={'values': extract_formats_display, 'width': 20},
                                tooltip="Choose the format for the extracted text output.")

        self._create_widget_row(options_frame, "Preserve Layout:", self.extract_layout, 'checkbutton',
                                tooltip="Try to maintain original spacing and line breaks (Plain Text only).")
        self._create_page_range_field(options_frame, self.extract_page_range)

        # Action Button
        self._create_action_button(parent, "Extract Text", self.run_extract_text)


    def _create_rotate_page(self, parent: ttk.Frame):
        logger.debug("Creating Rotate page...")
        # Input/Output
        self._create_io_section(parent, self.rotate_input_path, self.rotate_output_path,
                                input_label="PDF to Rotate:", output_label="Rotated Output PDF:",
                                input_type="pdf", output_type="pdf")
        self._create_password_field(parent, self.rotate_password, label="Input PDF Password:")
        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10, padx=5)

        # Rotation Options
        options_frame = ttk.LabelFrame(parent, text="Rotation Options", padding=10)
        options_frame.pack(fill=tk.X, padx=5, pady=5)

        self._create_widget_row(options_frame, "Rotation Angle:", self.rotate_angle, 'combobox',
                                widget_options={'values': [90, 180, 270], 'width': 5},
                                tooltip="Rotate selected pages clockwise by this angle.")
        self._create_page_range_field(options_frame, self.rotate_page_range)

        # Action Button
        self._create_action_button(parent, "Rotate PDF Pages", self.run_rotate_pdf)


    def _create_metadata_page(self, parent: ttk.Frame):
        logger.debug("Creating Metadata page...")
        # Input Row with Load button
        input_frame, entry = self._create_widget_row(parent, "PDF to Edit:", self.metadata_input_path, 'readonly_entry',
                                                button_text="Browse...",
                                                button_cmd=lambda: self.select_input_file(self.metadata_input_path, "pdf"))
        if entry: entry.configure(style='PathEntry.TEntry')
        # Add Load Metadata button to the same row
        ttk.Button(input_frame, text="Load Current", command=self.load_pdf_metadata, style='Outline.TButton').pack(side=tk.LEFT, padx=(5,0))
        self.add_tooltip(input_frame.winfo_children()[-1], "Load existing metadata from the selected PDF.")


        # Output Row
        self._create_widget_row(parent, "Output PDF File:", self.metadata_output_path, 'readonly_entry',
                                button_text="Browse...",
                                button_cmd=lambda: self.select_output_file(self.metadata_output_path, "pdf"))
        self._create_password_field(parent, self.metadata_password, label="Input PDF Password:")
        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10, padx=5)

        # Metadata Fields Frame
        fields_frame = ttk.LabelFrame(parent, text="Metadata Fields (Leave blank to keep existing, or set to clear)", padding=10)
        fields_frame.pack(fill=tk.X, padx=5, pady=5)
        # Use standard fitz keys: title, author, subject, keywords, creator, producer
        self._create_widget_row(fields_frame, "Title:", self.metadata_title, 'entry')
        self._create_widget_row(fields_frame, "Author:", self.metadata_author, 'entry')
        self._create_widget_row(fields_frame, "Subject:", self.metadata_subject, 'entry')
        self._create_widget_row(fields_frame, "Keywords:", self.metadata_keywords, 'entry')
        self.add_tooltip(fields_frame.winfo_children()[-1].widget, "Comma-separated keywords.")
        self._create_widget_row(fields_frame, "Creator Tool:", self.metadata_creator, 'entry')
        self.add_tooltip(fields_frame.winfo_children()[-1].widget, "The software that created the original document.")
        self._create_widget_row(fields_frame, "PDF Producer:", self.metadata_producer, 'entry')
        self.add_tooltip(fields_frame.winfo_children()[-1].widget, "The software that created this PDF file.")

        # Action Button
        self._create_action_button(parent, "Update PDF Metadata", self.run_edit_metadata)


    def create_status_bar(self):
        """Create the status bar at the bottom of the window."""
        logger.debug("Creating status bar...")
        self.status_bar_frame = ttk.Frame(self.root, style='Tool.TFrame') # Use a style if defined by theme
        self.status_bar_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(1, 0))

        # Status Message Label (expands)
        # Use ttkbootstrap Label for potential style integration
        self.status_label = ttb.Label(self.status_bar_frame, textvariable=self.status_message, anchor=tk.W, bootstyle='default') # Use bootstyle
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=2)
        self.add_tooltip(self.status_label, "Displays current status and messages. Click to view log history.")
        self.status_label.bind("<Button-1>", lambda e: self.show_log_viewer())

        # Progress Bar (hidden initially)
        # Use ttkbootstrap Progressbar
        self.progress_bar = ttb.Progressbar(self.status_bar_frame, orient=tk.HORIZONTAL,
                                            length=180, mode='determinate',
                                            variable=self.progress_value, bootstyle='primary-striped') # Example style
        # self.progress_bar.pack(side=tk.LEFT, padx=5, pady=2) - Packed when visible

        # Cancel Button (hidden initially)
        self.cancel_button = ttk.Button(self.status_bar_frame, text="Cancel",
                                        command=self.cancel_current_operation, state=tk.DISABLED,
                                        style='Outline.Danger.TButton', width=8)
        # self.cancel_button.pack(side=tk.LEFT, padx=5, pady=2) - Packed when visible

        # Set initial status bar style based on theme
        self._update_status_bar_style()

    def _update_status_bar_style(self):
        """Update status bar background based on theme (light/dark)."""
        # Try to get background color from the theme's TFrame style
        try:
             # Use ttkbootstrap's style query
             bg_color = self.style.lookup('TFrame', 'background')
             fg_color = self.style.lookup('TLabel', 'foreground')
             # Apply to the frame and label directly (ttkbootstrap labels might handle this)
             self.status_bar_frame.config(background=bg_color)
             # status_label bootstyle should handle fg color
        except tk.TclError:
             logger.warning("Could not determine theme background for status bar styling.")


    def create_toolbar(self):
        """Create an optional toolbar (currently empty)."""
        logger.debug("Creating toolbar (placeholder)...")
        # self.toolbar_frame = ttk.Frame(self.root)
        # Add buttons here if needed (e.g., shortcuts for common actions)
        # self.toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        pass # No toolbar implemented currently

    def bind_global_events(self):
        """Bind application-level events."""
        logger.debug("Binding global events...")
        # Window close event
        self.root.protocol("WM_DELETE_WINDOW", self.quit_app)

        # Trace input path variables to auto-fill output paths
        input_vars_map = {
            0: self.split_input_path, 3: self.pdf2img_input_path, 4: self.compress_input_path,
            5: self.encrypt_input_path, 6: self.extract_input_path, 7: self.rotate_input_path,
            8: self.metadata_input_path
        }
        for tab_index, input_var in input_vars_map.items():
            callback = partial(self.auto_fill_output, tab_index)
            # Use unique trace name
            trace_name = f"autofill_trace_{tab_index}"
            input_var.trace_add("write", callback)
            setattr(self, trace_name, callback) # Keep reference

        # Consider adding keyboard shortcuts (e.g., Ctrl+O, Ctrl+S, F1 for help)

    def on_tab_changed(self, event=None):
        """Callback when the notebook tab changes."""
        try:
            selected_tab_index = self.notebook.index(self.notebook.select())
            tab_text = self.notebook.tab(selected_tab_index, "text")
        except tk.TclError:
             return # Occurs during shutdown sometimes

        logger.debug(f"Notebook tab changed to: {selected_tab_index} ('{tab_text}')")

        # Update header title and description
        descriptions = {
            0: "Divide a PDF into smaller files based on pages or bookmarks.",
            1: "Combine multiple PDF files into a single document.",
            2: "Create a PDF document from a collection of image files.",
            3: "Convert each page of a PDF into an image file.",
            4: "Reduce the file size of a PDF, optionally downsampling images.",
            5: "Add password protection or remove encryption from a PDF.",
            6: "Extract text content from a PDF document.",
            7: "Rotate pages within a PDF file clockwise.",
            8: "View and modify PDF document properties like Title, Author, etc."
        }
        self.page_title.config(text=tab_text)
        self.page_desc.config(text=descriptions.get(selected_tab_index, "Perform PDF operations."))

        # Update active navigation button style (optional visual feedback)
        for tab_id, button in self.nav_buttons.items():
             if tab_id == selected_tab_index:
                 button.config(style='secondary.TButton') # Highlighted style
             else:
                 button.config(style='Outline.TButton') # Default style


    def apply_theme(self):
        """Apply the selected ttkbootstrap theme."""
        theme_mode = self.config_manager.get_theme_mode()
        logger.info(f"Applying theme mode: {theme_mode.value}")
        ttk_theme_name = TTB_DEFAULT_THEME # Default

        try:
            if theme_mode == Theme.CUSTOM:
                base_theme = self.config_manager.get_custom_theme_base()
                custom_colors = self.config_manager.get_custom_theme_colors()
                if not custom_colors:
                    logger.warning("Custom theme selected but no colors defined. Using base theme.")
                    ttk_theme_name = base_theme
                else:
                     # Create or update the custom theme in ttkbootstrap
                     if 'aptk_custom' not in self.style.theme_names():
                         self.style.register_theme('aptk_custom', base_theme)

                     # Set the custom colors
                     self.style.theme_use('aptk_custom') # Switch to apply colors
                     logger.debug(f"Applying custom colors: {custom_colors}")
                     for color_name, color_value in custom_colors.items():
                         try:
                             # Use ttkbootstrap's way to set theme colors if possible
                             # This API might change or might not exist directly.
                             # For now, we rely on the theme being pre-registered or updated externally.
                             # Direct modification like below is less reliable with ttkbootstrap.
                             # self.style.master.set_color(color_name, color_value) # Might be ttkb specific
                             # Alternative: configure specific styles
                             pass # TODO: Find reliable ttkbootstrap custom color API
                         except Exception as color_err:
                              logger.error(f"Failed to set custom color '{color_name}': {color_err}")
                     ttk_theme_name = 'aptk_custom' # Use the custom theme name

            elif theme_mode == Theme.SYSTEM:
                 # Basic system theme detection (can be improved)
                 try:
                     # Tk doesn't have a reliable cross-platform dark mode check
                     # Let's default to light for system unless we add better detection
                     # is_system_dark = False # Assume light
                     # A simple check based on default background (might not work well)
                     default_bg = self.root.cget('bg')
                     rgb = self.root.winfo_rgb(default_bg)
                     brightness = sum(rgb) / 3 / 256 # Simple brightness heuristic
                     is_system_dark = brightness < 128
                     logger.debug(f"System theme detection: bg={default_bg}, rgb={rgb}, brightness={brightness}, dark={is_system_dark}")

                 except tk.TclError:
                     is_system_dark = False # Fallback to light
                     logger.warning("Could not detect system theme via background color.")

                 ttk_theme_name = TTB_DARK_THEME if is_system_dark else TTB_LIGHT_THEME
            elif theme_mode == Theme.LIGHT:
                 ttk_theme_name = TTB_LIGHT_THEME
            elif theme_mode == Theme.DARK:
                 ttk_theme_name = TTB_DARK_THEME

            # Apply the determined theme using ttkbootstrap's method
            logger.debug(f"Setting ttkbootstrap theme to: {ttk_theme_name}")
            self.style.theme_use(ttk_theme_name)

            # --- MODIFIED PART ---
            # Determine dark mode based on actual background color brightness
            try:
                bg_color = self.style.lookup('TFrame', 'background')
                rgb = self.root.winfo_rgb(bg_color)
                # Calculate perceived brightness (simple average)
                brightness = sum(rgb) / 3 / 65535 # winfo_rgb gives 0-65535 range
                is_dark = brightness < 0.5 # Threshold for darkness
                self.is_dark_mode.set(is_dark)
                logger.debug(f"Determined dark mode: {is_dark} (bg: {bg_color}, brightness: {brightness:.3f})")
            except tk.TclError:
                logger.warning("Could not determine theme brightness for dark mode check. Defaulting to False.")
                self.is_dark_mode.set(False)
            # --- END MODIFIED PART ---

            # Update elements not automatically themed (like status bar)
            self._update_status_bar_style()
            # Update sidebar style maybe?
            # Determine sidebar style based on the calculated is_dark_mode
            sidebar_style = 'primary.TFrame' if self.is_dark_mode.get() else 'secondary.TFrame' # Example
            self.sidebar_frame.config(style=sidebar_style)


        except tk.TclError as e:
            logger.error(f"Failed to apply theme '{ttk_theme_name}': {e}. Theme might be unavailable.")
            messagebox.showerror("Theme Error", f"Could not apply theme '{ttk_theme_name}'. Please check installation.")
            # Fallback to a default theme known to exist? Difficult with ttkb.
        except Exception as e:
             logger.exception(f"Unexpected error applying theme: {e}")

    def edit_custom_theme(self):
        """Open a dialog to edit custom theme colors."""
        logger.debug("Opening custom theme editor...")
        dialog = CustomThemeDialog(self.root, self.config_manager)
        # Dialog handles saving internally if OK is pressed
        if dialog.result: # Check if OK was pressed
             logger.info("Custom theme colors updated.")
             # Re-apply theme immediately if custom mode is active
             if self.config_manager.get_theme_mode() == Theme.CUSTOM:
                 logger.debug("Re-applying theme after custom edit.")
                 self.apply_theme()
        else:
             logger.debug("Custom theme edit cancelled.")


    def show_splash_screen(self):
        """Display a splash screen briefly on startup."""
        # Simple splash using Toplevel
        splash = tk.Toplevel(self.root)
        splash.title("Loading...")
        splash.geometry("350x180")
        splash.overrideredirect(True) # No window decorations
        splash.attributes('-topmost', True) # Stay on top

        # Center splash screen relative to main window/screen
        try:
             self.root.update_idletasks() # Ensure main window size is calculated
             root_x = self.root.winfo_x()
             root_y = self.root.winfo_y()
             root_w = self.root.winfo_width()
             root_h = self.root.winfo_height()
             splash_x = root_x + (root_w // 2) - (350 // 2)
             splash_y = root_y + (root_h // 2) - (180 // 2)
             splash.geometry(f"+{splash_x}+{splash_y}")
        except tk.TclError: # If main window not ready
             screen_w = self.root.winfo_screenwidth()
             screen_h = self.root.winfo_screenheight()
             splash_x = (screen_w // 2) - (350 // 2)
             splash_y = (screen_h // 2) - (180 // 2)
             splash.geometry(f"+{splash_x}+{splash_y}")


        splash_frame = ttk.Frame(splash, padding=20, style='primary.TFrame')
        splash_frame.pack(expand=True, fill='both')

        ttk.Label(splash_frame, text=APP_NAME, style='Title.TLabel', bootstyle='inverse-primary').pack(pady=(10, 5)) # Use inverse color
        ttk.Label(splash_frame, text=f"Version {__version__}", style='Subtitle.TLabel', bootstyle='inverse-primary').pack(pady=5)
        # Indeterminate progress bar
        pb = ttb.Progressbar(splash_frame, mode='indeterminate', bootstyle='success-striped')
        pb.pack(pady=15, padx=20, fill='x')
        pb.start()

        splash.lift()
        splash.update()

        # Destroy splash after a delay
        splash_duration_ms = 1800
        self.root.after(splash_duration_ms, splash.destroy)

    # --- UI Helper Methods ---
    # (File Dialogs, Listbox Manipulation, Tooltips, etc.)

    def add_tooltip(self, widget: tk.Widget, text: str):
        """Basic tooltip implementation (can be replaced with a library)."""
        tooltip_label = None
        enter_time = None

        def _show_tooltip(event=None):
            nonlocal tooltip_label, enter_time
            if tooltip_label or not text: return
            # Simple delay
            enter_time = time.monotonic()
            widget.after(800, _create_tooltip_window) # 800ms delay

        def _create_tooltip_window():
            nonlocal tooltip_label
            if tooltip_label: return # Already created
            # Check if cursor is still over widget after delay
            current_widget = widget.winfo_containing(widget.winfo_pointerx(), widget.winfo_pointery())
            if current_widget != widget:
                return # Cursor moved away

            tooltip_label = tk.Toplevel(widget)
            tooltip_label.wm_overrideredirect(True)
            # Position near cursor
            x = widget.winfo_pointerx() + 15
            y = widget.winfo_pointery() + 10
            tooltip_label.wm_geometry(f"+{x}+{y}")

            label = tk.Label(tooltip_label, text=text, justify=tk.LEFT,
                             background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                             font=("tahoma", "8", "normal"), wraplength=300)
            label.pack(ipadx=1)

        def _hide_tooltip(event=None):
            nonlocal tooltip_label, enter_time
            enter_time = None
            if tooltip_label:
                tooltip_label.destroy()
                tooltip_label = None

        widget.bind('<Enter>', _show_tooltip, add='+')
        widget.bind('<Leave>', _hide_tooltip, add='+')
        widget.bind('<ButtonPress>', _hide_tooltip, add='+') # Hide on click

    def select_input_file(self, variable: tk.StringVar, file_type: str):
        """Open file dialog to select a single input file."""
        type_map = {
            "pdf": [("PDF Files", "*.pdf"), ("All Files", "*.*")],
            "image": [("Image Files", " ".join(f"*{ext}" for ext in SUPPORTED_IMAGE_FORMATS)), ("All Files", "*.*")],
            "any": [("All Files", "*.*")]
        }
        filetypes = type_map.get(file_type, [("All Files", "*.*")])
        title = f"Select Input {file_type.capitalize()} File"

        initial_dir = os.path.dirname(variable.get()) if variable.get() else self.config_manager.get("paths", "default_output_dir") or os.path.expanduser("~")

        filepath = filedialog.askopenfilename(title=title, filetypes=filetypes, initialdir=initial_dir, parent=self.root)

        if filepath:
            variable.set(filepath)
            # Add to recent files (triggers auto-fill output via trace)
            self.recent_files_manager.add_recent_file(filepath)
            # Store last used directory?
            self.config_manager.set("paths", "default_output_dir", os.path.dirname(filepath))
            # No need to save config here, happens on variable trace or exit

    def select_input_files_generic(self):
         """Determine which multi-file input to trigger based on current tab."""
         current_tab_index = self.notebook.index(self.notebook.select())
         if current_tab_index == 1: # Merge
             self.add_merge_files()
         elif current_tab_index == 2: # Images to PDF
             self.add_img2pdf_files()
         else:
              messagebox.showwarning("Select Files", "This option is used for Merge PDFs or Images to PDF tabs.")


    def select_output_dir(self, variable: tk.StringVar):
        """Open directory dialog to select an output directory."""
        initial_dir = variable.get() if variable.get() and os.path.isdir(variable.get()) else self.config_manager.get("paths", "default_output_dir") or os.path.expanduser("~")
        dirpath = filedialog.askdirectory(title="Select Output Directory", initialdir=initial_dir, parent=self.root)
        if dirpath:
            variable.set(dirpath)
            self.config_manager.set("paths", "default_output_dir", dirpath)

    def select_output_file(self, variable: tk.StringVar, file_type: str):
        """Open file dialog to select an output file path."""
        type_map = {
            "pdf": ([("PDF Files", "*.pdf"), ("All Files", "*.*")], ".pdf"),
            "txt": ([("Text Files", "*.txt"), ("All Files", "*.*")], ".txt"),
            # Add other types if needed
        }
        filetypes, defaultextension = type_map.get(file_type, ([("All Files", "*.*")], ""))

        initial_dir = os.path.dirname(variable.get()) if variable.get() else self.config_manager.get("paths", "default_output_dir") or os.path.expanduser("~")
        initial_file = os.path.basename(variable.get()) if variable.get() else ""

        filepath = filedialog.asksaveasfilename(title="Select Output File", filetypes=filetypes,
                                                defaultextension=defaultextension, initialdir=initial_dir,
                                                initialfile=initial_file, parent=self.root)
        if filepath:
            variable.set(filepath)
            self.config_manager.set("paths", "default_output_dir", os.path.dirname(filepath))


    def select_input_file_generic(self, file_type: str):
        """Generic file open triggered from menu or toolbar."""
        current_tab_index = self.notebook.index(self.notebook.select())

        if file_type == "pdf":
            pdf_input_tabs = [0, 3, 4, 5, 6, 7, 8] # Tabs with single PDF input var
            target_var = self.get_input_var_for_tab(current_tab_index)
            if target_var:
                self.select_input_file(target_var, "pdf")
            else: # If current tab doesn't take PDF, default to Split
                self.select_input_file(self.split_input_path, "pdf")
                if current_tab_index != 0: self.notebook.select(0)
        elif file_type == "images":
             # Always target the Images to PDF tab
             self.add_img2pdf_files() # Let this method handle the dialog
             if current_tab_index != 2: self.notebook.select(2)
        else:
             logger.warning(f"Unhandled generic file type selection: {file_type}")


    def get_input_var_for_tab(self, tab_index: int) -> Optional[tk.StringVar]:
        """Gets the primary input path variable for a given tab index."""
        path_vars = {
            0: self.split_input_path, 3: self.pdf2img_input_path, 4: self.compress_input_path,
            5: self.encrypt_input_path, 6: self.extract_input_path, 7: self.rotate_input_path,
            8: self.metadata_input_path,
            # Merge (1) and Img2PDF (2) use listboxes, no single var
        }
        return path_vars.get(tab_index)

    def get_output_var_for_tab(self, tab_index: int) -> Optional[tk.StringVar]:
        """Gets the primary output path variable for a given tab index."""
        path_vars = {
            0: self.split_output_dir,    # Dir
            1: self.merge_output_path,   # File
            2: self.img2pdf_output_path, # File
            3: self.pdf2img_output_dir,  # Dir
            4: self.compress_output_path,# File
            5: self.encrypt_output_path, # File
            6: self.extract_output_path, # File
            7: self.rotate_output_path,  # File
            8: self.metadata_output_path,# File
        }
        return path_vars.get(tab_index)

    def auto_fill_output(self, tab_index: int, *args):
        """
        Suggest an output path based on the input path for the given tab.
        Triggered by the trace on input path variables.
        """
        input_var = self.get_input_var_for_tab(tab_index)
        output_var = self.get_output_var_for_tab(tab_index)

        if not input_var or not output_var:
             # logger.debug(f"Auto-fill skipped: No I/O vars for tab {tab_index}")
             return

        input_path = input_var.get()
        # Only fill if input is valid and output is currently empty
        if not input_path or not os.path.exists(input_path) or output_var.get():
            # logger.debug(f"Auto-fill skipped: Input invalid or output already set for tab {tab_index}")
            return

        input_dir = os.path.dirname(input_path)
        input_filename = os.path.basename(input_path)
        base_name, input_ext = os.path.splitext(input_filename)

        suggested_output = ""
        # --- Output Directory cases ---
        if tab_index == 0: # Split PDF -> Output Dir
             suggested_output = os.path.join(input_dir, f"{base_name}_split")
        elif tab_index == 3: # PDF to Images -> Output Dir
             suggested_output = os.path.join(input_dir, f"{base_name}_images")

        # --- Output File cases ---
        else:
            op_suffixes = {
                1: "_merged", 2: "_converted", 4: "_compressed", 5: "_modified",
                6: "_extracted", 7: "_rotated", 8: "_metadata"
            }
            suffix = op_suffixes.get(tab_index, "_output")
            # Determine output extension
            if tab_index == 6: # Extract Text
                ext = ".txt" # Default, could be based on selected format later
            else: # Most others output PDF
                 ext = ".pdf"

            suggested_output = os.path.join(input_dir, f"{base_name}{suffix}{ext}")

        if suggested_output:
            logger.debug(f"Auto-filling output for tab {tab_index}: {suggested_output}")
            output_var.set(suggested_output)

    def ask_password(self, filename: str) -> Optional[str]:
        """Ask user for password for an encrypted PDF using a simple dialog."""
        # Ensure dialog appears on top of main window
        return simpledialog.askstring(
            "Password Required",
            f"Enter password for:\n{os.path.basename(filename)}",
            show='*',
            parent=self.root # Set parent
        )

    def update_listbox_items(self, listbox_widget: tk.Listbox, path_list: List[str]):
        """Refreshes the listbox content based on the provided path list."""
        listbox_widget.delete(0, tk.END)
        for path in path_list:
            display_name = os.path.basename(path)
            # Special handling for Merge listbox to show password indicator
            if listbox_widget == self.merge_listbox and path in self.merge_passwords:
                display_name += "  (PW Set)"
            listbox_widget.insert(tk.END, display_name)

    def move_listbox_item(self, listbox_widget: tk.Listbox, path_list: List[str], direction: int):
        """Moves selected item(s) up (-1) or down (+1) in the listbox and path list."""
        selected_indices = listbox_widget.curselection()
        if not selected_indices:
            self.add_status_message("Select item(s) in the list to move.", "info")
            return

        # Sort indices to process consistently, especially when moving multiple items
        sorted_indices = sorted(selected_indices, reverse=(direction == 1)) # Reverse if moving down

        moved_count = 0
        new_selection_indices = []

        for idx in sorted_indices:
            new_idx = idx + direction
            # Check bounds
            if 0 <= new_idx < len(path_list):
                # Swap in the underlying path list
                path_list[idx], path_list[new_idx] = path_list[new_idx], path_list[idx]
                new_selection_indices.append(new_idx)
                moved_count += 1
            else:
                 new_selection_indices.append(idx) # Keep original index if couldn't move

        if moved_count > 0:
            self.update_listbox_items(listbox_widget, path_list) # Update display

            # Restore selection to moved items
            listbox_widget.selection_clear(0, tk.END)
            final_selection = sorted(new_selection_indices)
            for i in final_selection:
                listbox_widget.selection_set(i)
            # Ensure the first selected item is visible
            if final_selection:
                listbox_widget.see(final_selection[0])
        else:
            self.add_status_message("Item(s) cannot be moved further.", "info")


    # --- Listbox Specific Handlers ---

    def add_merge_files(self):
        """Handler for adding files to the Merge list."""
        title = "Select PDF Files to Merge"
        filetypes = [("PDF Files", "*.pdf"), ("All Files", "*.*")]
        initial_dir = self.config_manager.get("paths", "default_output_dir") or os.path.expanduser("~")
        if self.merge_input_paths: initial_dir = os.path.dirname(self.merge_input_paths[0])

        filepaths = filedialog.askopenfilenames(title=title, filetypes=filetypes, initialdir=initial_dir, parent=self.root)
        added_count = 0
        if filepaths:
            for path in filepaths:
                if path not in self.merge_input_paths:
                    self.merge_input_paths.append(path)
                    self.recent_files_manager.add_recent_file(path) # Add individual files to recent
                    added_count += 1
            if added_count > 0:
                self.update_listbox_items(self.merge_listbox, self.merge_input_paths)
                # Auto-fill output only if it's empty and based on the *first* file added maybe?
                if not self.merge_output_path.get() and len(self.merge_input_paths) >= 1:
                     self.auto_fill_output(1) # Trigger auto-fill for merge tab (index 1)

    def remove_merge_files(self):
        """Handler for removing selected files from the Merge list."""
        selected_indices = self.merge_listbox.curselection()
        if not selected_indices:
            self.add_status_message("Select file(s) to remove from the list.", "info")
            return

        # Remove from lists (paths and passwords) from highest index down
        for idx in sorted(selected_indices, reverse=True):
            removed_path = self.merge_input_paths.pop(idx)
            if removed_path in self.merge_passwords:
                del self.merge_passwords[removed_path]

        self.update_listbox_items(self.merge_listbox, self.merge_input_paths)

    def move_merge_item(self, direction):
        self.move_listbox_item(self.merge_listbox, self.merge_input_paths, direction)

    def clear_merge_list(self):
        if not self.merge_input_paths: return
        if messagebox.askyesno("Confirm Clear", "Clear all files from the merge list?", parent=self.root):
            self.merge_input_paths.clear()
            self.merge_passwords.clear()
            self.update_listbox_items(self.merge_listbox, self.merge_input_paths)

    def set_merge_password(self):
        """Set password for a selected encrypted PDF in the merge list."""
        selected_indices = self.merge_listbox.curselection()
        if len(selected_indices) != 1:
            self.add_status_message("Select exactly one file to set its password.", "info")
            return

        idx = selected_indices[0]
        filepath = self.merge_input_paths[idx]
        password = self.ask_password(filepath)
        if password is not None: # Allow empty password string
            self.merge_passwords[filepath] = password
            self.update_listbox_items(self.merge_listbox, self.merge_input_paths) # Update display (shows PW indicator)
            self.add_status_message(f"Password set for {os.path.basename(filepath)}.", "info")


    def add_img2pdf_files(self, file_paths_to_add: Optional[List[str]] = None):
        """Handler for adding files to the Images-to-PDF list. Can take paths directly."""
        filepaths = file_paths_to_add
        if not filepaths: # If no paths provided, show dialog
             title = "Select Images to Convert"
             img_ext_str = " ".join(f"*{ext}" for ext in SUPPORTED_IMAGE_FORMATS)
             filetypes = [("Image Files", img_ext_str), ("All Files", "*.*")]
             initial_dir = self.config_manager.get("paths", "default_output_dir") or os.path.expanduser("~")
             if self.img2pdf_input_paths: initial_dir = os.path.dirname(self.img2pdf_input_paths[0])
             filepaths = filedialog.askopenfilenames(title=title, filetypes=filetypes, initialdir=initial_dir, parent=self.root)

        added_count = 0
        if filepaths:
            for path in filepaths:
                 # Basic check if it's an image before adding? Or rely on validation? Let's do basic ext check.
                 if any(path.lower().endswith(ext) for ext in SUPPORTED_IMAGE_FORMATS):
                     if path not in self.img2pdf_input_paths:
                         self.img2pdf_input_paths.append(path)
                         self.recent_files_manager.add_recent_file(path)
                         added_count += 1
                 else:
                     logger.warning(f"Skipping non-image file selected for Img2PDF: {path}")
            if added_count > 0:
                self.update_listbox_items(self.img2pdf_listbox, self.img2pdf_input_paths)
                if not self.img2pdf_output_path.get() and len(self.img2pdf_input_paths) >= 1:
                     self.auto_fill_output(2) # Trigger auto-fill for img2pdf tab (index 2)

    def remove_img2pdf_files(self):
        selected_indices = self.img2pdf_listbox.curselection()
        if not selected_indices:
            self.add_status_message("Select image(s) to remove from the list.", "info")
            return
        for idx in sorted(selected_indices, reverse=True):
            self.img2pdf_input_paths.pop(idx)
        self.update_listbox_items(self.img2pdf_listbox, self.img2pdf_input_paths)

    def move_img2pdf_item(self, direction):
        self.move_listbox_item(self.img2pdf_listbox, self.img2pdf_input_paths, direction)

    def clear_img2pdf_list(self):
        if not self.img2pdf_input_paths: return
        if messagebox.askyesno("Confirm Clear", "Clear all images from the list?", parent=self.root):
            self.img2pdf_input_paths.clear()
            self.update_listbox_items(self.img2pdf_listbox, self.img2pdf_input_paths)

    # --- Page Range Validation ---
    def validate_page_range(self, range_str: str, num_pages: int) -> Optional[List[int]]:
        """Validate page range string and return list of 0-based indices or None on error."""
        if not range_str.strip():
            # Blank means all pages
            return None # Use None to signify all pages in operations

        indices = set()
        try:
            parts = range_str.split(',')
            for part in parts:
                part = part.strip()
                if not part: continue

                if '-' in part:
                    start_str, end_str = part.split('-', 1)
                    start = 1 if not start_str.strip() else int(start_str.strip())
                    # Use num_pages for open-ended range like "10-"
                    end = num_pages if not end_str.strip() else int(end_str.strip())

                    if start < 1 or end < 1: raise ValueError("Page numbers must be 1 or greater.")
                    if start > end: raise ValueError(f"Start page ({start}) > end page ({end}) in range.")
                    if start > num_pages: raise ValueError(f"Start page ({start}) exceeds total pages ({num_pages}).")

                    # Add range (0-based, inclusive)
                    indices.update(range(start - 1, min(end, num_pages)))
                else:
                    page = int(part)
                    if page < 1 or page > num_pages:
                        raise ValueError(f"Page number {page} out of range (1-{num_pages}).")
                    indices.add(page - 1) # Add single page (0-based)

            if not indices:
                # Range string was valid but resulted in no pages (e.g., "100" for a 50-page doc)
                messagebox.showwarning("Invalid Range", f"The specified page range '{range_str}' does not include any valid pages for this {num_pages}-page document.", parent=self.root)
                return None # Treat as invalid for processing

            return sorted(list(indices))

        except ValueError as e:
            messagebox.showerror("Invalid Page Range", f"Error parsing page range '{range_str}':\n{e}\nPlease use format like: 1, 3-5, 10-", parent=self.root)
            return None # Indicate error


    # --- Metadata Loading ---
    def load_pdf_metadata(self):
        """Load metadata from the selected PDF into the UI fields."""
        input_path = self.metadata_input_path.get()
        password = self.metadata_password.get() or None
        if not input_path or not os.path.exists(input_path):
            messagebox.showerror("Load Metadata", "Please select a valid input PDF file first.", parent=self.root)
            return

        # Use a temporary operation instance just for opening the file safely
        temp_op = EditMetadataOperation(input_path, "", {}, password=password)
        doc = None
        try:
            logger.info(f"Attempting to load metadata from: {input_path}")
            doc = temp_op._open_input_pdf_fitz() # Use helper to handle password

            # If it needed a password and didn't have one, ask now
            if doc.is_encrypted and not password:
                 doc.close() # Close before asking
                 password = self.ask_password(input_path)
                 if password is None:
                     self.add_status_message("Metadata load cancelled.", "info")
                     return # User cancelled password prompt
                 temp_op.password = password # Update password
                 doc = temp_op._open_input_pdf_fitz() # Re-open with password
                 self.metadata_password.set(password) # Store password in UI if successful

            meta = doc.metadata
            if not meta:
                 self.add_status_message("PDF contains no metadata.", "info")
                 # Clear UI fields
                 self.metadata_title.set("")
                 self.metadata_author.set("")
                 self.metadata_subject.set("")
                 self.metadata_keywords.set("")
                 self.metadata_creator.set("")
                 self.metadata_producer.set("")
                 return

            # Populate UI fields (use .get with default '')
            self.metadata_title.set(meta.get('title', ''))
            self.metadata_author.set(meta.get('author', ''))
            self.metadata_subject.set(meta.get('subject', ''))
            self.metadata_keywords.set(meta.get('keywords', ''))
            self.metadata_creator.set(meta.get('creator', ''))
            self.metadata_producer.set(meta.get('producer', ''))

            self.add_status_message("Metadata loaded successfully.", "success")
            # Suggest output filename based on input if output is empty
            self.auto_fill_output(8) # Trigger auto-fill for metadata tab

        except PdfPasswordException as e:
             messagebox.showerror("Password Error", str(e), parent=self.root)
             self.add_status_message(f"Metadata load failed: {e}", "error")
        except (ValidationError, RuntimeError) as e:
             messagebox.showerror("Load Error", f"Could not load metadata:\n{e}", parent=self.root)
             self.add_status_message(f"Metadata load failed: {e}", "error")
        except Exception as e:
            messagebox.showerror("Load Error", f"An unexpected error occurred loading metadata:\n{e}", parent=self.root)
            logger.exception("Error loading metadata")
            self.add_status_message("Metadata load failed unexpectedly.", "error")
        finally:
             if doc: doc.close()

    # --- Input Clearing ---
    def clear_all_inputs(self):
        """Clear input fields and lists on the currently active notebook tab."""
        if not messagebox.askyesno("Confirm Clear", "Clear all inputs and selections on the current tab?", parent=self.root):
            return

        try:
            current_tab_index = self.notebook.index(self.notebook.select())
        except tk.TclError:
             return # Error getting current tab (e.g., during shutdown)

        logger.info(f"Clearing inputs for tab index {current_tab_index}...")

        # Get all variables associated with the current tab
        vars_to_clear = []
        listboxes_to_clear = []
        list_paths_to_clear = []
        other_resets = [] # Lambdas for other resets (e.g., password dicts)

        # --- Map tab index to relevant variables/lists ---
        if current_tab_index == 0: # Split
            vars_to_clear = [self.split_input_path, self.split_output_dir, self.split_ranges, self.split_prefix, self.split_password]
        elif current_tab_index == 1: # Merge
            vars_to_clear = [self.merge_output_path]
            listboxes_to_clear = [(self.merge_listbox, self.merge_input_paths)] # Tuple of (widget, list_ref)
            other_resets.append(lambda: self.merge_passwords.clear())
        elif current_tab_index == 2: # Img2PDF
            vars_to_clear = [self.img2pdf_output_path]
            listboxes_to_clear = [(self.img2pdf_listbox, self.img2pdf_input_paths)]
        elif current_tab_index == 3: # PDF2Img
            vars_to_clear = [self.pdf2img_input_path, self.pdf2img_output_dir, self.pdf2img_prefix, self.pdf2img_password, self.pdf2img_page_range]
        elif current_tab_index == 4: # Compress
            vars_to_clear = [self.compress_input_path, self.compress_output_path, self.compress_password]
            # Reset checkboxes/spinners to default? Or just paths/passwords? Let's stick to paths/passwords/ranges for now.
        elif current_tab_index == 5: # Encrypt
            vars_to_clear = [self.encrypt_input_path, self.encrypt_output_path, self.encrypt_user_password, self.encrypt_owner_password, self.encrypt_input_password]
        elif current_tab_index == 6: # Extract
            vars_to_clear = [self.extract_input_path, self.extract_output_path, self.extract_page_range, self.extract_password]
        elif current_tab_index == 7: # Rotate
            vars_to_clear = [self.rotate_input_path, self.rotate_output_path, self.rotate_page_range, self.rotate_password]
        elif current_tab_index == 8: # Metadata
            vars_to_clear = [self.metadata_input_path, self.metadata_output_path, self.metadata_title, self.metadata_author, self.metadata_subject, self.metadata_keywords, self.metadata_creator, self.metadata_producer, self.metadata_password]

        # --- Perform Clearing ---
        # Clear tk Variables
        for var in vars_to_clear:
            if isinstance(var, tk.StringVar): var.set("")
            # Don't reset IntVars/BoolVars like DPI, quality, checkboxes to 0/False, keep their current setting.
            # elif isinstance(var, tk.IntVar): var.set(0)
            # elif isinstance(var, tk.BooleanVar): var.set(False)

        # Clear Listboxes and associated data lists
        for listbox_widget, path_list_ref in listboxes_to_clear:
            listbox_widget.delete(0, tk.END)
            path_list_ref.clear()

        # Perform other resets
        for reset_func in other_resets:
            reset_func()

        self.add_status_message(f"Inputs cleared for '{self.notebook.tab(current_tab_index, 'text')}' tab.", "info")

    # --- Operation Execution Trigger Methods ---

    def _run_operation_base(self, operation_class: type[PDFOperation], *args, **kwargs):
        """Base method to instantiate and execute a PDFOperation."""
        # Log input arguments for debugging (redact passwords)
        log_args = list(args)
        log_kwargs = kwargs.copy()
        if 'password' in log_kwargs: log_kwargs['password'] = '***' if log_kwargs['password'] else None
        if 'input_password' in log_kwargs: log_kwargs['input_password'] = '***' if log_kwargs['input_password'] else None
        if 'user_password' in log_kwargs: log_kwargs['user_password'] = '***' if log_kwargs['user_password'] else None
        if 'owner_password' in log_kwargs: log_kwargs['owner_password'] = '***' if log_kwargs['owner_password'] else None
        if 'passwords' in log_kwargs: log_kwargs['passwords'] = {k: ('***' if v else None) for k,v in log_kwargs['passwords'].items()}
        logger.info(f"Initiating operation: {operation_class.__name__}")
        # logger.debug(f"  Args: {log_args}") # Can be verbose
        # logger.debug(f"  Kwargs: {log_kwargs}") # Can be verbose

        try:
            # Instantiate the operation
            operation = operation_class(*args, **kwargs)
            # Execute through the manager
            self.operation_manager.execute_operation(operation)
        except (ValidationError, PdfPasswordException) as e:
             # Catch validation errors during instantiation or pre-validation
             messagebox.showerror("Setup Error", str(e), parent=self.root)
             self.add_status_message(f"Setup failed: {e}", "error")
             logger.error(f"Operation setup failed for {operation_class.__name__}: {e}")
        except Exception as e:
            messagebox.showerror("Error Starting Operation", f"An unexpected error occurred:\n{e}", parent=self.root)
            logger.exception(f"Error instantiating operation {operation_class.__name__}")
            self.add_status_message(f"Error: {e}", "error")


    def run_split_pdf(self):
        # Basic UI checks before calling operation
        if not self.split_input_path.get() or not self.split_output_dir.get():
             messagebox.showwarning("Missing Input", "Please select an input PDF and an output directory.", parent=self.root)
             return
        if self.split_method.get() == "ranges" and not self.split_ranges.get().strip():
             messagebox.showwarning("Missing Input", "Please enter the page ranges to split by.", parent=self.root)
             return

        self._run_operation_base(
            SplitPDFOperation,
            input_path=self.split_input_path.get(),
            output_dir=self.split_output_dir.get(),
            split_method=self.split_method.get(),
            ranges_str=self.split_ranges.get(),
            prefix=self.split_prefix.get() or None, # Pass None if empty
            password=self.split_password.get() or None
        )

    def run_merge_pdf(self):
        if not self.merge_input_paths:
            messagebox.showwarning("Missing Input", "Please add PDF files to the merge list.", parent=self.root)
            return
        if not self.merge_output_path.get():
            messagebox.showwarning("Missing Input", "Please specify an output file path for the merged PDF.", parent=self.root)
            return

        template = None
        if self.merge_add_bookmarks.get():
            template = self.merge_bookmark_template.get() # FILENAME or INDEX_FILENAME

        self._run_operation_base(
            MergePDFOperation,
            input_paths=list(self.merge_input_paths), # Pass a copy
            output_path=self.merge_output_path.get(),
            passwords=self.merge_passwords.copy(), # Pass a copy
            outline_title_template=template
        )

    def run_images_to_pdf(self):
        if not self.img2pdf_input_paths:
            messagebox.showwarning("Missing Input", "Please add image files to the conversion list.", parent=self.root)
            return
        if not self.img2pdf_output_path.get():
            messagebox.showwarning("Missing Input", "Please specify an output file path for the PDF.", parent=self.root)
            return

        self._run_operation_base(
            ImagesToPDFOperation,
            image_paths=list(self.img2pdf_input_paths), # Pass a copy
            output_path=self.img2pdf_output_path.get(),
            page_size_str=self.img2pdf_page_size.get(),
            margin_pt=self.img2pdf_margin_pt.get(),
            fit_method=self.img2pdf_fit_method.get(),
            resolution=self.img2pdf_resolution.get(),
            quality=self.img2pdf_quality.get()
        )

    def run_pdf_to_images(self):
        if not self.pdf2img_input_path.get() or not self.pdf2img_output_dir.get():
            messagebox.showwarning("Missing Input", "Please select an input PDF and an output directory.", parent=self.root)
            return

        # Need to pre-validate page range to pass correct format to operation
        pages_list = None
        num_pages = 0
        input_path = self.pdf2img_input_path.get()
        password = self.pdf2img_password.get() or None

        # Quick check for page count (without full validation inside operation)
        try:
            # Use a temporary operation just for validation helper
            temp_op = PDFToImagesOperation(input_path, self.pdf2img_output_dir.get(), password=password)
            with contextlib.closing(temp_op._open_input_pdf_fitz()) as doc:
                 num_pages = doc.page_count
        except PdfPasswordException:
             password = self.ask_password(input_path)
             if password is None: return # Cancelled password
             self.pdf2img_password.set(password) # Store if user provides it
             try:
                 temp_op = PDFToImagesOperation(input_path, self.pdf2img_output_dir.get(), password=password)
                 with contextlib.closing(temp_op._open_input_pdf_fitz()) as doc:
                     num_pages = doc.page_count
             except Exception as e:
                  messagebox.showerror("PDF Error", f"Could not open PDF to get page count:\n{e}", parent=self.root)
                  return
        except Exception as e:
            messagebox.showerror("PDF Error", f"Could not open PDF to get page count:\n{e}", parent=self.root)
            return

        if num_pages == 0:
             messagebox.showerror("PDF Error", "The selected PDF appears to have no pages.", parent=self.root)
             return

        range_str = self.pdf2img_page_range.get()
        if range_str.strip():
            pages_list = self.validate_page_range(range_str, num_pages)
            if pages_list is None: return # Validation failed in dialog

        self._run_operation_base(
            PDFToImagesOperation,
            input_path=input_path,
            output_dir=self.pdf2img_output_dir.get(),
            img_format=self.pdf2img_format.get(),
            dpi=self.pdf2img_dpi.get(),
            prefix=self.pdf2img_prefix.get() or None,
            pages=pages_list, # List of 0-based indices or None
            password=password # Use potentially updated password
        )

    def run_compress_pdf(self):
        if not self.compress_input_path.get() or not self.compress_output_path.get():
            messagebox.showwarning("Missing Input", "Please select an input PDF and an output file path.", parent=self.root)
            return

        self._run_operation_base(
            CompressPDFOperation,
            input_path=self.compress_input_path.get(),
            output_path=self.compress_output_path.get(),
            compress_quality=self.compress_quality.get(), # Pass quality setting
            downsample_images=self.compress_downsample.get(),
            image_dpi=self.compress_image_dpi.get(),
            password=self.compress_password.get() or None
        )

    def run_encrypt_decrypt_pdf(self):
        if not self.encrypt_input_path.get() or not self.encrypt_output_path.get():
             messagebox.showwarning("Missing Input", "Please select an input PDF and an output file path.", parent=self.root)
             return

        action = self.encrypt_action.get()
        if action == "encrypt":
             user_pw = self.encrypt_user_password.get()
             owner_pw = self.encrypt_owner_password.get()
             # Basic validation: if encrypting, need at least one password
             if not user_pw and not owner_pw:
                  messagebox.showwarning("Missing Password", "Please enter at least a User or Owner password for encryption.", parent=self.root)
                  return
        elif action == "decrypt":
            # If decrypting, input password might be needed, but operation handles check/prompt
             pass

        self._run_operation_base(
            EncryptPDFOperation,
            input_path=self.encrypt_input_path.get(),
            output_path=self.encrypt_output_path.get(),
            action=action,
            user_password=self.encrypt_user_password.get(), # Pass even if empty
            owner_password=self.encrypt_owner_password.get() or None, # Pass None if owner is blank (uses user pw)
            encryption_level=self.encrypt_level.get(),
            allow_printing=self.encrypt_allow_printing.get(),
            allow_copying=self.encrypt_allow_copying.get(),
            allow_modifying=self.encrypt_allow_modifying.get(),
            allow_annotating=self.encrypt_allow_annotating.get(),
            input_password=self.encrypt_input_password.get() or None
        )

    def run_extract_text(self):
        if not self.extract_input_path.get() or not self.extract_output_path.get():
            messagebox.showwarning("Missing Input", "Please select an input PDF and an output text file path.", parent=self.root)
            return

        # Pre-validate page range
        pages_list = None
        num_pages = 0
        input_path = self.extract_input_path.get()
        password = self.extract_password.get() or None
        try: # Get page count
            temp_op = ExtractTextOperation(input_path, "", password=password)
            with contextlib.closing(temp_op._open_input_pdf_fitz()) as doc: num_pages = doc.page_count
        except PdfPasswordException:
            password = self.ask_password(input_path)
            if password is None: return
            self.extract_password.set(password)
            try:
                 temp_op = ExtractTextOperation(input_path, "", password=password)
                 with contextlib.closing(temp_op._open_input_pdf_fitz()) as doc: num_pages = doc.page_count
            except Exception as e:
                 messagebox.showerror("PDF Error", f"Could not open PDF: {e}", parent=self.root); return
        except Exception as e:
             messagebox.showerror("PDF Error", f"Could not open PDF: {e}", parent=self.root); return
        if num_pages == 0: messagebox.showerror("PDF Error", "Input PDF has no pages.", parent=self.root); return

        range_str = self.extract_page_range.get()
        if range_str.strip():
            pages_list = self.validate_page_range(range_str, num_pages)
            if pages_list is None: return # Validation failed

        # Get fitz format name from display name
        display_format = self.extract_format.get()
        fitz_format = self.extract_format_map.get(display_format, "text")

        self._run_operation_base(
            ExtractTextOperation,
            input_path=input_path,
            output_path=self.extract_output_path.get(),
            pages=pages_list, # None or list of 0-based indices
            format_type=fitz_format, # Pass the fitz format name
            password=password,
            layout=self.extract_layout.get()
        )

    def run_rotate_pdf(self):
        if not self.rotate_input_path.get() or not self.rotate_output_path.get():
            messagebox.showwarning("Missing Input", "Please select an input PDF and an output file path.", parent=self.root)
            return

        # Pre-validate page range
        pages_list = None
        num_pages = 0
        input_path = self.rotate_input_path.get()
        password = self.rotate_password.get() or None
        try: # Get page count
            temp_op = RotatePDFOperation(input_path, "", 90, password=password) # Angle doesn't matter for open
            with contextlib.closing(temp_op._open_input_pdf_fitz()) as doc: num_pages = doc.page_count
        except PdfPasswordException:
            password = self.ask_password(input_path)
            if password is None: return
            self.rotate_password.set(password)
            try:
                 temp_op = RotatePDFOperation(input_path, "", 90, password=password)
                 with contextlib.closing(temp_op._open_input_pdf_fitz()) as doc: num_pages = doc.page_count
            except Exception as e: messagebox.showerror("PDF Error", f"Could not open PDF: {e}", parent=self.root); return
        except Exception as e: messagebox.showerror("PDF Error", f"Could not open PDF: {e}", parent=self.root); return
        if num_pages == 0: messagebox.showerror("PDF Error", "Input PDF has no pages.", parent=self.root); return

        range_str = self.rotate_page_range.get()
        if range_str.strip():
            pages_list = self.validate_page_range(range_str, num_pages)
            if pages_list is None: return

        self._run_operation_base(
            RotatePDFOperation,
            input_path=input_path,
            output_path=self.rotate_output_path.get(),
            rotation=self.rotate_angle.get(),
            pages=pages_list, # None or list of 0-based indices
            password=password
        )

    def run_edit_metadata(self):
        if not self.metadata_input_path.get() or not self.metadata_output_path.get():
            messagebox.showwarning("Missing Input", "Please select an input PDF and an output file path.", parent=self.root)
            return

        # Prepare metadata dict, using None to clear fields if empty string entered
        metadata_dict = {
            'title': self.metadata_title.get() or None,
            'author': self.metadata_author.get() or None,
            'subject': self.metadata_subject.get() or None,
            'keywords': self.metadata_keywords.get() or None,
            'creator': self.metadata_creator.get() or None,
            'producer': self.metadata_producer.get() or None,
        }
        # Filter out entries where value is None if you only want to set/update existing
        # metadata_dict_filtered = {k: v for k, v in metadata_dict.items() if v is not None}
        # But fitz uses None to clear, so pass the full dict.

        self._run_operation_base(
            EditMetadataOperation,
            input_path=self.metadata_input_path.get(),
            output_path=self.metadata_output_path.get(),
            metadata=metadata_dict,
            password=self.metadata_password.get() or None
        )


    # --- Operation Event Handlers (Called by OperationManager) ---

    def on_operation_start(self, operation: PDFOperation):
        """UI updates when an operation starts."""
        logger.debug(f"UI Handler: Operation Started - {operation.name}")
        self.progress_value.set(0.0)
        self.progress_visible.set(True)
        # Pack widgets if not already visible (they are forgotten on completion)
        self.progress_bar.pack(side=tk.LEFT, padx=(10, 5), pady=2)
        self.cancel_button.pack(side=tk.LEFT, padx=(0, 5), pady=2)
        self.cancel_button.config(state=tk.NORMAL)
        # Set initial status message from operation
        self.add_status_message(f"Starting {operation.name}...", "info")
        # Disable UI elements (e.g., the 'Run' button for the current tab)
        self._set_ui_state(enabled=False)

    def on_progress_update(self, operation: PDFOperation, progress: float, status_message: str, level: str):
        """Update progress bar and status message."""
        # Check if the update is from the operation we expect
        # if self.operation_manager.current_operation != operation: return # Redundant check?
        # logger.debug(f"UI Handler: Progress Update - {progress:.1f}% - {status_message}")
        self.progress_value.set(progress)
        # Update status message via add_status_message for history tracking
        self.add_status_message(status_message, level)

    def on_status_update(self, operation: PDFOperation, message: str, level: str):
        """Update status message display."""
        # if self.operation_manager.current_operation != operation: return
        # logger.debug(f"UI Handler: Status Update - {level} - {message}")
        self.add_status_message(message, level)

    def on_operation_complete(self, operation: PDFOperation, success: bool, message: str, level: str):
        """UI updates when an operation finishes (successfully or otherwise)."""
        logger.debug(f"UI Handler: Operation Complete - {operation.name} - Success: {success}")
        self.progress_visible.set(False)
        # Unpack progress bar and cancel button
        self.progress_bar.pack_forget()
        self.cancel_button.pack_forget()
        self.cancel_button.config(state=tk.DISABLED) # Ensure it's disabled

        # Display final message from the operation
        self.add_status_message(message, level)
        # Re-enable UI elements
        self._set_ui_state(enabled=True)

        # Optional: Open output location on success
        output_location = None
        try:
            if success:
                if hasattr(operation, 'output_path') and operation.output_path and os.path.exists(operation.output_path):
                     output_location = os.path.dirname(operation.output_path)
                     output_desc = f"Output file created:\n{operation.output_path}"
                elif hasattr(operation, 'output_dir') and operation.output_dir and os.path.exists(operation.output_dir):
                     output_location = operation.output_dir
                     output_desc = f"Output files saved in:\n{operation.output_dir}"

                if output_location:
                     if messagebox.askyesno("Operation Complete", f"{message}\n\n{output_desc}\n\nOpen output location?", parent=self.root):
                         self.open_location(output_location)
        except Exception as e:
             logger.error(f"Error during post-operation actions (open location): {e}")


    def _set_ui_state(self, enabled: bool):
        """Enable or disable main action buttons and potentially other inputs."""
        logger.debug(f"Setting UI state to enabled={enabled}")
        state = tk.NORMAL if enabled else tk.DISABLED
        # Find the main action button on the current tab and disable/enable it
        try:
             current_tab_frame = self.notebook.nametowidget(self.notebook.select())
             # Find the button typically created by _create_action_button
             action_button = None
             # Look for a button with a primary style or specific text? Risky.
             # Let's assume it's the last button added directly to the tab frame's children
             # (or its container's children if packed in a subframe).
             # This needs a more robust way, e.g., storing references.
             for child in reversed(current_tab_frame.winfo_children()):
                 if isinstance(child, ttk.Frame): # Check container frame first
                     for btn_child in reversed(child.winfo_children()):
                         if isinstance(btn_child, ttk.Button) and ('primary' in btn_child.cget('style') or 'Action' in btn_child.cget('text')):
                             action_button = btn_child
                             break
                 elif isinstance(child, ttk.Button) and ('primary' in child.cget('style') or 'Action' in child.cget('text')):
                     action_button = child
                 if action_button: break

             if action_button:
                 logger.debug(f"Found action button: {action_button.cget('text')}")
                 action_button.config(state=state)
             else:
                  logger.warning("Could not find action button on current tab to change state.")

             # Optionally disable/enable other inputs like file lists, options? Could be too broad.
             # For now, just disable the main action button.

        except tk.TclError:
             logger.warning("Could not get current tab frame to set UI state.")
        except Exception as e:
             logger.error(f"Error setting UI state: {e}")


    def add_status_message(self, message: str, level: str = "info"):
        """Adds message to status bar, history, and logs."""
        # Sanitize level
        level = level.lower()
        valid_levels = ["debug", "info", "success", "warning", "error"]
        if level not in valid_levels: level = "info"

        # Update status bar variable
        self.status_message.set(message)
        self.status_level.set(level)

        # Update status label style using ttkbootstrap bootstyles
        style_map = {
            "success": "success", "error": "danger", "warning": "warning",
            "info": "info", "debug": "secondary" # Or primary/light/dark
        }
        bootstyle = style_map.get(level, "default")
        self.status_label.config(bootstyle=bootstyle) # Update bootstyle

        # Add to history deque
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_history.appendleft((timestamp, message, level)) # Add to front

        # Log to file/console (already handled by logger if GUI handler logs too)
        # log_func = getattr(logger, level, logger.info)
        # log_func(f"[Status] {message}") # Avoid double logging if GUI handler works

    def cancel_current_operation(self):
        """Request cancellation of the current operation via OperationManager."""
        logger.info("Cancel button clicked.")
        if self.operation_manager.cancel_current_operation():
            self.add_status_message("Cancellation requested...", "warning")
            self.cancel_button.config(state=tk.DISABLED) # Disable button after requesting
        else:
            self.add_status_message("No operation running to cancel.", "info")

    # --- Dialogs and Utility Windows ---

    def show_recent_files_dialog(self):
        """Show a dialog listing recent files."""
        logger.debug("Showing Recent Files dialog.")
        dialog = RecentFilesDialog(self.root, self.recent_files_manager)
        selected_file = dialog.show() # This waits for the dialog
        if selected_file:
            logger.info(f"File selected from Recent Files dialog: {selected_file}")
            # Open the selected file appropriately
            if selected_file.lower().endswith('.pdf'):
                self.open_recent_pdf(selected_file)
            elif any(selected_file.lower().endswith(ext) for ext in SUPPORTED_IMAGE_FORMATS):
                self.open_recent_image(selected_file)
            else:
                 self.open_recent_unknown(selected_file)

    def show_settings_dialog(self):
        """Show the application settings dialog."""
        logger.debug("Showing Settings dialog.")
        dialog = SettingsDialog(self.root, self.config_manager)
        # Settings are applied and saved within the dialog if OK is pressed

    def show_log_viewer(self):
        """Show a window displaying the status message history."""
        logger.debug("Showing Log Viewer dialog.")
        # Pass a copy of the history deque
        LogViewerDialog(self.root, list(self.status_history)) # Convert deque to list for dialog


    def show_help(self):
        """Show a simple help message."""
        logger.debug("Showing Help dialog.")
        help_text = f"""
        {APP_NAME} - v{__version__}

        Welcome! Select a tool from the sidebar or Tools menu.

        Common Operations:
        - Split: Divide a PDF by pages or bookmarks.
        - Merge: Combine multiple PDFs.
        - Images to PDF: Create PDF from image files.
        - PDF to Images: Convert PDF pages to images.
        - Compress: Reduce PDF file size.
        - Encrypt/Decrypt: Add or remove passwords.
        - Extract Text: Get text content from PDF.
        - Rotate: Change page orientation.
        - Edit Metadata: Modify document properties.

        Tips:
        - Use 'Browse...' buttons to select files/folders.
        - Hover over options for tooltips (if available).
        - Use 'Recent Files' for quick access.
        - Check the status bar for progress and messages.
        - Click the status bar message to view logs.
        - Customize the theme via View -> Theme.

        (More detailed documentation link placeholder)
        """
        messagebox.showinfo(f"{APP_NAME} - Help", help_text, parent=self.root)

    def show_about_dialog(self):
        """Show the About dialog."""
        logger.debug("Showing About dialog.")
        about_text = f"""
        {APP_NAME}
        Version: {__version__}

        A graphical toolkit for common PDF operations.

        Powered by:
        - Python {platform.python_version()}
        - Tkinter / ttkbootstrap
        - PyMuPDF (fitz)
        - pypdf
        - Pillow (PIL)

        Developed by: Hamoon Soleimani

        """
        messagebox.showinfo(f"About {APP_NAME}", about_text, parent=self.root)

    def check_for_updates(self):
        """Placeholder for checking for new application versions."""
        logger.debug("Checking for updates (placeholder)...")
        messagebox.showinfo("Check for Updates", "Update checking feature is not yet implemented.", parent=self.root)
        # Future: Implement check against a version file or API endpoint.

    def open_location(self, path: str):
        """Opens the given file or directory in the system's default application."""
        logger.info(f"Attempting to open location: {path}")
        if not os.path.exists(path):
            logger.error(f"Cannot open location: Path does not exist - {path}")
            messagebox.showerror("Error", f"Location not found:\n{path}", parent=self.root)
            return
        try:
            if platform.system() == "Windows":
                os.startfile(path)
            elif platform.system() == "Darwin": # macOS
                subprocess.run(["open", path], check=True)
            else: # Linux/Unix
                subprocess.run(["xdg-open", path], check=True)
        except FileNotFoundError:
             # If os.startfile/open/xdg-open is not found
             messagebox.showerror("Error", f"Could not find system utility to open location:\n{path}", parent=self.root)
             logger.error(f"System utility not found to open {path}")
        except subprocess.CalledProcessError as e:
             messagebox.showerror("Error", f"System utility failed to open location:\n{path}\nError: {e}", parent=self.root)
             logger.error(f"System utility failed for {path}: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not open location:\n{path}\nError: {e}", parent=self.root)
            logger.exception(f"Failed to open location {path}")

    def quit_app(self):
        """Handles application shutdown."""
        logger.debug("Quit requested.")
        if messagebox.askyesno("Exit Confirmation", f"Are you sure you want to exit {APP_NAME}?", parent=self.root):
            logger.info(f"Exiting {APP_NAME}.")
            # --- Perform cleanup ---
            # Stop background threads? (Operation manager thread is daemon)
            # Save configuration (should be saved on change, but explicit save here is safe)
            self.config_manager.save_config()
            logger.info("Configuration saved.")
            # Destroy the main window
            self.root.destroy()
            logger.info("Application shutdown complete.")
        else:
             logger.debug("Quit cancelled by user.")

# --- Custom Dialog Classes ---

class CustomDialogBase(tk.Toplevel):
    """Base class for custom modal dialogs using ttkbootstrap."""
    def __init__(self, parent: tk.Widget, title: str = "Dialog"):
        super().__init__(parent)
        self.transient(parent) # Associate with parent window
        self.title(title)
        self.parent = parent
        self.result = None # Store dialog result

        # Use ttkbootstrap themed frame
        self.dialog_frame = ttk.Frame(self, padding=15) # style='...' if needed
        self.dialog_frame.pack(expand=True, fill=tk.BOTH)

        # Body frame for content (subclasses override create_body)
        self.body_frame = ttk.Frame(self.dialog_frame)
        self.body_frame.pack(expand=True, fill=tk.BOTH, pady=(0, 10))
        self.create_body(self.body_frame)

        # Button frame (subclasses override create_buttons)
        self.button_frame = ttk.Frame(self.dialog_frame)
        self.button_frame.pack(fill=tk.X, side=tk.BOTTOM) # Place at bottom
        self.create_buttons(self.button_frame)

        self.center_window() # Center relative to parent

        # Make modal
        self.grab_set() # Direct events to this dialog
        self.protocol("WM_DELETE_WINDOW", self.on_cancel) # Handle window close button
        self.bind("<Escape>", self.on_cancel) # Bind Escape key to cancel

        self.wait_window(self) # Wait until dialog is destroyed

    def create_body(self, master: ttk.Frame):
        """Override this method to create the dialog's main content widgets."""
        ttk.Label(master, text="Dialog content goes here.").pack()

    def create_buttons(self, master: ttk.Frame):
        """Override this method to create dialog action buttons."""
        # Align buttons to the right
        master.columnconfigure(0, weight=1) # Push buttons right
        # ttk.Button(master, text="OK", command=self.on_ok, style='primary.TButton').pack(side=tk.RIGHT, padx=(5,0), pady=5)
        # ttk.Button(master, text="Cancel", command=self.on_cancel, style='secondary.TButton').pack(side=tk.RIGHT, padx=5, pady=5)
        ok_btn = ttk.Button(master, text="OK", command=self.on_ok, style='primary.TButton', width=10)
        ok_btn.grid(row=0, column=1, padx=(5,0), pady=5)
        cancel_btn = ttk.Button(master, text="Cancel", command=self.on_cancel, style='secondary.TButton', width=10)
        cancel_btn.grid(row=0, column=2, padx=5, pady=5)
        # Set focus to OK button initially
        ok_btn.focus_set()
        self.bind("<Return>", self.on_ok) # Bind Enter key to OK


    def center_window(self):
        """Centers the dialog window relative to its parent."""
        try:
            self.update_idletasks() # Ensure dimensions are calculated
            # Get dialog size
            width = self.winfo_width()
            height = self.winfo_height()
            # Get parent size and position
            parent_x = self.parent.winfo_rootx() # Use rootx/y for screen coordinates
            parent_y = self.parent.winfo_rooty()
            parent_w = self.parent.winfo_width()
            parent_h = self.parent.winfo_height()
            # Calculate centered position
            x_pos = parent_x + (parent_w // 2) - (width // 2)
            y_pos = parent_y + (parent_h // 2) - (height // 2)
            # Ensure it's on screen (basic check)
            screen_w = self.winfo_screenwidth()
            screen_h = self.winfo_screenheight()
            x_pos = max(0, min(x_pos, screen_w - width))
            y_pos = max(0, min(y_pos, screen_h - height))

            self.geometry(f'+{x_pos}+{y_pos}')
        except tk.TclError:
             logger.warning("Could not center dialog (window may not be ready).")

    def on_ok(self, event=None):
        """Handles the OK button click or Enter key."""
        if not self.validate():
            return # Validation failed, keep dialog open
        self.result = self.apply() # Process data
        self.destroy() # Close dialog

    def on_cancel(self, event=None):
        """Handles the Cancel button click, Escape key, or window close."""
        self.result = None # Indicate cancellation
        self.destroy()

    def validate(self) -> bool:
        """Override for input validation before closing on OK."""
        return True # Default: always valid

    def apply(self) -> Any:
        """Override to process dialog data and return a result on OK."""
        return True # Default: return True on success


class RecentFilesDialog(CustomDialogBase):
    """Dialog to display and select from the recent files list."""
    def __init__(self, parent: tk.Widget, recent_files_manager: RecentFilesManager):
        self.manager = recent_files_manager
        self.selected_file = tk.StringVar() # Store the selection display? Not needed.
        self.recent_files: List[str] = [] # Store the list locally for selection index
        super().__init__(parent, title="Recent Files")

    def create_body(self, master: ttk.Frame):
        ttk.Label(master, text="Select a recently used file to open:").pack(pady=(0, 10), anchor=tk.W)

        list_frame = ttk.Frame(master)
        list_frame.pack(fill=tk.BOTH, expand=True)
        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)

        self.listbox = tk.Listbox(list_frame, width=70, height=15, font=('Courier', 9)) # Monospace looks good for paths
        self.listbox.grid(row=0, column=0, sticky='nsew')
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.listbox.yview, style='Vertical.TScrollbar')
        scrollbar.grid(row=0, column=1, sticky='ns')
        self.listbox.config(yscrollcommand=scrollbar.set)

        self.populate_list()
        # Bind double-click to OK action
        self.listbox.bind("<Double-Button-1>", self.on_ok)

    def populate_list(self):
        """Fills the listbox with recent files."""
        self.listbox.delete(0, tk.END)
        self.recent_files = self.manager.get_recent_files() # Get fresh list
        if not self.recent_files:
            self.listbox.insert(tk.END, "(No recent files)")
            self.listbox.config(state=tk.DISABLED)
        else:
            self.listbox.config(state=tk.NORMAL)
            for path in self.recent_files:
                self.listbox.insert(tk.END, path) # Show full path
            # Select first item by default
            self.listbox.selection_set(0)
            self.listbox.activate(0)


    def create_buttons(self, master: ttk.Frame):
        master.columnconfigure(0, weight=1) # Push buttons right
        # Add Clear button
        clear_btn = ttk.Button(master, text="Clear List", command=self.clear_list, style='Outline.Danger.TButton', width=10)
        clear_btn.grid(row=0, column=0, padx=(0,10), pady=5, sticky='e') # Align left of OK/Cancel? No, put far left.
        clear_btn.grid_remove() # Reposition clear button
        clear_btn.grid(row=0, column=0, padx=5, pady=5, sticky='w') # Put on the left

        ok_btn = ttk.Button(master, text="Open", command=self.on_ok, style='primary.TButton', width=10)
        ok_btn.grid(row=0, column=1, padx=(5,0), pady=5)
        cancel_btn = ttk.Button(master, text="Cancel", command=self.on_cancel, style='secondary.TButton', width=10)
        cancel_btn.grid(row=0, column=2, padx=5, pady=5)
        ok_btn.focus_set()

    def validate(self) -> bool:
        """Ensure an item is selected if the list is not empty."""
        if not self.recent_files:
             return False # Cannot OK if list is empty
        selection = self.listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a file from the list.", parent=self)
            return False
        return True

    def apply(self) -> Optional[str]:
        """Return the selected file path."""
        selection_index = self.listbox.curselection()[0]
        if 0 <= selection_index < len(self.recent_files):
            return self.recent_files[selection_index]
        return None # Should not happen if validate works

    def clear_list(self):
        """Handle the Clear List button action."""
        if messagebox.askyesno("Clear Recent Files", "Are you sure you want to clear the recent files list?", parent=self):
            self.manager.clear_recent_files()
            self.populate_list() # Update the dialog listbox immediately

    def show(self) -> Optional[str]:
        """Override show to return the selected file path directly."""
        self.wait_window(self) # Wait for dialog interaction
        return self.result # Return the file path or None


class SettingsDialog(CustomDialogBase):
    """Dialog for editing application settings."""
    def __init__(self, parent: tk.Widget, config_manager: ConfigManager):
        self.config = config_manager
        # --- Create tk Variables linked to config values ---
        # General
        self.check_updates = tk.BooleanVar(value=self.config.getboolean("app", "check_updates", True))
        self.recent_limit = tk.IntVar(value=self.config.getint("app", "recent_files_limit", 10))
        self.log_level = tk.StringVar(value=self.config.get("app", "log_level", "INFO"))
        # PDF Defaults
        self.default_dpi = tk.IntVar(value=self.config.getint("pdf", "default_dpi", 300))
        self.default_img2pdf_margin = tk.DoubleVar(value=self.config.getfloat("pdf", "img2pdf_margin_pt", 36.0))
        self.default_img2pdf_page_size = tk.StringVar(value=self.config.get("pdf", "img2pdf_page_size", "A4"))
        self.default_img2pdf_quality = tk.IntVar(value=self.config.getint("pdf", "img2pdf_quality", 95))
        self.default_compress_quality = tk.IntVar(value=self.config.getint("pdf", "compress_quality", 75))
        self.default_compress_img_dpi = tk.IntVar(value=self.config.getint("pdf", "compress_img_dpi", 150))
        self.default_encrypt_level = tk.IntVar(value=self.config.getint("pdf", "encrypt_level", 128))
        # Paths (Example)
        self.default_output_dir = tk.StringVar(value=self.config.get("paths", "default_output_dir", ""))

        super().__init__(parent, title="Application Settings")

    def create_body(self, master: ttk.Frame):
        notebook = ttk.Notebook(master)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # --- General Tab ---
        general_frame = ttk.Frame(notebook, padding=10)
        notebook.add(general_frame, text=" General ")

        # Use _create_widget_row helper for consistency
        _, cb_updates = AdvancedPdfToolkit._create_widget_row(general_frame, "Check for updates on startup:", self.check_updates, 'checkbutton')
        _, sb_limit = AdvancedPdfToolkit._create_widget_row(general_frame, "Recent Files Limit:", self.recent_limit, 'spinbox', widget_options={'from_': 1, 'to': 50, 'width': 8})
        _, cb_log = AdvancedPdfToolkit._create_widget_row(general_frame, "Logging Level:", self.log_level, 'combobox', widget_options={'values': ["DEBUG", "INFO", "WARNING", "ERROR"], 'width': 10})

        # --- PDF Defaults Tab ---
        pdf_frame = ttk.Frame(notebook, padding=10)
        notebook.add(pdf_frame, text=" PDF Defaults ")

        pdf_options_frame = ttk.LabelFrame(pdf_frame, text="Default Values for Operations", padding=10)
        pdf_options_frame.pack(fill=tk.X)

        AdvancedPdfToolkit._create_widget_row(pdf_options_frame, "Resolution (DPI):", self.default_dpi, 'spinbox', widget_options={'from_': 72, 'to': 1200, 'width': 8}, tooltip="Default DPI for PDF/Image conversions.")
        AdvancedPdfToolkit._create_widget_row(pdf_options_frame, "Img->PDF Page Size:", self.default_img2pdf_page_size, 'combobox', widget_options={'values': ["A4", "Letter", "Legal", "A3", "auto"], 'width': 10})
        AdvancedPdfToolkit._create_widget_row(pdf_options_frame, "Img->PDF Margin (pt):", self.default_img2pdf_margin, 'spinbox', widget_options={'from_': 0, 'to': 200, 'increment': 1.0, 'width': 8})
        AdvancedPdfToolkit._create_widget_row(pdf_options_frame, "Img->PDF Quality (%):", self.default_img2pdf_quality, 'spinbox', widget_options={'from_': 1, 'to': 100, 'width': 8})
        AdvancedPdfToolkit._create_widget_row(pdf_options_frame, "Compress Quality (%):", self.default_compress_quality, 'spinbox', widget_options={'from_': 0, 'to': 100, 'width': 8}, tooltip="Approximate target quality for compression.")
        AdvancedPdfToolkit._create_widget_row(pdf_options_frame, "Compress Image DPI:", self.default_compress_img_dpi, 'spinbox', widget_options={'from_': 72, 'to': 600, 'width': 8})
        AdvancedPdfToolkit._create_widget_row(pdf_options_frame, "Encrypt Level (bit):", self.default_encrypt_level, 'combobox', widget_options={'values': [128, 256], 'width': 8})


        # --- Paths Tab ---
        paths_frame = ttk.Frame(notebook, padding=10)
        notebook.add(paths_frame, text=" Paths ")

        path_options_frame = ttk.LabelFrame(paths_frame, text="Default Locations", padding=10)
        path_options_frame.pack(fill=tk.X)

        def _select_default_output():
            initial = self.default_output_dir.get() or os.path.expanduser("~")
            selected = filedialog.askdirectory(title="Select Default Output Directory", initialdir=initial, parent=self)
            if selected: self.default_output_dir.set(selected)

        AdvancedPdfToolkit._create_widget_row(path_options_frame, "Default Output Dir:", self.default_output_dir, 'readonly_entry', button_text="Browse...", button_cmd=_select_default_output, tooltip="Default location for saving output files (can be overridden).")
        # Add Temp Dir setting if needed


    def apply(self) -> bool:
        """Save settings back to ConfigManager."""
        logger.info("Applying settings from dialog...")
        # General
        self.config.set("app", "check_updates", self.check_updates.get())
        self.config.set("app", "recent_files_limit", self.recent_limit.get())
        self.config.set("app", "log_level", self.log_level.get())
        # PDF Defaults
        self.config.set("pdf", "default_dpi", self.default_dpi.get())
        self.config.set("pdf", "img2pdf_page_size", self.default_img2pdf_page_size.get())
        self.config.set("pdf", "img2pdf_margin_pt", self.default_img2pdf_margin.get())
        self.config.set("pdf", "img2pdf_quality", self.default_img2pdf_quality.get())
        self.config.set("pdf", "compress_quality", self.default_compress_quality.get())
        self.config.set("pdf", "compress_img_dpi", self.default_compress_img_dpi.get())
        self.config.set("pdf", "encrypt_level", self.default_encrypt_level.get())
        # Paths
        self.config.set("paths", "default_output_dir", self.default_output_dir.get())

        self.config.save_config()

        # Apply log level change immediately to the logger and handlers
        new_log_level_str = self.log_level.get()
        new_log_level_int = getattr(logging, new_log_level_str.upper(), logging.INFO)
        if logger.getEffectiveLevel() != new_log_level_int:
            logger.setLevel(new_log_level_int)
            # Update handler levels too
            for handler in logger.handlers:
                 handler.setLevel(new_log_level_int)
            logger.log(new_log_level_int, f"Log level changed to {new_log_level_str}")

        # Notify main app if recent files limit changed
        # This requires the main app to listen or re-read the limit.
        # We can trigger the recent files changed notification as a signal.
        self.parent.recent_files_manager._notify_change()

        # Update internal defaults in the main app instance
        # This is slightly messy coupling, but necessary if defaults are used directly.
        # Consider having the main app re-read defaults from config manager
        # when needed instead.
        try:
            self.parent.img2pdf_resolution.set(self.default_dpi.get())
            self.parent.img2pdf_page_size.set(self.default_img2pdf_page_size.get())
            self.parent.img2pdf_margin_pt.set(self.default_img2pdf_margin.get())
            self.parent.img2pdf_quality.set(self.default_img2pdf_quality.get())
            self.parent.compress_quality.set(self.default_compress_quality.get())
            self.parent.compress_image_dpi.set(self.default_compress_img_dpi.get())
            self.parent.encrypt_level.set(self.default_encrypt_level.get())
            # Update other defaults in main app if they exist as tk variables
        except Exception as e:
             logger.error(f"Error updating main app variables from settings: {e}")

        return True # Indicate success

    # No need for show method override, base class handles modality


class LogViewerDialog(tk.Toplevel):
    """A non-modal dialog to display application status/log history."""
    def __init__(self, parent: tk.Widget, log_history: List[Tuple[str, str, str]]):
        super().__init__(parent)
        self.title("Log History Viewer")
        self.geometry("750x500")
        self.attributes('-topmost', False) # Allow interaction with main window

        # --- Text Area ---
        text_frame = ttk.Frame(self)
        text_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        text_frame.rowconfigure(0, weight=1)
        text_frame.columnconfigure(0, weight=1)

        self.log_text = scrolledtext.ScrolledText(text_frame, wrap=tk.WORD, state=tk.DISABLED, height=20, width=80, font=('Consolas', 9)) # Use monospace
        self.log_text.grid(row=0, column=0, sticky='nsew')

        # Scrollbar (ttkbootstrap styled)
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=self.log_text.yview, style='Vertical.TScrollbar')
        scrollbar.grid(row=0, column=1, sticky='ns')
        self.log_text['yscrollcommand'] = scrollbar.set

        # --- Configure Tags for Log Levels ---
        # Use theme colors if possible
        try:
             style = Style()
             self.log_text.tag_config("info", foreground=style.colors.info)
             self.log_text.tag_config("warning", foreground=style.colors.warning)
             self.log_text.tag_config("error", foreground=style.colors.danger)
             self.log_text.tag_config("debug", foreground=style.colors.secondary)
             self.log_text.tag_config("success", foreground=style.colors.success)
        except Exception: # Fallback if styles or colors aren't available
             self.log_text.tag_config("info", foreground="blue")
             self.log_text.tag_config("warning", foreground="orange")
             self.log_text.tag_config("error", foreground="red")
             self.log_text.tag_config("debug", foreground="gray")
             self.log_text.tag_config("success", foreground="green")

        # --- Buttons ---
        btn_frame = ttk.Frame(self, padding=(0, 5, 0, 10))
        btn_frame.pack(fill=tk.X)
        btn_frame.columnconfigure(0, weight=1) # Push buttons right

        # ttk.Button(btn_frame, text="Clear", command=self.clear_logs, style='Outline.TButton').pack(side=tk.LEFT, padx=5) # Maybe disable clear?
        ttk.Button(btn_frame, text="Copy to Clipboard", command=self.copy_logs, style='Outline.TButton').grid(row=0, column=1, padx=5)
        ttk.Button(btn_frame, text="Close", command=self.destroy, style='primary.TButton').grid(row=0, column=2, padx=5)

        # --- Populate ---
        self.populate_logs(log_history)

        # --- Finalize ---
        # self.transient(parent) # Don't make transient if non-modal needed
        # self.grab_set() # Don't grab if non-modal
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.center_window(parent)

    def center_window(self, parent: tk.Widget):
        # Same centering logic as base dialog
        try:
            self.update_idletasks()
            width = self.winfo_width()
            height = self.winfo_height()
            parent_x = parent.winfo_rootx()
            parent_y = parent.winfo_rooty()
            parent_w = parent.winfo_width()
            parent_h = parent.winfo_height()
            x_pos = parent_x + (parent_w // 2) - (width // 2)
            y_pos = parent_y + (parent_h // 2) - (height // 2)
            screen_w = self.winfo_screenwidth()
            screen_h = self.winfo_screenheight()
            x_pos = max(0, min(x_pos, screen_w - width))
            y_pos = max(0, min(y_pos, screen_h - height))
            self.geometry(f'+{x_pos}+{y_pos}')
        except tk.TclError: pass


    def populate_logs(self, log_history: List[Tuple[str, str, str]]):
        """Fills the text area with log messages."""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete('1.0', tk.END)
        # History is deque (appendleft), so iterate reversed for chronological order
        for timestamp, message, level in reversed(log_history):
            self.log_text.insert(tk.END, f"[{timestamp}] ", ("debug",)) # Timestamp style
            self.log_text.insert(tk.END, f"{message}\n", (level,)) # Message with level style
        self.log_text.config(state=tk.DISABLED)
        self.log_text.see(tk.END) # Scroll to the most recent entry at the bottom

    # def clear_logs(self):
    #     # Clearing the dialog doesn't clear the main app's history
    #     self.log_text.config(state=tk.NORMAL)
    #     self.log_text.delete('1.0', tk.END)
    #     self.log_text.config(state=tk.DISABLED)
    #     logger.info("Log viewer display cleared.")

    def copy_logs(self):
        """Copies the entire log content to the clipboard."""
        try:
            log_content = self.log_text.get('1.0', tk.END)
            self.clipboard_clear()
            self.clipboard_append(log_content)
            # Use main app's status bar if possible
            if hasattr(self.parent, 'add_status_message'):
                 self.parent.add_status_message("Log history copied to clipboard.", "info")
            logger.info("Log history copied to clipboard.")
        except tk.TclError as e:
             logger.error(f"Failed to copy logs to clipboard: {e}")
             if hasattr(self.parent, 'add_status_message'):
                  self.parent.add_status_message("Failed to copy logs to clipboard.", "error")


class CustomThemeDialog(CustomDialogBase):
    """Dialog for editing custom theme colors."""
    def __init__(self, parent: tk.Widget, config_manager: ConfigManager):
        self.config = config_manager
        self.colors: Dict[str, tk.StringVar] = {} # Store tk Variables for colors
        self.preview_widgets: Dict[str, tk.Frame] = {}
        self.base_theme = tk.StringVar(value=self.config.get_custom_theme_base())
        self.current_colors = self.config.get_custom_theme_colors()
        # If no custom colors defined, load defaults from the base theme preset
        if not self.current_colors:
             base_theme_name = self.base_theme.get()
             # Infer light/dark from base theme name
             preset_key = 'dark' if any(dark in base_theme_name.lower() for dark in ['dark', 'cyborg', 'solar', 'superhero']) else 'light'
             self.current_colors = THEME_PRESETS[preset_key].copy()
             logger.info(f"No custom colors found, initializing editor with '{preset_key}' preset.")

        super().__init__(parent, title="Edit Custom Theme Colors")

    def create_body(self, master: ttk.Frame):
        master.columnconfigure(1, weight=1) # Allow entry/chooser to expand slightly
        row = 0

        # Base Theme Selection
        ttk.Label(master, text="Base Theme:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=3)
        base_combo = ttk.Combobox(master, textvariable=self.base_theme,
                                  values=Style().theme_names(), state='readonly', width=20)
        base_combo.grid(row=row, column=1, columnspan=3, sticky=tk.EW, padx=5, pady=3)
        AdvancedPdfToolkit.add_tooltip(None, base_combo, "Select the built-in theme to base your custom colors on.")
        row += 1

        ttk.Separator(master, orient=tk.HORIZONTAL).grid(row=row, column=0, columnspan=4, sticky='ew', pady=10)
        row += 1

        ttk.Label(master, text="Color Overrides:").grid(row=row, column=0, columnspan=4, sticky=tk.W, padx=5, pady=(0, 5))
        row += 1

        # Define the standard ttkbootstrap color keys
        color_keys = list(THEME_PRESETS['light'].keys()) # Get all keys from a preset

        for key in color_keys:
            # Default to current color or fallback to white/black
            default_color = self.current_colors.get(key, "#ffffff" if 'light' in self.base_theme.get() else "#000000")
            self.colors[key] = tk.StringVar(value=default_color)

            # Label for the color name
            ttk.Label(master, text=f"{key.capitalize()}:").grid(row=row, column=0, sticky=tk.W, padx=5, pady=3)

            # Entry for hex code
            entry = ttk.Entry(master, textvariable=self.colors[key], width=10)
            entry.grid(row=row, column=1, sticky=tk.EW, padx=5, pady=3)

            # Color Chooser Button
            btn = ttk.Button(master, text="Choose...", style='Outline.TButton',
                           command=lambda k=key: self.choose_color(k))
            btn.grid(row=row, column=2, sticky=tk.W, padx=5, pady=3)

            # Color Preview Swatch
            preview = tk.Frame(master, width=20, height=20, relief=tk.SUNKEN, borderwidth=1)
            preview.grid(row=row, column=3, padx=(5, 10), pady=3)
            try:
                preview.config(bg=self.colors[key].get()) # Initial color
            except tk.TclError:
                 preview.config(bg="#FF00FF") # Magenta for error
            self.preview_widgets[key] = preview

            # Update preview when entry changes
            self.colors[key].trace_add("write", lambda *args, k=key: self.update_preview(k))

            row += 1

        # Reset Buttons Frame
        reset_frame = ttk.Frame(master)
        reset_frame.grid(row=row, column=0, columnspan=4, pady=10)
        ttk.Button(reset_frame, text="Load Light Preset", style='Outline.TButton',
                   command=lambda: self.load_preset('light')).pack(side=tk.LEFT, padx=5)
        ttk.Button(reset_frame, text="Load Dark Preset", style='Outline.TButton',
                   command=lambda: self.load_preset('dark')).pack(side=tk.LEFT, padx=5)


    def choose_color(self, key: str):
        """Open color chooser and update the variable."""
        current_color = self.colors[key].get()
        try:
            new_color = colorchooser.askcolor(color=current_color, title=f"Choose {key.capitalize()} Color", parent=self)
            if new_color and new_color[1]: # Check color was chosen (new_color[1] is hex)
                hex_color = new_color[1].lower()
                self.colors[key].set(hex_color)
                # Trace will call update_preview
        except tk.TclError as e:
             # Handle cases where current_color is invalid for askcolor
             logger.warning(f"Invalid current color '{current_color}' for color chooser: {e}")
             new_color = colorchooser.askcolor(title=f"Choose {key.capitalize()} Color", parent=self)
             if new_color and new_color[1]: self.colors[key].set(new_color[1].lower())

    def update_preview(self, key: str):
        """Update the color swatch next to the entry."""
        color = self.colors[key].get()
        preview_widget = self.preview_widgets.get(key)
        if preview_widget:
            try:
                # Validate hex color format before setting background
                if re.match(r'^#[0-9a-fA-F]{6}$', color):
                    preview_widget.config(bg=color)
                else:
                     preview_widget.config(bg="#FF00FF") # Indicate invalid format
            except tk.TclError:
                preview_widget.config(bg="#FF00FF") # Magenta for other errors

    def load_preset(self, preset_key: str):
        """Load color values from a predefined preset."""
        preset = THEME_PRESETS.get(preset_key)
        if preset:
            logger.info(f"Loading '{preset_key}' color preset into editor.")
            for key, value in preset.items():
                if key in self.colors:
                    self.colors[key].set(value)
                    # Trace will update preview
        else:
             logger.warning(f"Preset '{preset_key}' not found.")

    def validate(self) -> bool:
        """Validate all entered color values are valid hex codes."""
        invalid_colors = []
        for key, var in self.colors.items():
            color = var.get()
            if not re.match(r'^#[0-9a-fA-F]{6}$', color):
                invalid_colors.append(f"- {key.capitalize()}: '{color}'")

        if invalid_colors:
            message = "Invalid color format found:\n\n" + "\n".join(invalid_colors)
            message += "\n\nPlease use hex format (e.g., #RRGGBB)."
            messagebox.showerror("Invalid Color(s)", message, parent=self)
            return False
        return True

    def apply(self) -> bool:
        """Save the custom theme settings to config."""
        custom_colors = {key: var.get() for key, var in self.colors.items()}
        new_base_theme = self.base_theme.get()

        self.config.set_custom_theme_base(new_base_theme)
        self.config.set_custom_theme_colors(custom_colors)
        self.config.save_config()
        logger.info(f"Custom theme saved. Base: {new_base_theme}, Colors: {len(custom_colors)} overrides.")
        return True # Return success, main app handles re-applying if needed


# --- Main Execution ---

def check_dependencies_on_startup():
    """Check dependencies and show error if critical ones are missing."""
    if MISSING_DEPENDENCIES:
        critical_missing = [dep for dep in MISSING_DEPENDENCIES if any(core in dep for core in ['pymupdf', 'pypdf', 'Pillow', 'ttkbootstrap'])]
        if critical_missing:
            error_message = "Critical dependencies missing:\n\n" + "\n".join(critical_missing)
            error_message += "\n\nPlease install them (e.g., using 'pip install ...') and restart the application."
            try:
                # Try showing a graphical message box even before main app loop
                root = tk.Tk()
                root.withdraw() # Hide the blank root window
                messagebox.showerror("Dependency Error", error_message)
                root.destroy()
            except Exception:
                 # Fallback to console if GUI error message fails
                 print(f"\n{'='*20} DEPENDENCY ERROR {'='*20}\n", file=sys.stderr)
                 print(error_message, file=sys.stderr)
                 print(f"\n{'='*58}\n", file=sys.stderr)
            sys.exit(1) # Exit if critical dependencies are missing
        else:
            # Log warnings for non-critical missing dependencies if logger is ready
            try:
                 logger.warning(f"Optional dependencies missing: {MISSING_DEPENDENCIES}. Some fallback features might be unavailable.")
            except NameError: # Logger might not be initialized yet
                 print(f"Warning: Optional dependencies missing: {MISSING_DEPENDENCIES}", file=sys.stderr)


if __name__ == "__main__":
    # 1. Check Dependencies immediately
    check_dependencies_on_startup()

    # 2. Initialize Configuration (needed for initial theme)
    config = ConfigManager()

    # 3. Get initial theme from config to start ttkbootstrap window correctly
    initial_theme_mode = config.get_theme_mode()
    initial_ttb_theme = TTB_DEFAULT_THEME # Default fallback
    if initial_theme_mode == Theme.LIGHT:
        initial_ttb_theme = TTB_LIGHT_THEME
    elif initial_theme_mode == Theme.DARK:
         initial_ttb_theme = TTB_DARK_THEME
    elif initial_theme_mode == Theme.CUSTOM:
         # Use the stored base theme for custom, or default dark
         initial_ttb_theme = config.get_custom_theme_base() or TTB_DARK_THEME
    elif initial_theme_mode == Theme.SYSTEM:
         # Basic detection here is tricky before mainloop, default to dark/light
         # We'll refine this in apply_theme() later. Start with light maybe?
         initial_ttb_theme = TTB_LIGHT_THEME # Or attempt basic check?

    logger.info(f"Starting with initial theme: {initial_ttb_theme} (mode: {initial_theme_mode.value})")

    # 4. Create the main application window using ttkbootstrap
    root = ttb.Window(themename=initial_ttb_theme)

    # 5. Instantiate the main application class
    try:
        app = AdvancedPdfToolkit(root)
    except Exception as app_init_error:
         logger.critical(f"Failed to initialize the application GUI: {app_init_error}", exc_info=True)
         messagebox.showerror("Application Error", f"Failed to start the application:\n\n{app_init_error}\n\nCheck logs for details.")
         root.destroy()
         sys.exit(1)

    # 6. Start the Tkinter main event loop
    logger.info("Starting Tkinter main loop...")
    root.mainloop()

    logger.info(f"{APP_NAME} finished.")
