#!/usr/bin/env python3

"""
A full-featured GUI application for splitting and merging PDF files.

Requires the 'pypdf' library: pip install pypdf
Uses Python's built-in tkinter for the GUI.
"""

import tkinter as tk
from tkinter import ttk  # Themed Tkinter widgets
from tkinter import filedialog, messagebox, scrolledtext
import os
import sys
import threading
import queue  # For thread communication
import re
from typing import List, Tuple, Optional

# --- Core PDF Processing Logic (Adapted from CLI version) ---
# We keep this logic separate from the GUI code as much as possible.

try:
    from pypdf import PdfReader, PdfWriter, PageRange
    from pypdf.errors import PdfReadError, DependencyError
except ImportError:
    print("Error: pypdf library not found.", file=sys.stderr)
    print("Please install it: pip install pypdf", file=sys.stderr)
    # Attempt to show a GUI message box if possible, otherwise exit
    try:
        root = tk.Tk()
        root.withdraw() # Hide the main window
        messagebox.showerror("Dependency Error", "pypdf library not found.\nPlease install it using:\n\npip install pypdf")
        root.destroy()
    except tk.TclError:
        pass # If tkinter itself fails very early
    sys.exit(1)


def parse_page_ranges(ranges_str: str, num_pages: int) -> List[Tuple[int, int]]:
    """
    Parses a comma-separated string of page ranges (1-based index) into a list of
    tuples representing (start_page_index, end_page_index) (0-based, inclusive).
    (Identical to the CLI version's function)
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
                # Allow end page > num_pages, but adjust it down (handled later or in pypdf)
                # We'll rely on pypdf or the loop later to handle the actual max page index
                # Let's validate the adjusted end_page here though
                effective_end_page = min(end_page, num_pages)
                if start_page > effective_end_page and num_pages > 0 :
                     # This case happens if start > num_pages, caught above,
                     # or if start <= num_pages < end_page, making the effective range invalid.
                     # Let the range (start_page - 1, effective_end_page - 1) be checked later.
                     pass


                # Convert to 0-based inclusive indices
                parsed_ranges.append((start_page - 1, min(end_page, num_pages) - 1)) # Use min for upper bound

            except ValueError as e:
                if "invalid literal for int()" in str(e):
                    raise ValueError(f"Invalid number in page range '{part}'.") from e
                raise
        else:
            try:
                page = int(part)
                if page < 1 or page > num_pages:
                    raise ValueError(f"Page number {page} is out of valid range (1-{num_pages}).")
                parsed_ranges.append((page - 1, page - 1))
            except ValueError as e:
                if "invalid literal for int()" in str(e):
                     raise ValueError(f"Invalid page number '{part}'. Must be an integer or a range (e.g., '1-5').") from e
                raise

    # Sort and merge overlapping/adjacent ranges
    if not parsed_ranges: return []
    parsed_ranges.sort(key=lambda x: x[0])
    merged = []
    if parsed_ranges: # Ensure list is not empty
        current_start, current_end = parsed_ranges[0]
        # Filter out invalid ranges where start index > end index after adjustments
        if current_start > current_end:
            parsed_ranges = parsed_ranges[1:] # Skip invalid first range
        else:
            merged = [(current_start, current_end)] # Start with the first valid range

        for next_start, next_end in parsed_ranges[1:]:
             if next_start > next_end: continue # Skip invalid ranges
             last_start, last_end = merged[-1]
             if next_start <= last_end + 1: # Overlap or adjacent
                 merged[-1] = (last_start, max(last_end, next_end))
             else:
                 merged.append((next_start, next_end))

    # Final validation: Ensure all indices are within 0 to num_pages-1
    validated_merged = []
    for start, end in merged:
        valid_start = max(0, start)
        valid_end = min(num_pages - 1, end)
        if valid_start <= valid_end: # Only add if range is still valid after clamping
            validated_merged.append((valid_start, valid_end))

    return validated_merged


def perform_split(input_path: str, output_dir: str, ranges_str: Optional[str],
                  single_pages: bool, output_prefix: Optional[str],
                  progress_queue: queue.Queue, status_queue: queue.Queue):
    """
    Performs the PDF splitting in a separate thread.
    Communicates progress and status via queues.
    """
    try:
        status_queue.put(f"Reading '{os.path.basename(input_path)}'...")
        reader = PdfReader(input_path)
        num_pages = len(reader.pages)

        if num_pages == 0:
            status_queue.put(f"Warning: Input PDF '{os.path.basename(input_path)}' has no pages. Nothing to split.")
            progress_queue.put(("done", 1, 1)) # Signal completion
            return

        if not os.path.isdir(output_dir):
             status_queue.put(f"Creating output directory: {output_dir}")
             try:
                 os.makedirs(output_dir, exist_ok=True)
             except OSError as e:
                 raise OSError(f"Could not create output directory '{output_dir}': {e}") from e


        base_name = os.path.splitext(os.path.basename(input_path))[0]
        prefix = output_prefix if output_prefix else base_name + "_"

        split_instructions = [] # List of (output_filepath, list_of_page_indices)

        if single_pages:
            status_queue.put("Preparing to split into single pages...")
            pad_width = len(str(num_pages))
            for i in range(num_pages):
                page_num_str = str(i + 1).zfill(pad_width)
                output_filename = f"{prefix}page_{page_num_str}.pdf"
                output_filepath = os.path.join(output_dir, output_filename)
                split_instructions.append((output_filepath, [i]))
        elif ranges_str:
            status_queue.put(f"Parsing ranges: '{ranges_str}'...")
            page_indices_ranges = parse_page_ranges(ranges_str, num_pages) # Can raise ValueError

            if not page_indices_ranges:
                status_queue.put("No valid page ranges found after parsing. Nothing to split.")
                progress_queue.put(("done", 1, 1))
                return

            status_queue.put(f"Preparing {len(page_indices_ranges)} output file(s) based on ranges...")
            for i, (start_index, end_index) in enumerate(page_indices_ranges):
                range_desc = f"{start_index + 1}"
                if start_index != end_index:
                    range_desc += f"-{end_index + 1}"
                output_filename = f"{prefix}pages_{range_desc}.pdf"
                output_filepath = os.path.join(output_dir, output_filename)
                pages_in_range = list(range(start_index, end_index + 1))
                split_instructions.append((output_filepath, pages_in_range))
        else:
             # This case should be prevented by the GUI logic
             raise ValueError("Internal Error: No split method specified.")


        total_splits = len(split_instructions)
        if total_splits == 0:
            status_queue.put("No output files to generate based on instructions.")
            progress_queue.put(("done", 1, 1))
            return

        status_queue.put(f"Generating {total_splits} output file(s)...")
        progress_queue.put(("start", 0, total_splits)) # Initialize progress bar

        for i, (output_filepath, page_indices) in enumerate(split_instructions):
            if not page_indices:
                status_queue.put(f"Warning: Skipping empty page range for '{os.path.basename(output_filepath)}'")
                progress_queue.put(("update", i + 1, total_splits))
                continue

            current_pages_str = f"{min(page_indices)+1}-{max(page_indices)+1}" if len(page_indices) > 1 else f"{page_indices[0]+1}"
            status_queue.put(f"  [{i+1}/{total_splits}] Creating '{os.path.basename(output_filepath)}' (Pages {current_pages_str})...")

            writer = PdfWriter()
            pages_added = 0
            for page_index in page_indices:
                # Double-check index validity though parse_page_ranges should ensure it
                if 0 <= page_index < num_pages:
                    try:
                        writer.add_page(reader.pages[page_index])
                        pages_added += 1
                    except Exception as page_err: # Catch potential issues adding specific pages
                         status_queue.put(f"    Warning: Error adding page {page_index+1} to '{os.path.basename(output_filepath)}': {page_err}")

            if pages_added > 0:
                 with open(output_filepath, "wb") as output_pdf:
                     writer.write(output_pdf)
            else:
                 status_queue.put(f"Warning: No valid pages were added for '{os.path.basename(output_filepath)}'. File not created.")

            progress_queue.put(("update", i + 1, total_splits))

        status_queue.put(f"Splitting successfully completed. Output files are in '{output_dir}'.")
        progress_queue.put(("done", total_splits, total_splits))

    except PdfReadError as e:
        status_queue.put(f"Error: Failed to read PDF '{os.path.basename(input_path)}'. It might be corrupted or password-protected. Details: {e}")
        progress_queue.put(("error",))
    except DependencyError as e:
        status_queue.put(f"Error: Missing dependency required by pypdf. Details: {e}")
        progress_queue.put(("error",))
    except ValueError as e: # Catch range parsing errors
        status_queue.put(f"Error: Invalid input - {e}")
        progress_queue.put(("error",))
    except OSError as e: # Catch file/directory creation errors
        status_queue.put(f"Error: File system operation failed - {e}")
        progress_queue.put(("error",))
    except Exception as e:
        status_queue.put(f"Error: An unexpected error occurred during splitting: {e}")
        progress_queue.put(("error",))


def perform_merge(input_paths: List[str], output_path: str,
                  progress_queue: queue.Queue, status_queue: queue.Queue):
    """
    Performs the PDF merging in a separate thread.
    Communicates progress and status via queues.
    """
    merger = PdfWriter()
    total_input_files = len(input_paths)
    files_processed = 0
    total_pages_merged = 0

    try:
        status_queue.put(f"Starting merge of {total_input_files} file(s) into '{os.path.basename(output_path)}'...")
        progress_queue.put(("start", 0, total_input_files + 1)) # +1 for the final write step

        for i, input_path in enumerate(input_paths):
            status_queue.put(f"  [{i+1}/{total_input_files}] Processing '{os.path.basename(input_path)}'...")
            if not os.path.exists(input_path):
                status_queue.put(f"    Warning: File not found, skipping: '{os.path.basename(input_path)}'")
                progress_queue.put(("update", i + 1, total_input_files + 1))
                continue

            try:
                reader = PdfReader(input_path)
                num_pages_in_file = len(reader.pages)

                if num_pages_in_file > 0:
                    merger.append(reader) # Appends all pages
                    status_queue.put(f"    Appended {num_pages_in_file} pages.")
                    total_pages_merged += num_pages_in_file
                    files_processed += 1
                else:
                    status_queue.put(f"    Warning: '{os.path.basename(input_path)}' contains no pages. Skipping append.")

            except PdfReadError as e:
                status_queue.put(f"    Error reading '{os.path.basename(input_path)}': {e}. Skipping this file.")
            except DependencyError as e:
                 status_queue.put(f"    Error: Missing dependency for '{os.path.basename(input_path)}': {e}. Skipping.")
            except Exception as e:
                 status_queue.put(f"    An unexpected error occurred processing '{os.path.basename(input_path)}': {e}. Skipping.")

            progress_queue.put(("update", i + 1, total_input_files + 1))

        if files_processed == 0:
            status_queue.put("Error: No valid input PDF files could be processed. Output file not created.")
            progress_queue.put(("error",))
            merger.close() # Close the writer object
            return

        status_queue.put(f"Writing final merged PDF ({total_pages_merged} pages from {files_processed} files)...")
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
             status_queue.put(f"Creating output directory: {output_dir}")
             try:
                 os.makedirs(output_dir, exist_ok=True)
             except OSError as e:
                 raise OSError(f"Could not create output directory '{output_dir}': {e}") from e

        with open(output_path, "wb") as output_pdf:
            merger.write(output_pdf)

        status_queue.put(f"Merging completed successfully. Output saved as '{output_path}'.")
        progress_queue.put(("done", total_input_files + 1, total_input_files + 1))

    except OSError as e: # Catch file/directory creation/write errors
        status_queue.put(f"Error: File system operation failed - {e}")
        progress_queue.put(("error",))
    except Exception as e:
        status_queue.put(f"Error: An unexpected error occurred during merging/writing: {e}")
        progress_queue.put(("error",))
    finally:
        merger.close() # Ensure the writer is closed even on error


# --- GUI Application Class ---

class PdfToolApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PDF Splitter & Merger Tool")
        # self.root.geometry("650x550") # Adjust size as needed
        self.root.minsize(600, 500)

        # Style configuration
        self.style = ttk.Style()
        self.style.theme_use('clam') # Or 'alt', 'default', 'classic'

        # Variables
        self.split_input_path = tk.StringVar()
        self.split_output_dir = tk.StringVar()
        self.split_method = tk.StringVar(value="ranges") # Default to ranges
        self.split_ranges = tk.StringVar()
        self.split_prefix = tk.StringVar()

        self.merge_input_paths = [] # List to hold paths for merging
        self.merge_output_path = tk.StringVar()

        # Queues for thread communication
        self.progress_queue = queue.Queue()
        self.status_queue = queue.Queue()

        # --- Main Structure ---
        self.notebook = ttk.Notebook(root)

        self.split_frame = ttk.Frame(self.notebook, padding="10")
        self.merge_frame = ttk.Frame(self.notebook, padding="10")

        self.notebook.add(self.split_frame, text='Split PDF')
        self.notebook.add(self.merge_frame, text='Merge PDFs')
        self.notebook.pack(expand=True, fill='both', padx=10, pady=(10, 0))

        # --- Populate Split Frame ---
        self.create_split_widgets()

        # --- Populate Merge Frame ---
        self.create_merge_widgets()

        # --- Status Bar and Progress Bar ---
        self.status_frame = ttk.Frame(root, relief=tk.SUNKEN, padding=2)
        self.status_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(5, 10))

        self.status_label = ttk.Label(self.status_frame, text="Ready", anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.progress_bar = ttk.Progressbar(self.status_frame, orient=tk.HORIZONTAL, mode='determinate')
        # Pack progress bar later when needed, or pack hidden initially
        # self.progress_bar.pack(side=tk.RIGHT)

        # Start monitoring queues
        self.root.after(100, self.check_queues)

    # --- Widget Creation Methods ---

    def create_split_widgets(self):
        frame = self.split_frame
        frame.columnconfigure(1, weight=1) # Allow entry fields to expand

        # Input File
        ttk.Label(frame, text="Input PDF:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(frame, textvariable=self.split_input_path, width=50).grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        ttk.Button(frame, text="Browse...", command=self.browse_split_input).grid(row=0, column=2, padx=5, pady=5)

        # Output Directory
        ttk.Label(frame, text="Output Dir:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(frame, textvariable=self.split_output_dir, width=50).grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)
        ttk.Button(frame, text="Browse...", command=self.browse_split_output).grid(row=1, column=2, padx=5, pady=5)

        # Split Method Selection
        method_frame = ttk.LabelFrame(frame, text="Split Method", padding="5")
        method_frame.grid(row=2, column=0, columnspan=3, padx=5, pady=10, sticky=tk.EW)

        self.radio_ranges = ttk.Radiobutton(method_frame, text="Split by Page Ranges", variable=self.split_method,
                                             value="ranges", command=self.update_split_ui_state)
        self.radio_ranges.grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)

        self.radio_single = ttk.Radiobutton(method_frame, text="Split into Single Pages", variable=self.split_method,
                                            value="single", command=self.update_split_ui_state)
        self.radio_single.grid(row=1, column=0, padx=5, pady=2, sticky=tk.W)

        # Page Ranges Input (conditionally enabled)
        ttk.Label(frame, text="Page Ranges:").grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        self.ranges_entry = ttk.Entry(frame, textvariable=self.split_ranges, width=50)
        self.ranges_entry.grid(row=3, column=1, padx=5, pady=5, sticky=tk.EW)
        ttk.Label(frame, text="(e.g., 1-3, 5, 8-)").grid(row=3, column=2, padx=5, pady=5, sticky=tk.W)

        # Output Prefix (Optional)
        ttk.Label(frame, text="Output Prefix:").grid(row=4, column=0, padx=5, pady=5, sticky=tk.W)
        self.prefix_entry = ttk.Entry(frame, textvariable=self.split_prefix, width=50)
        self.prefix_entry.grid(row=4, column=1, padx=5, pady=5, sticky=tk.EW)
        ttk.Label(frame, text="(Optional)").grid(row=4, column=2, padx=5, pady=5, sticky=tk.W)

        # Action Button
        self.split_button = ttk.Button(frame, text="Split PDF", command=self.start_split_thread)
        self.split_button.grid(row=5, column=0, columnspan=3, pady=20)

        # Initial UI state
        self.update_split_ui_state()

    def create_merge_widgets(self):
        frame = self.merge_frame
        frame.columnconfigure(0, weight=1) # Allow listbox to expand
        frame.rowconfigure(1, weight=1) # Allow listbox frame to expand vertically

        # Input Files Section
        input_files_frame = ttk.LabelFrame(frame, text="Input PDFs (in order)", padding="5")
        input_files_frame.grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky=tk.NSEW)
        input_files_frame.columnconfigure(0, weight=1)
        input_files_frame.rowconfigure(0, weight=1)

        # Listbox for Files
        self.merge_listbox = tk.Listbox(input_files_frame, selectmode=tk.EXTENDED, width=60, height=10)
        self.merge_listbox.grid(row=0, column=0, padx=5, pady=5, sticky=tk.NSEW)
        listbox_scrollbar = ttk.Scrollbar(input_files_frame, orient=tk.VERTICAL, command=self.merge_listbox.yview)
        listbox_scrollbar.grid(row=0, column=1, sticky=tk.NS)
        self.merge_listbox.configure(yscrollcommand=listbox_scrollbar.set)

        # Buttons for Listbox Management
        button_frame = ttk.Frame(input_files_frame)
        button_frame.grid(row=1, column=0, columnspan=2, pady=5)

        ttk.Button(button_frame, text="Add Files...", command=self.add_merge_files).pack(side=tk.LEFT, padx=3)
        ttk.Button(button_frame, text="Remove Selected", command=self.remove_merge_files).pack(side=tk.LEFT, padx=3)
        ttk.Button(button_frame, text="Clear List", command=self.clear_merge_list).pack(side=tk.LEFT, padx=3)
        ttk.Button(button_frame, text="Move Up", command=lambda: self.move_merge_item(-1)).pack(side=tk.LEFT, padx=3)
        ttk.Button(button_frame, text="Move Down", command=lambda: self.move_merge_item(1)).pack(side=tk.LEFT, padx=3)


        # Output File
        output_frame = ttk.Frame(frame)
        output_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=10, sticky=tk.EW)
        output_frame.columnconfigure(1, weight=1)

        ttk.Label(output_frame, text="Output File:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Entry(output_frame, textvariable=self.merge_output_path, width=50).grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        ttk.Button(output_frame, text="Save As...", command=self.browse_merge_output).grid(row=0, column=2, padx=5, pady=5)

        # Action Button
        self.merge_button = ttk.Button(frame, text="Merge PDFs", command=self.start_merge_thread)
        self.merge_button.grid(row=2, column=0, columnspan=2, pady=15)

    # --- UI State and Event Handlers ---

    def update_split_ui_state(self):
        """Enable/disable ranges entry based on radio button selection."""
        if self.split_method.get() == "ranges":
            self.ranges_entry.config(state=tk.NORMAL)
        else:
            self.ranges_entry.config(state=tk.DISABLED)

    def browse_split_input(self):
        filepath = filedialog.askopenfilename(
            title="Select Input PDF",
            filetypes=[("PDF Files", "*.pdf"), ("All Files", "*.*")]
        )
        if filepath:
            self.split_input_path.set(filepath)
            self.set_status(f"Selected input: {os.path.basename(filepath)}")

    def browse_split_output(self):
        dirpath = filedialog.askdirectory(
            title="Select Output Directory",
            mustexist=False # Allow creating a new directory implicitly later
        )
        if dirpath:
            self.split_output_dir.set(dirpath)
            self.set_status(f"Selected output directory: {dirpath}")

    def add_merge_files(self):
        filepaths = filedialog.askopenfilenames(
             title="Select Input PDFs to Merge",
             filetypes=[("PDF Files", "*.pdf"), ("All Files", "*.*")]
        )
        if filepaths:
            count = 0
            for fp in filepaths:
                 if fp not in self.merge_input_paths: # Avoid duplicates visually
                    self.merge_input_paths.append(fp)
                    self.merge_listbox.insert(tk.END, os.path.basename(fp)) # Display only filename
                    count += 1
            self.set_status(f"Added {count} file(s) to merge list.")

    def remove_merge_files(self):
        selected_indices = self.merge_listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("Selection Error", "Please select file(s) to remove from the list.")
            return

        # Remove from listbox and internal list (iterate backwards for indices)
        removed_count = 0
        for i in reversed(selected_indices):
            del self.merge_input_paths[i]
            self.merge_listbox.delete(i)
            removed_count += 1
        self.set_status(f"Removed {removed_count} file(s) from merge list.")


    def clear_merge_list(self):
        if not self.merge_input_paths:
            return
        if messagebox.askyesno("Confirm Clear", "Are you sure you want to clear the entire merge list?"):
            self.merge_listbox.delete(0, tk.END)
            self.merge_input_paths.clear()
            self.set_status("Merge list cleared.")

    def move_merge_item(self, direction):
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
            return # Cannot move further

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
        self.merge_listbox.see(new_index) # Ensure visible

    def browse_merge_output(self):
         filepath = filedialog.asksaveasfilename(
             title="Save Merged PDF As...",
             defaultextension=".pdf",
             filetypes=[("PDF Files", "*.pdf"), ("All Files", "*.*")]
         )
         if filepath:
             # Ensure it ends with .pdf if user didn't type it
             if not filepath.lower().endswith(".pdf"):
                 filepath += ".pdf"
             self.merge_output_path.set(filepath)
             self.set_status(f"Selected output file: {os.path.basename(filepath)}")

    # --- Threading and Processing ---

    def set_status(self, message):
        """Updates the status bar label safely."""
        self.status_label.config(text=message)
        self.root.update_idletasks() # Force update

    def set_ui_state(self, enabled: bool):
        """Enable or disable major UI elements during processing."""
        state = tk.NORMAL if enabled else tk.DISABLED
        # Disable tabs/buttons
        for i in range(self.notebook.index("end")):
             self.notebook.tab(i, state=state)

        # Disable specific buttons within frames if needed (more granular control)
        # Example: self.split_button.config(state=state)
        # Example: self.merge_button.config(state=state)
        # For simplicity, disabling tabs might be sufficient, but below disables buttons too

        # Split Frame buttons/entries
        for child in self.split_frame.winfo_children():
            if isinstance(child, (ttk.Button, ttk.Entry, ttk.Radiobutton, ttk.LabelFrame)):
                 try:
                      child.config(state=state)
                 except tk.TclError: # Some widgets might not support state config directly
                      pass
            # Handle widgets inside LabelFrames
            if isinstance(child, ttk.LabelFrame):
                for sub_child in child.winfo_children():
                     if isinstance(sub_child, (ttk.Button, ttk.Entry, ttk.Radiobutton)):
                         try: sub_child.config(state=state)
                         except tk.TclError: pass

        # Merge Frame buttons/entries/listbox
        for child in self.merge_frame.winfo_children():
             if isinstance(child, (ttk.Button, ttk.Entry, ttk.LabelFrame, tk.Listbox)):
                  try: child.config(state=state)
                  except tk.TclError: pass
             # Handle widgets inside LabelFrames
             if isinstance(child, ttk.LabelFrame):
                  for sub_child in child.winfo_children():
                       if isinstance(sub_child, (ttk.Button, ttk.Entry, tk.Listbox)):
                           try: sub_child.config(state=state)
                           except tk.TclError: pass


        # Re-enable range entry based on radio button if enabling UI
        if enabled:
            self.update_split_ui_state()


    def show_progress(self):
        """Make the progress bar visible."""
        self.progress_bar.pack(side=tk.RIGHT, padx=(5,0))
        self.status_frame.update_idletasks()

    def hide_progress(self):
        """Hide the progress bar."""
        self.progress_bar.pack_forget()
        self.progress_bar['value'] = 0
        self.status_frame.update_idletasks()

    def start_split_thread(self):
        """Validate inputs and start the split operation in a thread."""
        input_pdf = self.split_input_path.get()
        output_dir = self.split_output_dir.get()
        method = self.split_method.get()
        ranges = self.split_ranges.get()
        prefix = self.split_prefix.get()

        # --- Input Validation ---
        if not input_pdf or not os.path.exists(input_pdf):
            messagebox.showerror("Input Error", "Please select a valid input PDF file.")
            return
        if not os.path.isfile(input_pdf):
             messagebox.showerror("Input Error", "The selected input path is not a file.")
             return
        if not output_dir:
            messagebox.showerror("Input Error", "Please select an output directory.")
            return
        # No need to check if output_dir exists here, perform_split handles creation

        if method == "ranges" and not ranges:
            messagebox.showerror("Input Error", "Please enter the page ranges to split by.")
            return
        if method == "ranges":
            # Basic validation for range format characters - more robust check in parse_page_ranges
             if not re.fullmatch(r"[\d\s,-]+", ranges):
                  messagebox.showerror("Input Error", "Invalid characters in page ranges.\nUse numbers, commas, hyphens, and spaces only.")
                  return


        # --- Start Processing ---
        self.set_ui_state(enabled=False)
        self.set_status("Starting PDF split...")
        self.show_progress()
        self.progress_bar['mode'] = 'determinate'
        self.progress_bar['value'] = 0

        # Clear queues before starting
        while not self.status_queue.empty(): self.status_queue.get()
        while not self.progress_queue.empty(): self.progress_queue.get()


        self.worker_thread = threading.Thread(
            target=perform_split,
            args=(input_pdf, output_dir, ranges if method == "ranges" else None,
                  method == "single", prefix, self.progress_queue, self.status_queue),
            daemon=True # Allows exiting the main app even if thread is running (use carefully)
        )
        self.worker_thread.start()


    def start_merge_thread(self):
        """Validate inputs and start the merge operation in a thread."""
        input_files = self.merge_input_paths
        output_file = self.merge_output_path.get()

        # --- Input Validation ---
        if not input_files:
            messagebox.showerror("Input Error", "Please add at least one PDF file to merge.")
            return
        if len(input_files) < 2:
             messagebox.showwarning("Input Warning", "Merging requires at least two files. Continuing will essentially copy the single selected file.")
             # Allow continuing if they really want to just copy one file via merge

        if not output_file:
            messagebox.showerror("Input Error", "Please select an output file path.")
            return

        # Basic check if output is same as any input (can cause issues)
        abs_output_path = os.path.abspath(output_file)
        for in_path in input_files:
             if os.path.abspath(in_path) == abs_output_path:
                  messagebox.showerror("Input Error", f"Output file cannot be the same as an input file:\n{os.path.basename(in_path)}")
                  return


        # --- Start Processing ---
        self.set_ui_state(enabled=False)
        self.set_status("Starting PDF merge...")
        self.show_progress()
        self.progress_bar['mode'] = 'determinate'
        self.progress_bar['value'] = 0

        # Clear queues before starting
        while not self.status_queue.empty(): self.status_queue.get()
        while not self.progress_queue.empty(): self.progress_queue.get()

        self.worker_thread = threading.Thread(
            target=perform_merge,
            args=(list(input_files), output_file, self.progress_queue, self.status_queue), # Pass a copy
            daemon=True
        )
        self.worker_thread.start()


    def check_queues(self):
        """Periodically check queues for updates from the worker thread."""
        try:
            # Process status messages
            while not self.status_queue.empty():
                msg = self.status_queue.get_nowait()
                self.set_status(msg)
                # Check for error messages to show popups
                if "Error:" in msg:
                     messagebox.showerror("Processing Error", msg)
                elif "Warning:" in msg:
                     messagebox.showwarning("Processing Warning", msg)


            # Process progress updates
            while not self.progress_queue.empty():
                progress_update = self.progress_queue.get_nowait()
                ptype = progress_update[0]

                if ptype == "start":
                    _, current, total = progress_update
                    self.progress_bar['maximum'] = total
                    self.progress_bar['value'] = current
                elif ptype == "update":
                    _, current, total = progress_update
                    self.progress_bar['maximum'] = total # Update total just in case
                    self.progress_bar['value'] = current
                elif ptype == "done":
                    _, current, total = progress_update
                    self.progress_bar['maximum'] = total
                    self.progress_bar['value'] = current
                    self.set_ui_state(enabled=True)
                    self.hide_progress()
                    # Optional: Show success message box
                    # messagebox.showinfo("Success", "Operation completed successfully!")
                elif ptype == "error":
                    self.set_ui_state(enabled=True)
                    self.hide_progress()
                    # Error message already shown via status queue check

        except queue.Empty:
            pass # No messages currently
        except Exception as e:
             print(f"Error checking queues: {e}") # Debugging

        # Reschedule the check
        self.root.after(100, self.check_queues)


# --- Main Execution ---
if __name__ == "__main__":
    root = tk.Tk()
    app = PdfToolApp(root)
    root.mainloop()
