#!/usr/bin/env python3

"""
A full-featured command-line utility for splitting and merging PDF files.

Requires the 'pypdf' library: pip install pypdf
"""

import argparse
import os
import sys
from pypdf import PdfReader, PdfWriter, PageRange
from pypdf.errors import PdfReadError
import re
from typing import List, Tuple, Optional

def parse_page_ranges(ranges_str: str, num_pages: int) -> List[Tuple[int, int]]:
    """
    Parses a comma-separated string of page ranges (1-based index) into a list of
    tuples representing (start_page_index, end_page_index) (0-based, inclusive).

    Examples:
        "1,3,5-7" -> [(0, 0), (2, 2), (4, 6)]
        "1-3,8-" -> [(0, 2), (7, num_pages - 1)]
        "-5,10" -> [(0, 4), (9, 9)]

    Args:
        ranges_str: The string containing page ranges.
        num_pages: The total number of pages in the PDF for validation and open ranges.

    Returns:
        A list of tuples (start_index, end_index), 0-based inclusive.

    Raises:
        ValueError: If the range string is invalid or page numbers are out of bounds.
    """
    if not ranges_str:
        raise ValueError("Page range string cannot be empty.")

    parsed_ranges = []
    parts = ranges_str.split(',')

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Check for range format (e.g., "1-5", "5-", "-3")
        if '-' in part:
            start_str, end_str = part.split('-', 1)
            start_str = start_str.strip()
            end_str = end_str.strip()

            try:
                # Handle open start (e.g., "-5")
                if not start_str:
                    start_page = 1
                else:
                    start_page = int(start_str)
                    if start_page < 1:
                        raise ValueError(f"Start page cannot be less than 1: '{start_str}'")

                # Handle open end (e.g., "5-")
                if not end_str:
                    end_page = num_pages
                else:
                    end_page = int(end_str)
                    if end_page < 1:
                         raise ValueError(f"End page cannot be less than 1: '{end_str}'")

                if start_page > end_page:
                    raise ValueError(f"Start page ({start_page}) cannot be greater than end page ({end_page}) in range '{part}'")
                if start_page > num_pages:
                     raise ValueError(f"Start page ({start_page}) exceeds total pages ({num_pages}) in range '{part}'")
                if end_page > num_pages:
                     print(f"Warning: End page ({end_page}) in range '{part}' exceeds total pages ({num_pages}). Adjusting to {num_pages}.", file=sys.stderr)
                     end_page = num_pages

                # Convert to 0-based inclusive indices
                parsed_ranges.append((start_page - 1, end_page - 1))

            except ValueError as e:
                # Re-raise with more context if it's just an int conversion error
                if "invalid literal for int()" in str(e):
                    raise ValueError(f"Invalid number in page range '{part}'.") from e
                raise # Re-raise validation errors

        # Handle single page (e.g., "3")
        else:
            try:
                page = int(part)
                if page < 1 or page > num_pages:
                    raise ValueError(f"Page number {page} is out of valid range (1-{num_pages}).")
                # Convert to 0-based inclusive indices
                parsed_ranges.append((page - 1, page - 1))
            except ValueError as e:
                if "invalid literal for int()" in str(e):
                     raise ValueError(f"Invalid page number '{part}'. Must be an integer or a range (e.g., '1-5').") from e
                raise # Re-raise validation errors

    # Sort ranges by start page and merge overlapping/adjacent ranges (optional but good practice)
    if not parsed_ranges:
        return []

    parsed_ranges.sort(key=lambda x: x[0])

    merged = []
    current_start, current_end = parsed_ranges[0]

    for next_start, next_end in parsed_ranges[1:]:
        if next_start <= current_end + 1:
            # Overlap or adjacent, extend the current range
            current_end = max(current_end, next_end)
        else:
            # Gap found, finalize the current range and start a new one
            merged.append((current_start, current_end))
            current_start, current_end = next_start, next_end

    merged.append((current_start, current_end)) # Add the last range

    return merged


def split_pdf(input_path: str, output_dir: str, ranges_str: Optional[str] = None, single_pages: bool = False, output_prefix: Optional[str] = None):
    """
    Splits a PDF file based on specified page ranges or into single pages.

    Args:
        input_path: Path to the input PDF file.
        output_dir: Directory where the output split files will be saved.
        ranges_str: Comma-separated string of page ranges (e.g., "1-3,5,7-").
                    Mutually exclusive with single_pages.
        single_pages: If True, split the PDF into individual pages.
                      Mutually exclusive with ranges_str.
        output_prefix: Optional prefix for output filenames. If None, uses the
                       input filename base.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    if not os.path.isdir(output_dir):
        try:
            print(f"Creating output directory: {output_dir}")
            os.makedirs(output_dir)
        except OSError as e:
            print(f"Error: Could not create output directory '{output_dir}': {e}", file=sys.stderr)
            sys.exit(1)

    if ranges_str and single_pages:
        print("Error: Cannot specify both --ranges and --single-pages.", file=sys.stderr)
        sys.exit(1)
    if not ranges_str and not single_pages:
        print("Error: Must specify either --ranges or --single-pages for splitting.", file=sys.stderr)
        sys.exit(1)

    try:
        reader = PdfReader(input_path)
        num_pages = len(reader.pages)

        if num_pages == 0:
            print(f"Warning: Input PDF '{input_path}' has no pages. Nothing to split.", file=sys.stderr)
            return

        print(f"Input PDF: '{input_path}' ({num_pages} pages)")

        if output_prefix is None:
             base_name = os.path.splitext(os.path.basename(input_path))[0]
             output_prefix = base_name + "_"

        split_instructions = [] # List of (output_filename, list_of_page_indices)

        if single_pages:
            print("Splitting into single pages.")
            if num_pages > 0:
                # Calculate padding width for page numbers (e.g., 001, 010, 100)
                pad_width = len(str(num_pages))
                for i in range(num_pages):
                    page_num_str = str(i + 1).zfill(pad_width)
                    output_filename = f"{output_prefix}page_{page_num_str}.pdf"
                    output_filepath = os.path.join(output_dir, output_filename)
                    split_instructions.append((output_filepath, [i]))
            else:
                 print("Input PDF has no pages, cannot split into single pages.", file=sys.stderr)


        elif ranges_str:
            print(f"Splitting using ranges: '{ranges_str}'")
            try:
                page_indices_ranges = parse_page_ranges(ranges_str, num_pages)
            except ValueError as e:
                print(f"Error parsing page ranges: {e}", file=sys.stderr)
                sys.exit(1)

            if not page_indices_ranges:
                 print("No valid page ranges were parsed. Nothing to split.", file=sys.stderr)
                 return

            for i, (start_index, end_index) in enumerate(page_indices_ranges):
                # Generate a descriptive filename for the range
                # Add 1 to indices for 1-based display in filename
                range_desc = f"{start_index + 1}"
                if start_index != end_index:
                    range_desc += f"-{end_index + 1}"

                output_filename = f"{output_prefix}pages_{range_desc}.pdf"
                output_filepath = os.path.join(output_dir, output_filename)
                pages_in_range = list(range(start_index, end_index + 1))
                split_instructions.append((output_filepath, pages_in_range))

        # --- Perform the actual splitting based on instructions ---
        total_splits = len(split_instructions)
        print(f"Generating {total_splits} output file(s)...")

        for i, (output_filepath, page_indices) in enumerate(split_instructions):
            if not page_indices: # Should not happen with current logic, but safety check
                print(f"Warning: Skipping empty page range for '{output_filepath}'", file=sys.stderr)
                continue

            writer = PdfWriter()
            try:
                print(f"  [{i+1}/{total_splits}] Creating '{os.path.basename(output_filepath)}' (Pages {min(page_indices)+1}-{max(page_indices)+1})...")
                for page_index in page_indices:
                    if 0 <= page_index < num_pages:
                        writer.add_page(reader.pages[page_index])
                    else:
                        # This should be caught by parse_page_ranges, but defensive check
                         print(f"Warning: Invalid page index {page_index+1} requested. Skipping.", file=sys.stderr)

                if len(writer.pages) > 0:
                     with open(output_filepath, "wb") as output_pdf:
                         writer.write(output_pdf)
                else:
                     print(f"Warning: No valid pages found for range resulting in '{output_filepath}'. File not created.", file=sys.stderr)

            except Exception as e:
                 print(f"Error writing output file '{output_filepath}': {e}", file=sys.stderr)
            finally:
                # pypdf doesn't require explicit closing of writer in this context
                # but good practice if it held file handles directly
                pass

        print(f"Splitting completed. Output files are in '{output_dir}'.")

    except PdfReadError as e:
        print(f"Error reading input PDF '{input_path}': {e}. Is it a valid PDF?", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during splitting: {e}", file=sys.stderr)
        sys.exit(1)


def merge_pdfs(input_paths: List[str], output_path: str):
    """
    Merges multiple PDF files into a single output PDF file.

    Args:
        input_paths: A list of paths to the input PDF files, in the desired merge order.
        output_path: Path to the output merged PDF file.
    """
    if not input_paths:
        print("Error: No input files specified for merging.", file=sys.stderr)
        sys.exit(1)

    if os.path.dirname(output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    merger = PdfWriter()
    total_input_files = len(input_paths)
    print(f"Starting merge of {total_input_files} PDF file(s) into '{output_path}'...")

    files_processed = 0
    total_pages_merged = 0

    for i, input_path in enumerate(input_paths):
        if not os.path.exists(input_path):
            print(f"Warning: Input file not found, skipping: {input_path}", file=sys.stderr)
            continue

        try:
            print(f"  [{i+1}/{total_input_files}] Processing '{input_path}'...")
            reader = PdfReader(input_path)
            num_pages_in_file = len(reader.pages)

            if num_pages_in_file > 0:
                 merger.append(reader) # Efficiently appends all pages
                 print(f"    Appended {num_pages_in_file} pages.")
                 total_pages_merged += num_pages_in_file
                 files_processed += 1
            else:
                 print(f"    Warning: '{input_path}' contains no pages. Skipping append.", file=sys.stderr)

        except PdfReadError as e:
            print(f"Error reading input PDF '{input_path}': {e}. Skipping this file.", file=sys.stderr)
        except Exception as e:
            print(f"An unexpected error occurred while processing '{input_path}': {e}. Skipping this file.", file=sys.stderr)

    if files_processed == 0:
        print("Error: No valid input PDF files could be processed. Output file not created.", file=sys.stderr)
        sys.exit(1)

    try:
        print(f"Writing final merged PDF ({total_pages_merged} pages from {files_processed} files)...")
        with open(output_path, "wb") as output_pdf:
            merger.write(output_pdf)
        print(f"Merging completed. Output file saved as '{output_path}'.")
    except Exception as e:
        print(f"Error writing merged output file '{output_path}': {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        # pypdf doesn't require explicit closing of writer in this context
        # but good practice if it held file handles directly
        pass


def main():
    parser = argparse.ArgumentParser(
        description="A command-line tool to split or merge PDF files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  Split PDF into single pages:
    %(prog)s split -i document.pdf -o output_dir --single-pages

  Split PDF using specific ranges:
    %(prog)s split -i document.pdf -o output_dir --ranges "1-3,5,8-"

  Split PDF using ranges with custom output prefix:
    %(prog)s split -i doc.pdf -o out --ranges "1,3" --prefix "split_doc_"

  Merge multiple PDFs into one:
    %(prog)s merge -i file1.pdf file2.pdf file3.pdf -o merged_output.pdf
"""
    )

    subparsers = parser.add_subparsers(dest="command", required=True, help="Choose a command: 'split' or 'merge'")

    # --- Split Subparser ---
    parser_split = subparsers.add_parser("split", help="Split a single PDF into multiple files.")
    parser_split.add_argument("-i", "--input", required=True, help="Path to the input PDF file.")
    parser_split.add_argument("-o", "--output-dir", required=True, help="Directory to save the split output files.")
    parser_split.add_argument("--prefix", help="Optional prefix for the output filenames (default uses input filename).")

    # Group for mutually exclusive split methods
    split_method_group = parser_split.add_mutually_exclusive_group(required=True)
    split_method_group.add_argument("--ranges", help="Comma-separated page ranges (e.g., '1-3,5,7-', '-5'). Pages are 1-based.")
    split_method_group.add_argument("--single-pages", action="store_true", help="Split the PDF into individual pages.")

    # --- Merge Subparser ---
    parser_merge = subparsers.add_parser("merge", help="Merge multiple PDF files into a single file.")
    parser_merge.add_argument("-i", "--input", required=True, nargs='+', help="Paths to the input PDF files (in merge order).")
    parser_merge.add_argument("-o", "--output", required=True, help="Path for the final merged output PDF file.")


    args = parser.parse_args()

    if args.command == "split":
        split_pdf(
            input_path=args.input,
            output_dir=args.output_dir,
            ranges_str=args.ranges,
            single_pages=args.single_pages,
            output_prefix=args.prefix
        )
    elif args.command == "merge":
        merge_pdfs(
            input_paths=args.input,
            output_path=args.output
        )
    else:
        # Should be caught by argparse `required=True` on subparsers
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
