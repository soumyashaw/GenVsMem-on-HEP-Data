#!/usr/bin/env python
"""Utility script to manage data cache files.

Usage:
    python -m src.data.manage_cache list      # List all cache files
    python -m src.data.manage_cache clear     # Clear all cache files
    python -m src.data.manage_cache --help    # Show help
"""

import argparse
from src.data.cache_data import list_cache_files, clear_cache


def main():
    parser = argparse.ArgumentParser(
        description="Manage data cache files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "action",
        choices=["list", "clear"],
        help="Action to perform: 'list' shows cache files, 'clear' deletes them",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="cache",
        help="Cache directory path (default: cache)",
    )
    
    args = parser.parse_args()
    
    if args.action == "list":
        list_cache_files(cache_dir=args.cache_dir)
    elif args.action == "clear":
        response = input(f"Are you sure you want to delete all cache files in '{args.cache_dir}'? [y/N]: ")
        if response.lower() in ['y', 'yes']:
            clear_cache(cache_dir=args.cache_dir)
        else:
            print("Cancelled")


if __name__ == "__main__":
    main()
