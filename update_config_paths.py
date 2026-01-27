#!/usr/bin/env python3
"""
Script to automatically update dataset paths and results paths in aggregator config YAML files.

This script finds all aggregator YAML config files (config/aggregator_*.yaml) and updates:
- data_file (dataset path) in config/aggregator_*.yaml files
- results_file (results path) in config/aggregator_*.yaml files
"""

import os
import sys
import yaml
import glob
from pathlib import Path
from typing import Dict, Any, Optional, Tuple


def find_config_files(config_dirs: list = None) -> list:
    """Find all aggregator YAML config files in specified directories."""
    if config_dirs is None:
        config_dirs = ["config"]
    
    config_files = []
    for config_dir in config_dirs:
        if os.path.exists(config_dir):
            # Only find files starting with "aggregator_"
            pattern = os.path.join(config_dir, "aggregator_*.yaml")
            config_files.extend(glob.glob(pattern))
            pattern = os.path.join(config_dir, "aggregator_*.yml")
            config_files.extend(glob.glob(pattern))
    
    return sorted(config_files)


def derive_results_path_from_dataset_path(dataset_path: str, current_results_file: str) -> str:
    """
    Derive results path from dataset path.
    If dataset path is 'manufacturing_datasets/index.csv', 
    results path becomes 'manufacturing_datasets_results/results_<algorithm>.pkl'
    """
    # Extract directory from dataset path
    dataset_dir = os.path.dirname(dataset_path)
    
    # If dataset_dir is empty (just filename), use current results directory
    if not dataset_dir:
        return current_results_file
    
    # Append '_results' to the directory name
    results_dir = dataset_dir + '_results'
    
    # Keep the original filename from current_results_file
    results_filename = os.path.basename(current_results_file)
    
    # Combine to create new results path
    new_results_path = os.path.join(results_dir, results_filename)
    
    # Normalize path separators (use forward slashes for consistency)
    return new_results_path.replace('\\', '/')


def ensure_results_directory_exists(results_path: str, dry_run: bool = False) -> Tuple[bool, str]:
    """
    Create the directory for a results file path if it doesn't exist.
    
    Args:
        results_path: Full path to the results file (e.g., "data/results/file.pkl")
        dry_run: If True, don't actually create the directory
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    # Extract directory from the results path
    results_dir = os.path.dirname(results_path)
    
    # If no directory specified (just filename), no directory to create
    if not results_dir:
        return True, "No directory to create (results path is just a filename)"
    
    # Check if directory already exists
    if os.path.exists(results_dir):
        return True, f"Directory already exists: {results_dir}"
    
    # Create directory if not in dry run mode
    if not dry_run:
        try:
            os.makedirs(results_dir, exist_ok=True)
            return True, f"Created directory: {results_dir}"
        except Exception as e:
            return False, f"Failed to create directory {results_dir}: {str(e)}"
    else:
        return True, f"Would create directory: {results_dir} (dry run)"


def update_config_file(
    filepath: str,
    new_dataset_path: Optional[str] = None,
    new_results_path: Optional[str] = None,
    auto_derive_results: bool = True,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Update dataset and results paths in a config YAML file.
    
    Returns:
        Dictionary with update information: {
            'file': filepath,
            'updated': bool,
            'changes': list of change descriptions
        }
    """
    changes = []
    updated = False
    
    try:
        # Read the YAML file
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            config = yaml.safe_load(content)
        
        if config is None:
            return {'file': filepath, 'updated': False, 'changes': ['Empty or invalid YAML']}
        
        # Determine file type based on directory
        file_dir = os.path.normpath(os.path.dirname(filepath))
        dir_basename = os.path.basename(file_dir)
        # Check if file is in config directory
        is_config_dir = dir_basename == 'config'
        
        # Only process aggregator files in config directory
        filename = os.path.basename(filepath)
        is_aggregator_file = filename.startswith('aggregator_')
        
        # Update config/aggregator_*.yaml files only
        if is_config_dir and is_aggregator_file:
            if 'data' in config:
                # Update data_file
                if new_dataset_path is not None and 'data_file' in config['data']:
                    old_path = config['data']['data_file']
                    if old_path != new_dataset_path:
                        if not dry_run:
                            config['data']['data_file'] = new_dataset_path
                        changes.append(f"data_file: '{old_path}' -> '{new_dataset_path}'")
                        updated = True
                    
                    # Auto-derive results path from dataset path if enabled
                    if auto_derive_results and 'results_file' in config['data']:
                        current_results_file = config['data']['results_file']
                        derived_results_path = derive_results_path_from_dataset_path(new_dataset_path, current_results_file)
                        if current_results_file != derived_results_path:
                            # Create directory for results file if it doesn't exist
                            dir_success, dir_message = ensure_results_directory_exists(derived_results_path, dry_run=dry_run)
                            if not dir_success:
                                changes.append(f"WARNING: {dir_message}")
                            elif not dry_run:
                                changes.append(f"Directory: {dir_message}")
                            
                            if not dry_run:
                                config['data']['results_file'] = derived_results_path
                            changes.append(f"results_file: '{current_results_file}' -> '{derived_results_path}' (auto-derived)")
                            updated = True
                
                # Update results_file manually if explicitly provided (overrides auto-derive)
                if new_results_path is not None and 'results_file' in config['data']:
                    old_path = config['data']['results_file']
                    if old_path != new_results_path:
                        # Create directory for results file if it doesn't exist
                        dir_success, dir_message = ensure_results_directory_exists(new_results_path, dry_run=dry_run)
                        if not dir_success:
                            changes.append(f"WARNING: {dir_message}")
                        elif not dry_run:
                            changes.append(f"Directory: {dir_message}")
                        
                        if not dry_run:
                            config['data']['results_file'] = new_results_path
                        changes.append(f"results_file: '{old_path}' -> '{new_results_path}'")
                        updated = True
        
        # Write back if changes were made
        if updated and not dry_run:
            # Preserve YAML structure and comments as much as possible
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        
        return {'file': filepath, 'updated': updated, 'changes': changes}
    
    except Exception as e:
        return {'file': filepath, 'updated': False, 'changes': [f'Error: {str(e)}']}


def main():
    """Main function to update all config files."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Update dataset and results paths in aggregator config YAML files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Update dataset path only (results path auto-derived: manufacturing_datasets -> manufacturing_datasets_results/results_*.pkl)
  python update_config_paths.py --dataset-path "manufacturing_datasets/index.csv"
  
  # Update results path only
  python update_config_paths.py --results-path "results_new.pkl"
  
  # Update both paths explicitly
  python update_config_paths.py --dataset-path "data/new_datasets/index.csv" --results-path "results_new.pkl"
  
  # Update dataset path without auto-deriving results path
  python update_config_paths.py --dataset-path "manufacturing_datasets/index.csv" --no-auto-derive
  
  # Dry run to see what would change
  python update_config_paths.py --dataset-path "manufacturing_datasets/index.csv" --dry-run
        """
    )
    
    parser.add_argument(
        '--dataset-path',
        type=str,
        default=None,
        help='New dataset path to set in config files (e.g., "data/manufacturing_datasets/index.csv")'
    )
    
    parser.add_argument(
        '--results-path',
        type=str,
        default=None,
        help='New results path to set in config files (e.g., "results_new.pkl"). If not provided and --dataset-path is set, results path will be auto-derived from dataset path (directory + "_results" + original filename).'
    )
    
    parser.add_argument(
        '--no-auto-derive',
        action='store_true',
        help='Disable automatic derivation of results path from dataset path'
    )
    
    parser.add_argument(
        '--config-dirs',
        type=str,
        nargs='+',
        default=['config'],
        help='Directories to search for aggregator config files (default: config)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be changed without actually modifying files'
    )
    
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Interactive mode: prompt for paths if not provided'
    )
    
    args = parser.parse_args()
    
    # Interactive mode
    if args.interactive:
        if args.dataset_path is None:
            current_dir = os.getcwd()
            print(f"\nCurrent directory: {current_dir}")
            dataset_input = input("Enter new dataset path (or press Enter to skip): ").strip()
            if dataset_input:
                args.dataset_path = dataset_input
        
        if args.results_path is None:
            results_input = input("Enter new results path (or press Enter to skip): ").strip()
            if results_input:
                args.results_path = results_input
    
    # Validate that at least one path is provided
    if args.dataset_path is None and args.results_path is None:
        print("Error: At least one of --dataset-path or --results-path must be provided.")
        print("Use --interactive to enter paths interactively, or provide them as arguments.")
        sys.exit(1)
    
    # Find all config files
    config_files = find_config_files(args.config_dirs)
    
    if not config_files:
        print(f"No aggregator YAML config files found in: {args.config_dirs}")
        print("Looking for files matching pattern: aggregator_*.yaml")
        sys.exit(1)
    
    print("="*80)
    print("CONFIG PATH UPDATER")
    print("="*80)
    print(f"\nFound {len(config_files)} config file(s)")
    if args.dataset_path:
        print(f"New dataset path: {args.dataset_path}")
    if args.results_path:
        print(f"New results path: {args.results_path}")
    if args.dry_run:
        print("\n[DRY RUN MODE - No files will be modified]")
    print("\n" + "-"*80)
    
    # Update each file
    updated_count = 0
    error_count = 0
    
    # Auto-derive results path from dataset path unless explicitly disabled
    auto_derive = args.dataset_path is not None and args.results_path is None and not args.no_auto_derive
    
    for config_file in config_files:
        result = update_config_file(
            config_file,
            new_dataset_path=args.dataset_path,
            new_results_path=args.results_path,
            auto_derive_results=auto_derive,
            dry_run=args.dry_run
        )
        
        if result['updated']:
            updated_count += 1
            print(f"\n[OK] {os.path.basename(config_file)}")
            for change in result['changes']:
                print(f"  -> {change}")
        elif result['changes'] and 'Error' in result['changes'][0]:
            error_count += 1
            print(f"\n[ERROR] {os.path.basename(config_file)}")
            print(f"  Error: {result['changes'][0]}")
        else:
            # File didn't need updates (paths already match or fields don't exist)
            pass
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Files processed: {len(config_files)}")
    print(f"Files updated: {updated_count}")
    if error_count > 0:
        print(f"Errors: {error_count}")
    
    if args.dry_run:
        print("\nThis was a dry run. No files were modified.")
        print("Run without --dry-run to apply changes.")
    elif updated_count > 0:
        print(f"\n[SUCCESS] Successfully updated {updated_count} file(s)!")
    else:
        print("\nNo files needed updates (paths already match or fields don't exist).")


if __name__ == "__main__":
    main()

