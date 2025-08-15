"""
Command line interface for the Lithic Editor and Annotator.

This module provides the main CLI entry point with argument parsing,
command dispatch, and comprehensive help as specified in PythonPackaging.md.
"""

import argparse
import sys
import os
from pathlib import Path

from lithic_editor import __version__
from lithic_editor.cli.help import show_help, show_version, show_api_help
from lithic_editor.processing import process_lithic_drawing_improved
from lithic_editor.gui import launch_gui


def create_parser():
    """
    Create the command line argument parser.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        prog='lithic-editor',
        description='Lithic Editor and Annotator - Archaeological image processing tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  lithic-editor --gui                              # Launch GUI
  lithic-editor process image.png                  # Process image
  lithic-editor process image.png --debug          # Process with debug output
  lithic-editor docs                               # View full documentation
  lithic-editor --help                             # Show comprehensive help

📚 DOCUMENTATION:
  Online: https://jasongellis.github.io/lithic-editor/
  Local:  lithic-editor docs --offline
        """
    )
    
    # Version information
    parser.add_argument(
        '--version', 
        action='version', 
        version=f'Lithic Editor and Annotator v{__version__}'
    )
    
    # Main command subparsers
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # GUI command
    gui_parser = subparsers.add_parser(
        'gui', 
        help='Launch the graphical user interface'
    )
    
    # Process command
    process_parser = subparsers.add_parser(
        'process',
        help='Process lithic images via command line'
    )
    process_parser.add_argument(
        'input_image',
        help='Path to input lithic drawing image'
    )
    process_parser.add_argument(
        '--output', '-o',
        default='image_debug',
        help='Output directory (default: image_debug)'
    )
    process_parser.add_argument(
        '--debug',
        action='store_true',
        help='Save debug images showing processing steps'
    )
    process_parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress processing output'
    )
    
    # Help command
    help_parser = subparsers.add_parser(
        'help',
        help='Show detailed help information'
    )
    help_parser.add_argument(
        'topic',
        nargs='?',
        choices=['api'],
        help='Show help for specific topic'
    )
    
    # Docs command - opens documentation
    docs_parser = subparsers.add_parser(
        'docs',
        help='Open documentation in browser'
    )
    docs_parser.add_argument(
        '--offline',
        action='store_true',
        help='Serve documentation locally (requires mkdocs)'
    )
    
    # Add top-level flags for convenience
    parser.add_argument(
        '--gui',
        action='store_true',
        help='Launch GUI (equivalent to "lithic-editor gui")'
    )
    
    return parser


def validate_input_file(file_path):
    """
    Validate input image file.
    
    Args:
        file_path (str): Path to input file
        
    Returns:
        Path: Validated path object
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format not supported
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    if not path.is_file():
        raise ValueError(f"Input path is not a file: {file_path}")
    
    # Check file extension
    supported_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp'}
    if path.suffix.lower() not in supported_extensions:
        raise ValueError(
            f"Unsupported file format: {path.suffix}\\n"
            f"Supported formats: {', '.join(supported_extensions)}"
        )
    
    return path


def process_image_cli(args):
    """
    Process image via command line interface.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    try:
        # Validate input file
        input_path = validate_input_file(args.input_image)
        
        if not args.quiet:
            print(f"Processing: {input_path}")
            print(f"Output directory: {args.output}")
        
        # Suppress print output if quiet mode
        if args.quiet:
            import builtins
            original_print = builtins.print
            builtins.print = lambda *args, **kwargs: None
        
        try:
            # Process the image using the exact working algorithm
            result = process_lithic_drawing_improved(
                image_path=str(input_path),
                output_folder=args.output,
                save_debug=args.debug
            )
            
            # Restore print if it was suppressed
            if args.quiet:
                builtins.print = original_print
            
            # Report success
            output_file = Path(args.output) / "9_high_quality.png"
            if not args.quiet:
                print(f"✓ Processing completed successfully!")
                print(f"✓ Output saved to: {output_file}")
                
                if args.debug:
                    print(f"✓ Debug images saved to: {args.output}")
            
            return 0
            
        except Exception as e:
            # Restore print if it was suppressed
            if args.quiet:
                builtins.print = original_print
            
            print(f"✗ Processing failed: {e}")
            return 1
            
    except (FileNotFoundError, ValueError) as e:
        print(f"✗ Error: {e}")
        return 1
    except KeyboardInterrupt:
        print("\\n✗ Processing interrupted by user")
        return 1


def launch_gui_cli():
    """
    Launch GUI via command line interface.
    
    Returns:
        int: Exit code from GUI application
    """
    try:
        return launch_gui()
    except Exception as e:
        print(f"✗ Failed to launch GUI: {e}")
        return 1


def show_help_cli(args):
    """
    Show help information based on topic.
    
    Args:
        args: Parsed command line arguments
    """
    if args.topic == 'api':
        show_api_help()
    else:
        show_help()


def open_docs(args):
    """
    Open documentation in browser or serve locally.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        int: Exit code
    """
    import webbrowser
    
    if args.offline:
        # Use built-in documentation server (no mkdocs required)
        from lithic_editor.cli.docs_server import serve_docs
        return serve_docs()
    else:
        # Open online documentation
        docs_url = "https://jasongellis.github.io/lithic-editor/"
        print(f"Opening documentation in browser: {docs_url}")
        webbrowser.open(docs_url)
        return 0


def main():
    """
    Main CLI entry point.
    
    This function handles argument parsing, command dispatch, and error handling
    for the lithic editor command line interface as specified in PythonPackaging.md.
    
    Returns:
        int: Exit code (0 for success, non-zero for error)
    """
    parser = create_parser()
    
    # Handle case with no arguments - show comprehensive help
    if len(sys.argv) == 1:
        show_help()
        return 0
    
    try:
        args = parser.parse_args()
        
        # Handle top-level GUI flag
        if args.gui:
            return launch_gui_cli()
        
        # Dispatch based on command
        if args.command == 'gui':
            return launch_gui_cli()
        elif args.command == 'process':
            return process_image_cli(args)
        elif args.command == 'help':
            show_help_cli(args)
            return 0
        elif args.command == 'docs':
            return open_docs(args)
        else:
            # No command specified, show help
            show_help()
            return 0
            
    except KeyboardInterrupt:
        print("\\n✗ Interrupted by user")
        return 1
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())