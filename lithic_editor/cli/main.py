"""
Command line interface for the Lithic Editor and Annotator.

This module provides the main CLI entry point with argument parsing,
command dispatch, and comprehensive help as specified in PythonPackaging.md.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from lithic_editor import __version__
from lithic_editor.cli.help import show_help, show_api_help
from lithic_editor.processing import process_lithic_drawing
from lithic_editor.processing.upscaling import detect_image_dpi
from lithic_editor.gui import launch_gui


def create_parser():
    """
    Create the command line argument parser.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        prog='lithic-editor',
        description='Remove the ripple lines from a lithic drawing and add direction arrows.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  lithic-editor gui                                Start the graphical interface
  lithic-editor process image.png                  Process one image
  lithic-editor process image.png --debug          Process and write the debug images
  lithic-editor docs                               Open the documentation
  lithic-editor help                               Show the full help

Documentation: https://jasongellis.github.io/lithic-editor/
        """
    )
    
    # Version information
    parser.add_argument(
        '--version', 
        action='version', 
        version=f'Lithic Editor and Annotator v{__version__}'
    )
    
    # Main command subparsers
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # GUI command
    gui_parser = subparsers.add_parser(
        'gui', 
        help='Start the graphical interface'
    )
    
    # Process command
    process_parser = subparsers.add_parser(
        'process',
        help='Process one image'
    )
    process_parser.add_argument(
        'input_image',
        help='The lithic drawing to process'
    )
    process_parser.add_argument(
        '--output', '-o',
        default='image_debug',
        help='Output directory (default: image_debug). The result is written there as <name>_cleaned.png.'
    )
    process_parser.add_argument(
        '--debug',
        action='store_true',
        help='Write the debug images for each processing step'
    )
    process_parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Do not print processing messages'
    )
    
    # Upscaling parameters
    process_parser.add_argument(
        '--auto-upscale',
        action='store_true',
        help='Upscale for processing when the lines are too thin or too near to each other. '
             'The factor is measured from the drawing.'
    )
    process_parser.add_argument(
        '--default-dpi',
        type=int,
        metavar='DPI',
        help='The DPI to use when the image file has no DPI value'
    )
    process_parser.add_argument(
        '--upscale-model',
        choices=['espcn', 'fsrcnn'],
        default=None,
        help='The neural model for upscaling (default: the configuration value, espcn)'
    )
    process_parser.add_argument(
        '--config',
        metavar='PATH',
        help='A configuration file. Default: the file named by LITHIC_EDITOR_CONFIG, '
             'or the file shipped with the package.'
    )
    
    process_parser.add_argument(
        '--keep-upscaled',
        action='store_true',
        help='Keep the result at the upscaled size. The DPI value increases by the same factor. '
             'Without this option the result has the pixel size and DPI of the input.'
    )
    process_parser.add_argument(
        '--scale-image',
        metavar='PATH',
        help='The scale bar image scanned with the drawing. With --keep-upscaled it is scaled '
             'by the same factor and written as <name>_scale.png.'
    )

    # Cortex preservation parameters
    process_parser.add_argument(
        '--no-preserve-cortex',
        action='store_true',
        help='Process the cortex stipple as lines'
    )
    
    # Help command
    help_parser = subparsers.add_parser(
        'help',
        help='Show the full help'
    )
    help_parser.add_argument(
        'topic',
        nargs='?',
        choices=['api'],
        help='Show the help for one topic: api'
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
            # Handle upscaling parameters
            upscale_params = {
                'upscale_low_dpi': bool(getattr(args, 'auto_upscale', False)),
                'default_dpi': getattr(args, 'default_dpi', None),
                'upscale_model': getattr(args, 'upscale_model', None),
                'config': getattr(args, 'config', None),
            }
            
            keep_upscaled = bool(getattr(args, 'keep_upscaled', False))
            scale_image = getattr(args, 'scale_image', None)
            result = process_lithic_drawing(
                image_path=str(input_path),
                output_folder=args.output,
                save_debug=args.debug,
                preserve_cortex=not args.no_preserve_cortex,  # Default True, inverted flag
                restore_original_size=not keep_upscaled,
                scale_image_path=scale_image,
                return_scale_factor=True,
                **upscale_params
            )
            
            # Restore print if it was suppressed
            if args.quiet:
                builtins.print = original_print
            
            # Save the cleaned drawing with a truthful DPI tag, never a guessed one
            output_file, scale_file, factor = save_cli_result(result, input_path, Path(args.output))
            if not args.quiet:
                print("Processing complete.")
                print(f"Result written to: {output_file}")
                if factor > 1:
                    print(f"The result is {factor}x the input size. The DPI value is {factor}x the input DPI.")
                    if scale_file:
                        print(f"Scale image scaled {factor}x and written to: {scale_file}")
                    else:
                        print(f"No scale image was given. Scale the scale bar image {factor}x before you measure.")
                
                if args.debug:
                    print(f"Debug images written to: {args.output}")
            
            return 0
            
        except Exception as e:
            # Restore print if it was suppressed
            if args.quiet:
                builtins.print = original_print
            
            print(f"Processing failed: {e}")
            return 1
            
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return 1
    except KeyboardInterrupt:
        print("\nProcessing stopped by the user")
        return 1


def save_cli_result(result, input_path: Path, output_dir: Path):
    """
    Write the processed image as ``<stem>_cleaned.png`` and return
    ``(output_file, scale_file, factor)``.

    ``result`` is the pipeline's return value: an image, or the dict from
    ``return_scale_factor=True``. The DPI tag is the input's, raised by the
    factor when the result was kept at the working size; an input with no DPI
    tag produces an output with none. A scale image in the result is saved as
    ``<stem>_scale.png`` next to it, so the pair keeps one pixel scale.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(result, dict):
        image = np.asarray(result['processed_image'])
        factor = int(result.get('scale_factor', 1))
        scale_image = result.get('processed_scale')
    else:
        image, factor, scale_image = np.asarray(result), 1, None

    dpi = detect_image_dpi(str(input_path))
    save_kwargs = {'dpi': (dpi * factor, dpi * factor)} if dpi else {}
    output_file = output_dir / f"{input_path.stem}_cleaned.png"
    Image.fromarray(image).save(output_file, **save_kwargs)

    scale_file = None
    if scale_image is not None and factor > 1:
        scale_file = output_dir / f"{input_path.stem}_scale.png"
        Image.fromarray(np.asarray(scale_image)).save(scale_file, **save_kwargs)
    return output_file, scale_file, factor


def launch_gui_cli():
    """
    Launch GUI via command line interface.
    
    Returns:
        int: Exit code from GUI application
    """
    try:
        return launch_gui()
    except Exception as e:
        print(f"The graphical interface did not start: {e}")
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
        print(f"Documentation opened in the web browser: {docs_url}")
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
        print("\nStopped by the user")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())