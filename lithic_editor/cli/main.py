"""
Command line interface for the Lithic Editor and Annotator.

This module provides the main CLI entry point with argument parsing,
command dispatch, and comprehensive help as specified in PythonPackaging.md.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from lithic_editor import __version__
from lithic_editor.cli.help import show_help, show_api_help
from lithic_editor.config import load_config, replace_values
from lithic_editor.processing import process_lithic_drawing
from lithic_editor.processing.upscaling import detect_image_dpi
from lithic_editor.gui import launch_gui


class _FullHelp(argparse.Action):
    """``-h`` and ``--help``: print the full help text and exit."""

    def __init__(self, option_strings, dest=argparse.SUPPRESS, default=argparse.SUPPRESS, help=None):
        super().__init__(option_strings, dest, nargs=0, default=default, help=help)

    def __call__(self, parser, namespace, values, option_string=None):
        show_help()
        parser.exit(0)


class _ApiHelp(_FullHelp):
    """``--help-api``: print the Python API help text and exit."""

    def __call__(self, parser, namespace, values, option_string=None):
        show_api_help()
        parser.exit(0)


def create_parser():
    """
    Create the command line argument parser.

    Every action is a flag. The drawing to process is the one positional argument.

    Returns:
        argparse.ArgumentParser: Configured argument parser
    """
    parser = argparse.ArgumentParser(
        prog='lithic-editor',
        description='Remove the ripple lines from a lithic drawing and add direction arrows.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
        epilog="""
Examples:
  lithic-editor --gui                              Start the graphical interface
  lithic-editor drawing.png                        Process one drawing
  lithic-editor drawing.png --debug                Process and write the debug images
  lithic-editor --docs                             Open the documentation
  lithic-editor --help                             Show the full help

Documentation: https://jasongellis.github.io/lithic-editor/
        """
    )

    # Actions
    parser.add_argument('-h', '--help', action=_FullHelp, help='Show the full help and exit')
    parser.add_argument('--help-api', action=_ApiHelp, help='Show the Python API help and exit')
    parser.add_argument(
        '--version',
        action='version',
        version=f'Lithic Editor and Annotator v{__version__}'
    )
    parser.add_argument(
        '--gui',
        action='store_true',
        help='Start the graphical interface'
    )
    parser.add_argument(
        '--docs',
        action='store_true',
        help='Open the documentation in the web browser'
    )
    parser.add_argument(
        '--offline',
        action='store_true',
        help='With --docs: serve the documentation on this computer'
    )

    # The drawing to process and the processing options
    parser.add_argument(
        'input_image',
        nargs='?',
        metavar='INPUT',
        help='The lithic drawing to process'
    )
    parser.add_argument(
        '--output', '-o',
        default='image_debug',
        metavar='DIR',
        help='Output directory (default: image_debug). The result is written there as <name>_cleaned.png.'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Write the debug images for each processing step'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Do not print processing messages'
    )
    parser.add_argument(
        '--auto-upscale',
        action='store_true',
        help='Upscale for processing when the lines are too thin or too near to each other. '
             'The factor is measured from the drawing.'
    )
    parser.add_argument(
        '--default-dpi',
        type=int,
        metavar='DPI',
        help='The DPI to use when the image file has no DPI value'
    )
    parser.add_argument(
        '--upscale-model',
        choices=['espcn', 'fsrcnn'],
        default=None,
        help='The neural model for upscaling (default: the configuration value, espcn)'
    )
    parser.add_argument(
        '--config',
        metavar='PATH',
        help='A configuration file. Default: the file named by LITHIC_EDITOR_CONFIG, '
             'or the file shipped with the package.'
    )
    parser.add_argument(
        '--keep-upscaled',
        action='store_true',
        help='Keep the result at the upscaled size. The DPI value increases by the same factor. '
             'Without this option the result has the pixel size and DPI of the input.'
    )
    parser.add_argument(
        '--scale-image',
        metavar='PATH',
        help='The scale bar image scanned with the drawing. With --keep-upscaled it is scaled '
             'by the same factor and written as <name>_scale.png.'
    )
    parser.add_argument(
        '--angles',
        action='store_true',
        help='Measure the striking angle of each scar from its ripples. The angles are '
             'written as <name>_angles.csv.'
    )
    parser.add_argument(
        '--min-ripples',
        type=int,
        metavar='N',
        default=None,
        help='A scar with fewer ripples than N gets no angle (default: the configuration value, 4)'
    )
    parser.add_argument(
        '--min-arc-depth',
        type=float,
        metavar='PX',
        default=None,
        help='When the mean arc depth of the ripples of a scar is less than PX pixels, the direction is marked '
             'uncertain (default: the configuration value, 2.0)'
    )
    parser.add_argument(
        '--max-cortex-dots',
        type=int,
        metavar='N',
        default=None,
        help='A face with more than N cortex dots is cortex and gets no angle '
             '(default: the configuration value, 2)'
    )
    parser.add_argument(
        '--no-preserve-cortex',
        action='store_true',
        help='Process the cortex stipple as lines'
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
                'config': config_with_angle_options(args),
            }
            
            keep_upscaled = bool(getattr(args, 'keep_upscaled', False))
            scale_image = getattr(args, 'scale_image', None)
            measure_angles = bool(getattr(args, 'angles', False))
            result = process_lithic_drawing(
                image_path=str(input_path),
                output_folder=args.output,
                save_debug=args.debug,
                preserve_cortex=not args.no_preserve_cortex,  # Default True, inverted flag
                restore_original_size=not keep_upscaled,
                scale_image_path=scale_image,
                return_scale_factor=True,
                measure_angles=measure_angles,
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

            if measure_angles and isinstance(result, dict):
                records = result.get('striking_angles', [])
                angles_file = save_angles_csv(records, input_path, Path(args.output))
                if not args.quiet:
                    for record in records:
                        note = "" if record['direction_certain'] else " (direction not certain)"
                        print(f"Scar {record['scar']}: {record['compass']} degrees from "
                              f"{record['ripples']} ripples at ({record['x']:.0f}, {record['y']:.0f}){note}")
                    print(f"Striking angles written to: {angles_file}")

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


ANGLE_COLUMNS = ('scar', 'x', 'y', 'compass', 'ripples', 'spread', 'direction_certain', 'extent')


ANGLE_OPTIONS = {
    'min_ripples': 'min_ripples',
    'min_arc_depth': 'min_arc_depth_px',
    'max_cortex_dots': 'max_cortex_dots',
}


def config_with_angle_options(args):
    """
    The configuration for the pipeline: the ``--config`` path, or a ``Config`` with
    the ``angles`` values from the command line applied over that file.

    Returns the path (or None) unchanged when no angle option is given, so the
    pipeline resolves the configuration as usual.
    """
    overrides = {}
    for option, key in ANGLE_OPTIONS.items():
        value = getattr(args, option, None)
        if value is not None:
            overrides[key] = value
    config_path = getattr(args, 'config', None)
    if not overrides:
        return config_path
    base = load_config(config_path)
    return replace_values(base, 'angles', source=f"{base.source} with command line options", **overrides)


def save_angles_csv(records, input_path: Path, output_dir: Path):
    """
    Write the striking angles as ``<stem>_angles.csv`` and return its path.

    One row per scar: ``scar``, ``x``, ``y`` (the centroid of its ripples in
    pixels of the result), ``compass`` (the direction of the force, away from the
    point of percussion, in the PyLithics ``arrow_angle`` frame), ``ripples``, ``spread``,
    ``direction_certain`` and ``extent``. The file holds only the header when no scar had
    enough ripples.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    angles_file = output_dir / f"{input_path.stem}_angles.csv"
    with open(angles_file, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=ANGLE_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record[key] for key in ANGLE_COLUMNS})
    return angles_file


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
        
        if args.docs:
            return open_docs(args)
        if args.gui:
            return launch_gui_cli()
        if args.input_image:
            return process_image_cli(args)
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