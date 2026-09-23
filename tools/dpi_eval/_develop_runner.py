"""
Run the ``develop`` branch's pipeline in an isolated process.

Usage: python _develop_runner.py VARIANT_PNG DPI WORKDIR OUTPUT_PNG WORKTREE

The worktree is put first on ``sys.path`` so its ``lithic_editor`` shadows the
installed one, and the import is checked before anything runs.
"""

import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    variant_path, dpi, workdir, output_path, worktree = argv[1:6]
    worktree_dir = Path(worktree).resolve()
    sys.path.insert(0, str(worktree_dir))

    import lithic_editor

    loaded_from = Path(lithic_editor.__file__).resolve()
    if worktree_dir not in loaded_from.parents:
        print(f"lithic_editor was imported from {loaded_from}, not from {worktree_dir}",
              file=sys.stderr)
        return 2

    import numpy as np
    from PIL import Image
    from lithic_editor.processing import process_lithic_drawing

    image = np.array(Image.open(variant_path).convert("L"))
    result = process_lithic_drawing(
        image,
        output_folder=workdir,
        dpi_info=int(dpi),
        save_debug=False,
        preserve_cortex=True,
    )
    Image.fromarray(result).save(output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
