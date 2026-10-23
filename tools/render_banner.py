"""
Write the README banner as SVG images.

The banner shows the Lithic Editor logo, the version, a tagline and the commands
to start with, in the style of the PyLithics welcome screen. GitHub shows one
image in dark mode and another in light mode, so two images are written from
the same code. Run this after a change to the banner or to the version::

    python tools/render_banner.py

Output: ``docs/assets/images/banner-dark.svg`` and
``docs/assets/images/banner-light.svg``.
"""

import os
import sys

from rich.align import Align
from rich.console import CONSOLE_SVG_FORMAT, Console, Group
from rich.panel import Panel
from rich.terminal_theme import SVG_EXPORT_THEME, TerminalTheme
from rich.text import Text

from lithic_editor import __version__

WIDTH = 100
ASSETS = os.path.join('docs', 'assets', 'images')
TAGLINE = "Ripple removal and arrow annotation for lithic drawings"

# ANSI Shadow glyphs, the face of the PyLithics logo. Each glyph is six rows of
# equal width and includes its own trailing space where the face has one.
GLYPHS = {
    'L': ["██╗     ", "██║     ", "██║     ", "██║     ", "███████╗", "╚══════╝"],
    'I': ["██╗", "██║", "██║", "██║", "██║", "╚═╝"],
    'T': ["████████╗", "╚══██╔══╝", "   ██║   ", "   ██║   ", "   ██║   ", "   ╚═╝   "],
    'H': ["██╗  ██╗", "██║  ██║", "███████║", "██╔══██║", "██║  ██║", "╚═╝  ╚═╝"],
    'C': [" ██████╗", "██╔════╝", "██║     ", "██║     ", "╚██████╗", " ╚═════╝"],
    'E': ["███████╗", "██╔════╝", "█████╗  ", "██╔══╝  ", "███████╗", "╚══════╝"],
    'D': ["██████╗ ", "██╔══██╗", "██║  ██║", "██║  ██║", "██████╔╝", "╚═════╝ "],
    'O': [" ██████╗ ", "██╔═══██╗", "██║   ██║", "██║   ██║", "╚██████╔╝", " ╚═════╝ "],
    'R': ["██████╗ ", "██╔══██╗", "██████╔╝", "██╔══██╗", "██║  ██║", "╚═╝  ╚═╝"],
}


def block_word(word: str) -> list[str]:
    """The word in block letters: six rows joined from the glyphs."""
    return ["".join(GLYPHS[letter][row] for letter in word) for row in range(6)]


def block_lines(*words: str) -> str:
    """The words in block letters, one under the other, each centred on the widest."""
    rows = [row for word in words for row in block_word(word)]
    width = max(len(row) for row in rows)
    return "\n".join(row.center(width) for row in rows)


LOGO = block_lines("LITHIC", "EDITOR")

START = [
    ("Start the graphical interface", "lithic-editor --gui"),
    ("Process one drawing", "lithic-editor drawing.png --output results"),
    ("Measure the striking angle of each scar", "lithic-editor drawing.png --angles"),
    ("Write the debug images", "lithic-editor drawing.png --debug"),
    ("Help and documentation", "lithic-editor --help    lithic-editor --docs"),
    ("GitHub", "github.com/JasonGellis/lithic-editor"),
]


def print_banner(console: Console) -> None:
    """Print the banner to ``console``."""
    panel_width = min(WIDTH, console.width)
    hero = Panel(
        Group(
            Align.center(Text(LOGO, style="bold cyan")),
            Text(""),
            Align.center(Text(TAGLINE, style="italic")),
        ),
        title=f"[italic]Lithic Editor[/]  v{__version__}",
        border_style="cyan",
        padding=(1, 4),
        width=panel_width,
    )
    parts = []
    for heading, command in START:
        parts.append((f"▶ {heading}\n", "bold cyan"))
        parts.append((f"  {command}\n\n", "white"))
    parts[-1] = (parts[-1][0].rstrip("\n"), "white")
    body = Panel(
        Text.assemble(*parts), title="[bold]Start[/]",
        border_style="cyan", padding=(1, 2), width=panel_width,
    )
    console.print(Align.center(hero))
    console.print(Align.center(body))
    console.print()


# The banner styles its text "cyan" and "white". On a light page those read as
# pale, so the light theme maps them to darker inks. The 16 colours are the ANSI
# palette: black, red, green, yellow, blue, magenta, cyan, white, then the same
# eight in their bright form.
LIGHT_THEME = TerminalTheme(
    (255, 255, 255),
    (36, 41, 46),
    [
        (36, 41, 46), (203, 36, 49), (34, 134, 58), (176, 136, 0),
        (0, 92, 197), (111, 66, 193), (0, 117, 140), (60, 66, 72),
    ],
    [
        (96, 103, 112), (203, 36, 49), (34, 134, 58), (176, 136, 0),
        (0, 92, 197), (111, 66, 193), (0, 117, 140), (60, 66, 72),
    ],
)

THEMES = {
    'dark': SVG_EXPORT_THEME,
    'light': LIGHT_THEME,
}


def panel_format(theme: TerminalTheme) -> str:
    """Rich's SVG template with a plain rounded panel in place of the terminal window.

    The default template draws a title bar with three window buttons. The banner is
    a panel on a page, not a window, so the chrome is replaced by one rounded
    rectangle in the theme's background colour.
    """
    panel = (
        f'<rect fill="{theme.background_color.hex}" x="0" y="0" '
        'width="{width}" height="{height}" rx="12"/>'
    )
    return CONSOLE_SVG_FORMAT.replace("{chrome}", panel)


def render(mode: str, theme: TerminalTheme, directory: str) -> str:
    """Write the banner as ``banner-<mode>.svg`` and return the path."""
    with open(os.devnull, 'w') as sink:
        console = Console(
            record=True, width=WIDTH, force_terminal=True,
            color_system='truecolor', file=sink,
        )
        print_banner(console)
        path = os.path.join(directory, f'banner-{mode}.svg')
        console.save_svg(path, title='', theme=theme, code_format=panel_format(theme))
    return path


def main(directory: str = ASSETS) -> int:
    """Render both modes into ``directory``."""
    os.makedirs(directory, exist_ok=True)
    for mode, theme in THEMES.items():
        print(render(mode, theme, directory))
    return 0


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:]))
