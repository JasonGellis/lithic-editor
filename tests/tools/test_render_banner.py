"""The README banner tool: glyphs, logo and the rendered content."""

import pytest

rich = pytest.importorskip("rich")

from rich.console import Console  # noqa: E402

from lithic_editor import __version__  # noqa: E402
from tools.render_banner import GLYPHS, LOGO, START, TAGLINE, block_word, print_banner, render  # noqa: E402


def test_every_glyph_is_six_rows_of_one_width():
    for letter, rows in GLYPHS.items():
        assert len(rows) == 6, letter
        assert len({len(row) for row in rows}) == 1, letter


def test_logo_rows_share_one_width():
    rows = LOGO.split("\n")
    assert len(rows) == 12
    assert len({len(row) for row in rows}) == 1
    assert block_word("LITHIC")[0].strip() in rows[0]


def test_banner_holds_the_version_tagline_and_commands():
    console = Console(record=True, width=100, force_terminal=False, file=open("/dev/null", "w"))
    print_banner(console)
    text = console.export_text()
    assert f"v{__version__}" in text and TAGLINE in text
    for heading, command in START:
        assert heading in text and command in text


def test_render_writes_an_svg(tmp_path):
    from tools.render_banner import THEMES
    path = render("dark", THEMES["dark"], str(tmp_path))
    assert path.endswith("banner-dark.svg")
    assert "<svg" in open(path).read()


def test_the_svg_is_a_panel_without_window_chrome(tmp_path):
    from tools.render_banner import THEMES
    svg = open(render("light", THEMES["light"], str(tmp_path))).read()
    assert "<circle" not in svg
    assert 'rx="12"' in svg
