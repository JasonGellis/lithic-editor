"""
Configuration for the processing pipeline.

Every tunable number lives in ``config.yaml`` next to this module. ``load_config``
reads that file, or a user's copy of it, into frozen dataclasses. The dataclass
defaults and the shipped file hold the same values; a test checks that.

Precedence for the file: the ``path`` argument, then the ``LITHIC_EDITOR_CONFIG``
environment variable, then the shipped default. A user file may hold only the
keys it changes. Unknown sections or keys raise ``ConfigError``.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

ENV_VAR = "LITHIC_EDITOR_CONFIG"
DEFAULT_FILENAME = "config.yaml"
SUPPORTED_MODELS = ("espcn", "fsrcnn")


class ConfigError(ValueError):
    """The configuration file is missing, malformed, or holds an unknown or invalid value."""


@dataclass(frozen=True)
class ResolutionSettings:
    min_line_width_px: float = 6.0
    min_hatch_gap_px: float = 12.0
    max_upscale_factor: int = 4
    upscale_model: str = "espcn"
    dot_extent_in_line_widths: float = 4.0
    restore_ink_coverage: float = 0.35


@dataclass(frozen=True)
class SmoothingSettings:
    enabled: bool = True
    sigma_in_line_widths: float = 0.25


@dataclass(frozen=True)
class ThresholdSettings:
    sauvola_k: float = 0.2
    window_high_dpi: int = 51
    window_medium_dpi: int = 25
    window_low_dpi: int = 15

    def window_for_dpi(self, dpi: float | None) -> int:
        """Sauvola window for a working DPI, always odd."""
        if dpi and dpi >= 600:
            window = self.window_high_dpi
        elif dpi and dpi >= 300:
            window = self.window_medium_dpi
        else:
            window = self.window_low_dpi
        return window if window % 2 == 1 else window + 1


@dataclass(frozen=True)
class CortexSettings:
    reference_dpi: float = 150.0
    max_area_px: int = 60
    min_area_px: int = 3
    max_area_floor_px: int = 30
    min_area_floor_px: int = 2

    def area_limits_for_dpi(self, dpi: float | None) -> tuple[int, int]:
        """``(min_area, max_area)`` in pixels for a working DPI."""
        if not dpi:
            return self.min_area_px, self.max_area_px
        scale = (dpi / self.reference_dpi) ** 2
        return (
            max(self.min_area_floor_px, int(self.min_area_px * scale)),
            max(self.max_area_floor_px, int(self.max_area_px * scale)),
        )


@dataclass(frozen=True)
class SkeletonSettings:
    spur_length_px: int = 5
    bridge_in_line_widths: float = 1.5
    bridge_max_hatch_gap_fraction: float = 0.5
    bridge_min_px: float = 3.0

    def bridge_distance(self, line_width: float | None, hatch_gap: float | None) -> float:
        """Largest gap between two free ends that is bridged, in pixels."""
        if line_width is None:
            return self.bridge_min_px
        distance = max(self.bridge_min_px, self.bridge_in_line_widths * line_width)
        if hatch_gap is not None:
            distance = min(distance, self.bridge_max_hatch_gap_fraction * hatch_gap)
        return distance


@dataclass(frozen=True)
class RippleSettings:
    y_tip_distance_near_300_px: int = 5
    y_tip_distance_medium_px: int = 3
    y_tip_distance_low_px: int = 2

    def y_tip_distance_for_dpi(self, dpi: float | None) -> int:
        """Junction-to-endpoint distance below which a junction becomes an endpoint."""
        if dpi and abs(dpi - 300) < 50:
            return self.y_tip_distance_near_300_px
        if dpi and dpi >= 150:
            return self.y_tip_distance_medium_px
        if dpi:
            return self.y_tip_distance_low_px
        return self.y_tip_distance_near_300_px


@dataclass(frozen=True)
class LineSettings:
    radius_in_line_widths: float = 0.5
    radius_max_in_line_widths: float = 0.75

    def radius_range(self, line_width: float) -> tuple[int, int]:
        """``(min_radius, max_radius)`` in pixels for rebuilding lines."""
        min_radius = max(1, round(self.radius_in_line_widths * line_width))
        max_radius = max(min_radius, round(self.radius_max_in_line_widths * line_width))
        return min_radius, max_radius


@dataclass(frozen=True)
class Config:
    resolution: ResolutionSettings = field(default_factory=ResolutionSettings)
    smoothing: SmoothingSettings = field(default_factory=SmoothingSettings)
    threshold: ThresholdSettings = field(default_factory=ThresholdSettings)
    cortex: CortexSettings = field(default_factory=CortexSettings)
    skeleton: SkeletonSettings = field(default_factory=SkeletonSettings)
    ripples: RippleSettings = field(default_factory=RippleSettings)
    lines: LineSettings = field(default_factory=LineSettings)
    source: str = "defaults"


_SECTIONS = {
    "resolution": ResolutionSettings,
    "smoothing": SmoothingSettings,
    "threshold": ThresholdSettings,
    "cortex": CortexSettings,
    "skeleton": SkeletonSettings,
    "ripples": RippleSettings,
    "lines": LineSettings,
}


def default_config_path() -> Path:
    """Path of the configuration file shipped with the package."""
    return Path(__file__).resolve().parent / DEFAULT_FILENAME


def resolve_config_path(path: str | os.PathLike | None = None) -> Path:
    """The file ``load_config`` reads: the argument, the environment variable, or the default."""
    if path:
        return Path(path)
    from_env = os.environ.get(ENV_VAR)
    if from_env:
        return Path(from_env)
    return default_config_path()


def load_config(path: str | os.PathLike | None = None) -> Config:
    """
    Read a configuration file into a ``Config``.

    Keys the file does not set keep their defaults. Unknown keys, wrong types
    and out-of-range values raise ``ConfigError`` that names the key.
    """
    file_path = resolve_config_path(path)
    try:
        with open(file_path, encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except FileNotFoundError as exc:
        raise ConfigError(f"Configuration file not found: {file_path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"Configuration file is not valid YAML: {file_path}: {exc}") from exc
    return config_from_mapping(data, source=str(file_path))


def config_from_mapping(data: Any, source: str = "mapping") -> Config:
    """Build a ``Config`` from a nested mapping such as parsed YAML."""
    if not isinstance(data, dict):
        raise ConfigError(f"{source}: the top level must be a mapping of sections")
    unknown = set(data) - set(_SECTIONS)
    if unknown:
        raise ConfigError(f"{source}: unknown section(s): {', '.join(sorted(unknown))}")
    sections = {
        name: _build_section(cls, data.get(name) or {}, name, source)
        for name, cls in _SECTIONS.items()
    }
    config = Config(**sections, source=source)
    _validate(config, source)
    return config


def _build_section(cls, mapping: Any, section: str, source: str):
    if not isinstance(mapping, dict):
        raise ConfigError(f"{source}: section '{section}' must be a mapping")
    known = {f.name: f for f in fields(cls)}
    unknown = set(mapping) - set(known)
    if unknown:
        raise ConfigError(f"{source}: unknown key(s) in '{section}': {', '.join(sorted(unknown))}")
    values = {}
    for key, raw in mapping.items():
        values[key] = _coerce(raw, known[key].type, f"{section}.{key}", source)
    return cls(**values)


def _coerce(value: Any, annotation: str, key: str, source: str):
    kind = annotation if isinstance(annotation, str) else annotation.__name__
    try:
        if kind == "bool":
            if isinstance(value, bool):
                return value
            raise TypeError
        if kind == "int":
            if isinstance(value, bool) or int(value) != value:
                raise TypeError
            return int(value)
        if kind == "float":
            if isinstance(value, bool):
                raise TypeError
            return float(value)
        if kind == "str":
            return str(value)
    except (TypeError, ValueError):
        pass
    raise ConfigError(f"{source}: '{key}' must be a {kind}, got {value!r}")


def _validate(config: Config, source: str) -> None:
    r = config.resolution
    checks = [
        (r.min_line_width_px > 0, "resolution.min_line_width_px must be positive"),
        (r.min_hatch_gap_px > 0, "resolution.min_hatch_gap_px must be positive"),
        (1 <= r.max_upscale_factor <= 4, "resolution.max_upscale_factor must be 1 to 4"),
        (r.upscale_model in SUPPORTED_MODELS, f"resolution.upscale_model must be one of {SUPPORTED_MODELS}"),
        (r.dot_extent_in_line_widths >= 0, "resolution.dot_extent_in_line_widths must not be negative"),
        (0 < r.restore_ink_coverage <= 1, "resolution.restore_ink_coverage must be in (0, 1]"),
        (config.smoothing.sigma_in_line_widths >= 0, "smoothing.sigma_in_line_widths must not be negative"),
        (0 < config.threshold.sauvola_k < 1, "threshold.sauvola_k must be in (0, 1)"),
        (min(config.threshold.window_high_dpi, config.threshold.window_medium_dpi, config.threshold.window_low_dpi) >= 3,
         "threshold windows must be at least 3"),
        (config.cortex.reference_dpi > 0, "cortex.reference_dpi must be positive"),
        (0 < config.cortex.min_area_px <= config.cortex.max_area_px, "cortex.min_area_px must be positive and not above max_area_px"),
        (config.skeleton.spur_length_px >= 0, "skeleton.spur_length_px must not be negative"),
        (config.skeleton.bridge_in_line_widths >= 0, "skeleton.bridge_in_line_widths must not be negative"),
        (0 < config.skeleton.bridge_max_hatch_gap_fraction <= 1, "skeleton.bridge_max_hatch_gap_fraction must be in (0, 1]"),
        (config.skeleton.bridge_min_px >= 0, "skeleton.bridge_min_px must not be negative"),
        (config.lines.radius_in_line_widths > 0, "lines.radius_in_line_widths must be positive"),
        (config.lines.radius_max_in_line_widths >= config.lines.radius_in_line_widths,
         "lines.radius_max_in_line_widths must not be below radius_in_line_widths"),
    ]
    for ok, message in checks:
        if not ok:
            raise ConfigError(f"{source}: {message}")


def resolve_config(config: Config | str | os.PathLike | None) -> Config:
    """Accept a ``Config``, a path, or None (environment variable or default) and return a ``Config``."""
    if isinstance(config, Config):
        return config
    return load_config(config)


def to_mapping(config: Config) -> dict:
    """The configuration as a nested dict, without the ``source`` entry."""
    data = dataclasses.asdict(config)
    data.pop("source", None)
    return data


__all__ = [
    "Config", "ConfigError", "ResolutionSettings", "SmoothingSettings", "ThresholdSettings",
    "CortexSettings", "SkeletonSettings", "RippleSettings", "LineSettings",
    "load_config", "resolve_config", "config_from_mapping", "default_config_path",
    "resolve_config_path", "to_mapping", "ENV_VAR",
]
