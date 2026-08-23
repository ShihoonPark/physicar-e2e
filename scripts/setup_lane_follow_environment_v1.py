#!/usr/bin/env python3
"""Recreate and verify the cone-free Lane-Follow Environment V1 assets."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from xml.parsers import expat


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "lane_follow_environment_v1.json"


class SetupError(Exception):
    """A controlled environment setup or verification failure."""


@dataclass(frozen=True)
class EnvironmentConfig:
    canonical_world: str
    derived_world: str
    removed_models: tuple[str, ...]


@dataclass(frozen=True)
class AssetPaths:
    world: Path
    route: Path
    model: Path


def load_config(path: Path) -> EnvironmentConfig:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SetupError(f"cannot load configuration {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SetupError("configuration root must be a JSON object")
    required = {"canonical_world", "derived_world", "removed_models"}
    unknown = set(payload) - required
    missing = required - set(payload)
    if missing:
        raise SetupError(f"configuration missing fields: {', '.join(sorted(missing))}")
    if unknown:
        raise SetupError(f"configuration has unknown fields: {', '.join(sorted(unknown))}")
    canonical = payload["canonical_world"]
    derived = payload["derived_world"]
    removed = payload["removed_models"]
    if not isinstance(canonical, str) or not canonical:
        raise SetupError("canonical_world must be a non-empty string")
    if not isinstance(derived, str) or not derived:
        raise SetupError("derived_world must be a non-empty string")
    if canonical == derived:
        raise SetupError("canonical_world and derived_world must differ")
    if any(Path(value).name != value for value in (canonical, derived)):
        raise SetupError("world names must be basenames, not paths")
    if not isinstance(removed, list) or not removed or not all(
        isinstance(item, str) and item for item in removed
    ):
        raise SetupError("removed_models must be a non-empty list of model names")
    if len(set(removed)) != len(removed):
        raise SetupError("removed_models must not contain duplicates")
    return EnvironmentConfig(canonical, derived, tuple(removed))


def share_path(sim_root: Path) -> Path:
    return sim_root.expanduser().resolve() / "src" / "physicar-sim" / "share"


def asset_paths(share: Path, basename: str) -> AssetPaths:
    return AssetPaths(
        share / "worlds" / f"{basename}.world",
        share / "routes" / f"{basename}.npy",
        share / "models" / basename,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_world(path: Path) -> tuple[ET.Element, ET.Element]:
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise SetupError(f"world XML is unreadable or malformed: {path}: {exc}") from exc
    worlds = [root] if root.tag == "world" else root.findall("world")
    if len(worlds) != 1:
        raise SetupError(f"expected exactly one top-level <world> in {path}")
    return root, worlds[0]


def require_canonical(config: EnvironmentConfig, paths: AssetPaths) -> bytes:
    if not paths.world.is_file():
        raise SetupError(f"canonical world is missing: {paths.world}")
    if not paths.route.is_file():
        raise SetupError(f"canonical route is missing: {paths.route}")
    if not paths.model.is_dir():
        raise SetupError(f"canonical model directory is missing: {paths.model}")
    for expected in ("model.config", f"{config.canonical_world}.sdf"):
        if not (paths.model / expected).is_file():
            raise SetupError(f"canonical model metadata is missing: {paths.model / expected}")

    _, world = parse_world(paths.world)
    if world.get("name") != config.canonical_world:
        raise SetupError(
            f"canonical internal world name is {world.get('name')!r}; "
            f"expected {config.canonical_world!r}"
        )
    counts = Counter(model.get("name") for model in world.findall("model"))
    invalid = [name for name in config.removed_models if counts[name] != 1]
    if invalid:
        details = ", ".join(f"{name}={counts[name]}" for name in invalid)
        raise SetupError(f"expected each removable top-level model exactly once: {details}")
    return paths.world.read_bytes()


def _model_spans(xml_bytes: bytes, removed_models: tuple[str, ...]) -> list[tuple[int, int]]:
    """Find byte spans of selected direct world children using a real XML parser."""
    parser = expat.ParserCreate()
    depth = 0
    world_depth: int | None = None
    active: list[tuple[str, int, int]] = []
    spans: list[tuple[int, int]] = []
    selected = set(removed_models)

    def start(name: str, attrs: dict[str, str]) -> None:
        nonlocal depth, world_depth
        depth += 1
        if name == "world":
            world_depth = depth
        elif (
            name == "model"
            and world_depth is not None
            and depth == world_depth + 1
            and attrs.get("name") in selected
        ):
            active.append((attrs["name"], depth, parser.CurrentByteIndex))

    def end(name: str) -> None:
        nonlocal depth
        if name == "model" and active and active[-1][1] == depth:
            _, _, start_index = active.pop()
            close_end = xml_bytes.find(b">", parser.CurrentByteIndex)
            if close_end < 0:
                raise SetupError("could not locate the end of a removable model block")
            spans.append((start_index, close_end + 1))
        depth -= 1

    parser.StartElementHandler = start
    parser.EndElementHandler = end
    try:
        parser.Parse(xml_bytes, True)
    except expat.ExpatError as exc:
        raise SetupError(f"canonical world XML is malformed: {exc}") from exc
    if len(spans) != len(removed_models):
        raise SetupError("could not identify every removable top-level model block")
    return spans


def _include_line_whitespace(xml_bytes: bytes, start: int, end: int) -> tuple[int, int]:
    line_start = xml_bytes.rfind(b"\n", 0, start) + 1
    if xml_bytes[line_start:start].strip() == b"":
        start = line_start
    if xml_bytes[end : end + 2] == b"\r\n":
        end += 2
    elif xml_bytes[end : end + 1] == b"\n":
        end += 1
    return start, end


def derive_world_bytes(xml_bytes: bytes, config: EnvironmentConfig) -> bytes:
    spans = [_include_line_whitespace(xml_bytes, *span) for span in _model_spans(xml_bytes, config.removed_models)]
    result = xml_bytes
    for start, end in sorted(spans, reverse=True):
        result = result[:start] + result[end:]

    world_match = re.search(br"<world\b[^>]*>", result)
    if world_match is None:
        raise SetupError("could not locate the canonical <world> start tag")
    tag = world_match.group(0)
    name_pattern = re.compile(
        br"(\bname\s*=\s*)(['\"])" + re.escape(config.canonical_world.encode("utf-8")) + br"\2"
    )
    replaced, count = name_pattern.subn(
        lambda match: match.group(1) + match.group(2) + config.derived_world.encode("utf-8") + match.group(2),
        tag,
    )
    if count != 1:
        raise SetupError("could not replace exactly one canonical world-name attribute")
    return result[: world_match.start()] + replaced + result[world_match.end() :]


def directory_manifest(root: Path) -> dict[str, tuple[str, str]]:
    manifest: dict[str, tuple[str, str]] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            manifest[relative] = ("symlink", os.readlink(path))
        elif path.is_dir():
            manifest[relative] = ("directory", "")
        elif path.is_file():
            manifest[relative] = ("file", sha256(path))
        else:
            manifest[relative] = ("other", "")
    return manifest


def canonical_fingerprint(paths: AssetPaths) -> tuple[str, str, dict[str, tuple[str, str]]]:
    return sha256(paths.world), sha256(paths.route), directory_manifest(paths.model)


def _xml_signature(element: ET.Element) -> tuple[object, ...]:
    """Represent XML semantics while ignoring formatting-only whitespace/comments."""
    return (
        element.tag,
        tuple(sorted(element.attrib.items())),
        (element.text or "").strip(),
        tuple(_xml_signature(child) for child in element),
    )


def _expected_world_signature(canonical_world: Path, config: EnvironmentConfig) -> tuple[object, ...]:
    root, world = parse_world(canonical_world)
    world.set("name", config.derived_world)
    for model in list(world.findall("model")):
        if model.get("name") in config.removed_models:
            world.remove(model)
    return _xml_signature(root)


def verify_derived(
    config: EnvironmentConfig,
    canonical: AssetPaths,
    derived: AssetPaths,
    expected_world: bytes | None = None,
) -> list[str]:
    canonical_bytes = require_canonical(config, canonical)
    expected_world = expected_world if expected_world is not None else derive_world_bytes(canonical_bytes, config)
    missing = [str(path) for path in (derived.world, derived.route, derived.model) if not path.exists()]
    if missing:
        raise SetupError(f"derived assets are missing: {', '.join(missing)}")
    if not derived.world.is_file() or not derived.route.is_file() or not derived.model.is_dir():
        raise SetupError("one or more derived asset targets have the wrong filesystem type")

    derived_root, world = parse_world(derived.world)
    if world.get("name") != config.derived_world:
        raise SetupError(f"derived internal world name is incorrect: {world.get('name')!r}")
    model_names = [model.get("name") for model in world.findall("model")]
    remaining_removed = [name for name in config.removed_models if name in model_names]
    if remaining_removed:
        raise SetupError(f"derived world still contains removed models: {', '.join(remaining_removed)}")
    if _xml_signature(derived_root) != _expected_world_signature(canonical.world, config):
        raise SetupError("derived world content differs from the canonical cone-removal transformation")
    if sha256(canonical.route) != sha256(derived.route):
        raise SetupError("derived route is not byte-identical to the canonical route")
    if directory_manifest(canonical.model) != directory_manifest(derived.model):
        raise SetupError("derived model directory does not match canonical model metadata")
    return [
        "canonical inputs valid (six expected cones present)",
        "derived world valid (world name correct; cone count 0)",
        "derived route byte-identical (SHA-256)",
        "derived model metadata matches canonical directory",
    ]


def _remove_exact_target(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _install_staged(staged: AssetPaths, destination: AssetPaths, force: bool) -> None:
    destinations = (destination.route, destination.model, destination.world)
    staged_paths = (staged.route, staged.model, staged.world)
    if force:
        for path in destinations:
            _remove_exact_target(path)
    for source, target in zip(staged_paths, destinations):
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, target)


def generate(config: EnvironmentConfig, share: Path, force: bool = False) -> str:
    canonical = asset_paths(share, config.canonical_world)
    derived = asset_paths(share, config.derived_world)
    canonical_bytes = require_canonical(config, canonical)
    fingerprint_before = canonical_fingerprint(canonical)
    expected_world = derive_world_bytes(canonical_bytes, config)

    if any(path.exists() for path in (derived.world, derived.route, derived.model)):
        try:
            verify_derived(config, canonical, derived, expected_world)
        except SetupError:
            if not force:
                raise SetupError(
                    "derived assets already exist but are invalid; rerun with --force to regenerate only those targets"
                )
        else:
            if canonical_fingerprint(canonical) != fingerprint_before:
                raise SetupError("canonical assets changed during verification")
            return "environment is already valid; no files rewritten"

    share.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".lane-follow-v1-", dir=share) as temp:
        stage_share = Path(temp)
        staged = asset_paths(stage_share, config.derived_world)
        staged.world.parent.mkdir(parents=True)
        staged.route.parent.mkdir(parents=True)
        staged.model.parent.mkdir(parents=True)
        staged.world.write_bytes(expected_world)
        shutil.copy2(canonical.route, staged.route)
        shutil.copytree(canonical.model, staged.model, symlinks=True, copy_function=shutil.copy2)
        verify_derived(config, canonical, staged, expected_world)
        _install_staged(staged, derived, force)

    verify_derived(config, canonical, derived, expected_world)
    if canonical_fingerprint(canonical) != fingerprint_before:
        raise SetupError("canonical assets changed during generation")
    return "environment generated and validated"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-root", type=Path, required=True, help="root of the physicar-ai-sim-docker checkout")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="environment identity configuration")
    parser.add_argument("--verify-only", action="store_true", help="validate canonical and derived assets without writing")
    parser.add_argument("--force", action="store_true", help="regenerate invalid derived assets only")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.verify_only and args.force:
        print("ERROR: --verify-only and --force cannot be combined", file=sys.stderr)
        return 2
    try:
        config = load_config(args.config)
        share = share_path(args.sim_root)
        canonical = asset_paths(share, config.canonical_world)
        derived = asset_paths(share, config.derived_world)
        if args.verify_only:
            fingerprint_before = canonical_fingerprint(canonical) if all(
                (canonical.world.is_file(), canonical.route.is_file(), canonical.model.is_dir())
            ) else None
            diagnostics = verify_derived(config, canonical, derived)
            if fingerprint_before is not None and canonical_fingerprint(canonical) != fingerprint_before:
                raise SetupError("canonical assets changed during verification")
            for diagnostic in diagnostics:
                print(f"PASS: {diagnostic}")
            print("PASS: canonical asset integrity unchanged during verification")
        else:
            print(f"PASS: {generate(config, share, force=args.force)}")
        return 0
    except (SetupError, OSError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
