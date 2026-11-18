"""Persist web-drawn zones back to config/nodes.yaml.

PyYAML round-trips comments away, so the leading comment header is
re-attached after the dump; inline comments inside the node list are lost
(ponytail: switch to ruamel.yaml only if nodes.yaml gains inline comments).
Writes are atomic (tmp file + os.replace) and lock-guarded so two node
workers saving at once cannot interleave or leave a half-written file.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import yaml

_LOCK = threading.Lock()
_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "config" / "nodes.yaml"


def _leading_comments(text: str) -> str:
    """Everything before the first YAML content line (the file header)."""
    lines = []
    for line in text.splitlines():
        if line.strip() == "" or line.lstrip().startswith("#"):
            lines.append(line)
        else:
            break
    return "\n".join(lines)


def persist_zone(node_id: str, pixel_points, zone_name: str = "zone",
                 path=None) -> list:
    """Overwrite `node_id`'s zones[0] polygon while keeping zone attrs & extra zones.

    Returns a shallow copy of the saved zones list. Raises KeyError when the
    node id is not in the registry file; callers log it and keep the in-memory zone."""
    path = Path(path) if path else _DEFAULT_PATH
    with _LOCK:
        raw = path.read_text(encoding="utf-8") if path.exists() else ""
        doc = yaml.safe_load(raw) or {}
        nodes = doc.get("nodes") or []
        saved_zones = None
        for node in nodes:
            if node.get("id") == node_id:
                existing_zones = node.get("zones")
                poly = [[int(x), int(y)] for x, y in pixel_points]
                if (isinstance(existing_zones, list) and existing_zones
                        and isinstance(existing_zones[0], dict)):
                    first = dict(existing_zones[0])
                    first["polygon"] = poly
                    first.setdefault("name", zone_name)
                    saved_zones = [first, *[dict(z) if isinstance(z, dict) else z for z in existing_zones[1:]]]
                else:
                    saved_zones = [{"name": zone_name, "polygon": poly}]
                node["zones"] = saved_zones
                break
        else:
            raise KeyError(f"node {node_id} not found in {path.name}")
        body = yaml.safe_dump(doc, allow_unicode=True, sort_keys=False)
        header = _leading_comments(raw)
        tmp = path.with_suffix(".yaml.tmp")
        tmp.write_text(header + ("\n" if header else "") + body,
                       encoding="utf-8")
        os.replace(tmp, path)
        return [dict(z) if isinstance(z, dict) else z for z in saved_zones]
