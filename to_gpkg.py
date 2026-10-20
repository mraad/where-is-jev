"""Combine GeoJSON files into one spatially indexed GeoPackage for ArcGIS Pro, then verify it."""

import argparse
from pathlib import Path
import sqlite3

from pyogrio.raw import read, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path)
    p.add_argument("layers", nargs="+", help="LAYER=path.geojson")
    a = p.parse_args()
    if a.out.exists():
        p.error("Output exists; choose a new path")
    pairs = [layer.split("=", 1) for layer in a.layers]
    if any(len(pair) != 2 or not pair[0] for pair in pairs):
        p.error("Use LAYER=path.geojson")
    for layer, path in pairs:
        meta, _, geometry, fields = read(path)
        # Mixed Polygon/MultiPolygon reads as Unknown; ArcGIS Pro needs one geometry type per layer.
        kind = "MultiPolygon" if meta["geometry_type"] == "Unknown" else meta["geometry_type"]
        write(a.out, geometry, fields, meta["fields"], layer=layer, driver="GPKG", geometry_type=kind,
              crs=meta["crs"], promote_to_multi=kind.startswith("Multi"),
              layer_options={"SPATIAL_INDEX": "YES"})
        assert len(read(a.out, layer=layer)[2]) == len(geometry), layer
        print(f"{layer}: {len(geometry)} {kind} features from {path}")
    with sqlite3.connect(a.out) as db:
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


if __name__ == "__main__":
    main()
