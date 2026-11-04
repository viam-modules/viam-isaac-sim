#!/usr/bin/env python3
"""Turn OpenStreetMap marina features into Isaac Sim world props.

Takes a JSON list of OSM ways (as extracted from the OSM map API: each entry
{"id", "tags", "c": [[lat, lon], ...]}) and emits the "props" list for a
viam-labs:isaac-sim:world plus the "origin" for a movement_sensor, so the
sim's +X east / +Y north frame is anchored at a real place and GetPosition
reports real lat/lon.

  man_made=pier ways become yawed box props: fixed piers (floating=no) get a
  raised deck on pilings; floating docks sit at the waterline; unnamed
  floating ways are finger piers. natural=coastline becomes a seawall.

Usage:
  osm_to_props.py features.json > props.json
  osm_to_props.py features.json --origin 40.7858,-73.9856 --box -200,-200,150,150

The example input, 79th-street-boat-basin.osm.json, was fetched from
https://api.openstreetmap.org/api/0.6/map?bbox=-73.9905,40.7832,-73.9830,40.7895
and filtered to piers, the marina polygon and the coastline. Map data (c)
OpenStreetMap contributors, ODbL.
"""
import argparse
import json
import math
import sys

R = 6_371_000.0
WOOD = "wood"
PILING = [0.33, 0.27, 0.2]
ROCK = [0.45, 0.45, 0.42]


def yaw_quat(yaw):
    return [round(math.cos(yaw / 2), 6), 0, 0, round(math.sin(yaw / 2), 6)]


class Builder:
    def __init__(self):
        self.props = []
        self._pilings = set()

    def segment_box(self, name, a, b, width, height, z_center, color=None, material=None):
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        if length < 0.3:
            return
        prop = {
            "name": name, "type": "cube", "fixed": True,
            "position": [round((a[0] + b[0]) / 2, 3), round((a[1] + b[1]) / 2, 3), z_center],
            "orientation_wxyz": yaw_quat(math.atan2(dy, dx)),
            "size": 1.0, "scale": [round(length, 3), width, height],
        }
        if color:
            prop["color"] = color
        if material:
            prop["visual_material"] = material
        self.props.append(prop)

    def piling(self, name, p, size=0.35, height=2.4):
        key = (round(p[0]), round(p[1]))
        if key in self._pilings:
            return
        self._pilings.add(key)
        self.props.append({
            "name": name, "type": "cube", "fixed": True,
            "position": [round(p[0], 3), round(p[1], 3), round(height / 2 - 0.6, 3)],
            "size": 1.0, "scale": [size, size, height], "color": PILING,
        })

    def fixed_pier(self, tag, xy, wave_screen=True):
        for i in range(len(xy) - 1):
            a, b = xy[i], xy[i + 1]
            self.segment_box(f"{tag}_{i}", a, b, 3.0, 1.0, 0.6, material=WOOD)
            dx, dy = b[0] - a[0], b[1] - a[1]
            L = math.hypot(dx, dy)
            if L <= 1:
                continue
            if wave_screen:
                nx, ny = -dy / L, dx / L
                if ny < 0:
                    nx, ny = -nx, -ny
                self.segment_box(f"{tag}_screen_{i}", [a[0] + nx * 1.8, a[1] + ny * 1.8],
                                 [b[0] + nx * 1.8, b[1] + ny * 1.8], 0.3, 2.2, 0.1, color=[0.55, 0.45, 0.35])
            n = max(1, int(L / 12))
            for k in range(n + 1):
                s = k / n
                self.piling(f"{tag}_piling_{i}_{k}", (a[0] + dx * s, a[1] + dy * s), 0.4, 3.0)

    def floating_dock(self, tag, xy, width=2.4):
        for i in range(len(xy) - 1):
            self.segment_box(f"{tag}_{i}", xy[i], xy[i + 1], width, 0.6, 0.1, material=WOOD)
        for i, p in enumerate(xy):
            if i % 3 == 0 or i == len(xy) - 1:
                self.piling(f"{tag}_piling_{i}", p)

    def finger(self, tag, xy):
        for i in range(len(xy) - 1):
            self.segment_box(f"{tag}_{i}", xy[i], xy[i + 1], 1.0, 0.5, 0.1, color=[0.76, 0.6, 0.42])

    def seawall(self, tag, xy):
        for i in range(len(xy) - 1):
            self.segment_box(f"{tag}_{i}", xy[i], xy[i + 1], 2.0, 3.0, 0.5, color=ROCK)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("features", help="JSON list of OSM ways with lat/lon coords")
    ap.add_argument("--origin", help="lat,lon of the sim origin; default the marina polygon centroid")
    ap.add_argument("--box", default="-250,-250,250,250", help="xmin,ymin,xmax,ymax meters to keep")
    args = ap.parse_args()

    with open(args.features) as fh:
        feats = json.load(fh)
    if args.origin:
        lat0, lon0 = (float(v) for v in args.origin.split(","))
    else:
        marina = next((f for f in feats if f["tags"].get("leisure") == "marina"), None)
        if marina is None:
            sys.exit("no leisure=marina way to center on; pass --origin")
        lat0 = sum(c[0] for c in marina["c"]) / len(marina["c"])
        lon0 = sum(c[1] for c in marina["c"]) / len(marina["c"])
    xmin, ymin, xmax, ymax = (float(v) for v in args.box.split(","))

    def enu(lat, lon):
        return ((lon - lon0) * math.pi / 180 * R * math.cos(math.radians(lat0)),
                (lat - lat0) * math.pi / 180 * R)

    b = Builder()
    for f in feats:
        t = f["tags"]
        xy = [enu(*c) for c in f["c"]]
        xy = [p for p in xy if xmin <= p[0] <= xmax and ymin <= p[1] <= ymax]
        if len(xy) < 2:
            continue
        fid = f["id"][-4:]
        if t.get("man_made") == "pier":
            name = t.get("name", "")
            slug = "".join(ch if ch.isalnum() else "_" for ch in name.lower()).strip("_") or f"pier_{fid}"
            if t.get("floating") == "no":
                b.fixed_pier(slug, xy)
            elif name or t.get("floating") != "yes":
                b.floating_dock(slug if name else f"main_{fid}", xy)
            else:
                b.finger(f"finger_{fid}", xy)
        elif t.get("natural") == "coastline":
            b.seawall(f"seawall_{fid}", xy)

    json.dump({"origin": {"latitude": round(lat0, 6), "longitude": round(lon0, 6), "altitude_m": 0},
               "props": b.props}, sys.stdout, indent=1)
    print(file=sys.stdout)
    print(f"{len(b.props)} props about ({lat0:.6f}, {lon0:.6f})", file=sys.stderr)


if __name__ == "__main__":
    main()
