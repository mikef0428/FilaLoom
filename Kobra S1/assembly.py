"""Assembled Kobra S1 FilaLoom head as a GLB (one node per part, for the viewer's explode tool).

Run after faceplate.py:  uv run --with trimesh --with manifold3d --with pymeshfix --with scipy --with networkx --with shapely --with rtree --with mapbox-earcut python assembly.py
Frame: faceplate frame (+Y out of the printer front, +Z up). Needle carrier set for A = 4 mm.
"""
import itertools

import numpy as np
import trimesh

import faceplate as F

FL = F.HERE.parent / "FilaLoom Components STLs"
A_MM = 4.0                    # nozzle below cover edge (guess until measured)


def cover_frame(pz_face):
    """FilaLoom A1 cover/depositor file frame -> faceplate frame (their docking face at pz_face)."""
    return np.array([[-1, 0, 0, F.PANEL_CX + F.PATTERN_CENTER[0]],
                     [0, 0, 1, F.FACE_Y - pz_face],
                     [0, 1, 0, F.PANEL_CZ - F.PATTERN_CENTER[1]],
                     [0, 0, 0, 1.0]])


FRONT_BODY = cover_frame(67.5 + 39.0)          # Back Body face 67.5; Front Body glue face 108 = Back Body 69
SLOT = FRONT_BODY @ trimesh.transformations.translation_matrix([5.0, 0.0, 120.0])  # needle slot floor
# ponytail: docking-bar seat read off the Front Body side cutout, not from an assembly file; check by eye
BARS = FRONT_BODY @ trimesh.transformations.translation_matrix([51.5 - 70.0, 0.0, 108.0 - 78.5])


def placed(mesh, T):
    m = mesh.copy()
    m.apply_transform(T)
    return m


def needle(t):
    cx, z = F.SLOT_X / 2, F.HOLDER_T + 2.0 + F.HUB_BORE_D / 2
    root = t - F.CARRIER_H + F.HUB_STEP_H - F.HUB_COLLAR

    def cyl(d, y0, y1):
        c = trimesh.creation.cylinder(radius=d / 2, height=y1 - y0, sections=32)
        c.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
        c.apply_translation([cx, (y0 + y1) / 2, z])
        return c

    return trimesh.util.concatenate([cyl(2.8, root - F.TUBE_L, root), cyl(3.6, root, root + F.HUB_COLLAR),
                                     cyl(5.7, root + F.HUB_COLLAR, root + 12.0)])


def parts():
    t = F.carrier_top_for(A_MM)
    back = F.back_cover()
    back.apply_transform(F.BACK_TO_FRONT)
    load = lambda p: trimesh.load(p)
    dep = {spec[0]: F.depositor_5x2(*spec) for spec in F.DEPOSITOR_POCKETS}   # file frame (STLs are on-bed)
    return [
        ("S1 Faceplate", F.faceplate(), [74, 111, 165]),
        ("S1 Back Cover", back, [90, 150, 120]),
        ("Lock Cylinder", placed(load(FL / "Print Head Cover/Cover - Lock Cylinder.stl"), cover_frame(26.9)), [170, 170, 170]),
        ("Depositor Back Body", placed(dep["Back Body"], cover_frame(67.5)), [80, 110, 220]),
        ("Depositor Front Body", placed(dep["Front Body"], FRONT_BODY), [60, 90, 200]),
        ("Depositor Docking Bars", placed(dep["Docking Bars"], BARS), [100, 130, 230]),
        ("Needle Holder Plate", placed(F.holder_plate(), SLOT), [210, 140, 60]),
        ("Needle Carrier", placed(F.carrier(printable=False),
                                  SLOT @ trimesh.transformations.translation_matrix([0, t, F.HOLDER_T])), [220, 110, 50]),
        ("Needle 12G (stand-in)", placed(needle(t), SLOT), [200, 200, 205]),
    ]


if __name__ == "__main__":
    ps = parts()
    scene = trimesh.Scene()
    y_up = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])   # glTF is Y-up
    for name, m, col in ps:
        g = m.copy()
        g.apply_transform(y_up)
        g.visual.face_colors = col + [255]
        scene.add_geometry(g, node_name=name, geom_name=name)
    out = F.HERE / "STL" / "S1 FilaLoom Head - Assembled.glb"
    scene.export(out)
    print("wrote", out.name)
    # interference report: pairs sharing volume (needle stand-in is a surface mesh, so it's left out)
    solid = [(n, m) for n, m, _ in ps if m.is_watertight]
    for (na, a), (nb, b) in itertools.combinations(solid, 2):
        if not trimesh.bounds.contains(np.vstack([a.bounds[0] - 1, a.bounds[1] + 1]), b.vertices).any() and \
           not trimesh.bounds.contains(np.vstack([b.bounds[0] - 1, b.bounds[1] + 1]), a.vertices).any():
            continue
        v = a.intersection(b, engine="manifold").volume
        if v > 0.05:
            print(f"  overlap {na} x {nb}: {v:.2f} mm3")
    tip_z = ps[-1][1].bounds[0][2]
    print(f"needle tip z {tip_z:.1f} (cover edge {F.COVER_BOTTOM_Z}); depositor docked {F.FACE_Y:.1f} face")
