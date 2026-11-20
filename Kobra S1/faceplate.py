"""Kobra S1 low-profile magnetic faceplate with the FilaLoom docking face.

Remix of "Kobra S1 Head Cover Magnetic" (MakerWorld 2147233): keeps its rear half
(magnet cups, side rails, locating ribs), moves the front panel back to the hotend-fan
plane, and puts the FilaLoom A1 docking interface (3 cone holes + lock lug) on it.
Magnet pockets are re-cut for 5x2 mm discs, front and back.

Run:  uv run --with trimesh --with manifold3d --with pymeshfix --with scipy --with networkx --with shapely --with rtree python faceplate.py
Frame is the reference mesh's: +Y out of the printer front, +Z up, mm.
"""
from pathlib import Path

import numpy as np
import pymeshfix
import trimesh
from shapely.geometry import Point, box as sbox
from shapely.ops import unary_union

HERE = Path(__file__).parent
REF = HERE.parent / "Kobra+S1+Head+Cover+Magnetic"

# --- reference geometry (measured from the STLs) ---
PARTING_Y = -56.93            # front shell's rear face, meets the back cover
FAN_FACE_Y = -21.6            # rear end of the stock fan posts = fan front face
FAN_HOLES = [(101.05, 111.6), (145.05, 149.8), (122.9, 173.2)]  # x, z
FAN_HOLE_D, FAN_CSK_D = 3.8, 5.4
TOP_POST_END_Y = -26.1        # the top fan ear sits 4.5 mm deeper
FAN_CENTER, FAN_OPEN_D = (125.0, 130.0), 36.0
FRONT_CUPS = [(96.25, 77.2), (154.6, 77.25), (96.2, 157.35), (154.65, 157.35)]
BACK_CUPS = [(398.3, 94.4), (398.35, 152.9), (478.5, 94.4), (478.5, 152.9)]
OLD_CUP_D, OLD_CUP_DEPTH = 5.78, 3.0
X0, X1, Z0, Z1, CORNER_R = 92.4, 158.4, 92.0, 176.59, 5.0

# --- FilaLoom A1 docking plate (Cover - Front.stl), plate coords px right, py up, pz out ---
PLATE_T = 4.9
CONES = [(9.0, 9.0), (44.0, 9.0), (26.5, 66.0)]
CONE_TOP_D, CONE_TIP_D, CONE_DEPTH = 10.1, 2.2, 3.95
LUG_PX, LUG_PY = (19.01, 34.01), (34.05, 40.05)
LUG_H, LUG_TOP_R = 15.25, 6.0
LOCK_D, LOCK_PZ = 6.06, 7.77   # hole axis along py, height above plate face
PATTERN_CENTER = (26.5, 37.0)

DOCK_MAGNET_D, DOCK_MAGNET_T = 5.3, 2.1  # 5x2 discs behind the cones (A1 used 8x2), glue fit

# --- docked depositor needle, from its STLs (plate coords) ---
# needle axis: holder bore 6 mm above slot floor (Front Body z 120) -> z 126; docking face = Back Body
# plate face 1.5 mm past Front Body z 108 -> 19.5 mm in front of the cover face, centred under the lug.
# tip: 12G 38 mm tube, hub hex seats on the bore step (y 3), collar 2.3 mm -> tube root y 0.7 -> tip y -37.3.
# Cross-check on the A1: paper says tip is 25 below nozzle -> nozzle 12.3 below plate bottom (photo ~13.5).
NEEDLE_FWD, NEEDLE_TIP_PY = 19.5, -37.3
COVER_BOTTOM_Z = 73.44        # lowest edge of the S1 shell (and the magnetic back cover)

# --- knobs ---
MAGNET_T = 2.1                 # 5x2 disc seat depth; pocket stays the original 5.78 (glue fit)
BRIDGE_H = 10.0                # bar across the fan opening that carries the lug

PANEL_T = PLATE_T + DOCK_MAGNET_T   # A1 plate + its glued-on magnet layer, in one piece
FACE_Y = FAN_FACE_Y + PANEL_T
PANEL_CX, PANEL_CZ = (X0 + X1) / 2, (Z0 + Z1) / 2


def to_panel(px, py):
    """Plate coords -> panel x, z (viewer's right is -X)."""
    return PANEL_CX - (px - PATTERN_CENTER[0]), PANEL_CZ + (py - PATTERN_CENTER[1])


def cyl_y(x, z, d, y0, y1, sections=64):
    """Cylinder along Y from y0 to y1."""
    c = trimesh.creation.cylinder(radius=d / 2, height=abs(y1 - y0), sections=sections)
    c.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    c.apply_translation([x, (y0 + y1) / 2, z])
    return c


def cone_y(x, z, d_top, d_tip, y_top, depth):
    """Frustum opening at y_top, narrowing toward -Y."""
    c = trimesh.creation.cone(radius=d_top / 2, height=depth * d_top / (d_top - d_tip), sections=64)
    c.apply_transform(trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]))  # apex down -Z
    c = c.intersection(trimesh.creation.box(extents=[d_top + 1, d_top + 1, depth + 0.2],
                       transform=trimesh.transformations.translation_matrix([0, 0, -(depth + 0.2) / 2 + 0.1])))
    c.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))  # -Z -> -Y
    c.apply_translation([x, y_top, z])
    return c


def extrude_xz(poly, y0, y1):
    """Extrude a shapely polygon in the X-Z plane along Y."""
    m = trimesh.creation.extrude_polygon(poly, height=y1 - y0)
    m.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))  # (x,y,z)->(x,-z,y)
    m.apply_translation([0, y1, 0])
    return m


def recut_magnets(mesh, cups, face_y, inward):
    """Raise the floor of the old 5.78 x 3 cups so a 5x2 magnet sits flush (glue fit)."""
    shims = [cyl_y(x, z, OLD_CUP_D + 0.4, face_y + inward * MAGNET_T, face_y + inward * (OLD_CUP_DEPTH + 0.3))
             for x, z in cups]
    return mesh.union(shims, engine="manifold")


def load_fixed(path):
    m = trimesh.load(path)
    m.merge_vertices()
    v, f = pymeshfix.clean_from_arrays(np.ascontiguousarray(m.vertices, dtype=np.float64),
                                       np.ascontiguousarray(m.faces, dtype=np.int32))
    return trimesh.Trimesh(v, f)


def faceplate():
    shell = load_fixed(REF / "kobra s1 front 5x3 magnets.stl")
    shell = shell.intersection(trimesh.creation.box(bounds=[[0, -100, 0], [300, FAN_FACE_Y, 300]]), engine="manifold")
    shell = recut_magnets(shell, FRONT_CUPS, PARTING_Y, +1)

    panel = extrude_xz(sbox(X0, Z0, X1, Z1).buffer(-CORNER_R).buffer(CORNER_R), FAN_FACE_Y, FACE_Y)

    lx0, lz0 = to_panel(LUG_PX[1], LUG_PY[0])
    lx1, lz1 = to_panel(LUG_PX[0], LUG_PY[1])
    w = lx1 - lx0
    fin = unary_union([sbox(0, 0, w, LUG_H - LUG_TOP_R), sbox(LUG_TOP_R, 0, w - LUG_TOP_R, LUG_H),
                       Point(LUG_TOP_R, LUG_H - LUG_TOP_R).buffer(LUG_TOP_R, 64),
                       Point(w - LUG_TOP_R, LUG_H - LUG_TOP_R).buffer(LUG_TOP_R, 64)])
    lug = trimesh.creation.extrude_polygon(fin, height=lz1 - lz0)        # x=width, y=height out, z=thickness
    lug.apply_translation([lx0, FACE_Y - 0.01, lz0])

    boss = cyl_y(*FAN_HOLES[2], 7.0, TOP_POST_END_Y, FAN_FACE_Y + 0.01)
    body = shell.union([panel, lug, boss], engine="manifold")

    cx, cz = to_panel(*PATTERN_CENTER)
    fan_open = Point(*FAN_CENTER).buffer(FAN_OPEN_D / 2, 128).difference(
        sbox(X0, cz - BRIDGE_H / 2, X1, cz + BRIDGE_H / 2))
    cuts = [extrude_xz(p, FAN_FACE_Y - 1, FACE_Y + 1) for p in getattr(fan_open, "geoms", [fan_open])]
    for x, z in FAN_HOLES:
        cuts.append(cyl_y(x, z, FAN_HOLE_D, TOP_POST_END_Y - 1, FACE_Y + 1, 32))
        cuts.append(cone_y(x, z, FAN_CSK_D + 0.02, FAN_HOLE_D - 0.6, FACE_Y + 0.01, (FAN_CSK_D - FAN_HOLE_D + 0.6) / 2))
    for px, py in CONES:
        cuts.append(cone_y(*to_panel(px, py), CONE_TOP_D, CONE_TIP_D, FACE_Y + 0.01, CONE_DEPTH))
        cuts.append(cyl_y(*to_panel(px, py), DOCK_MAGNET_D, FAN_FACE_Y - 1, FAN_FACE_Y + DOCK_MAGNET_T))
    lock = trimesh.creation.cylinder(radius=LOCK_D / 2, height=lz1 - lz0 + 2, sections=64)
    lock.apply_translation([cx, FACE_Y + LOCK_PZ, (lz0 + lz1) / 2])
    cuts.append(lock)
    return body.difference(cuts, engine="manifold")


# --- adjustable needle holder (replaces Depositor - Needle Holder) ---
# Local frame = depositor plate coords: x across, y up (0 = depositor bottom edge), z out of the
# Front Body's flat side (0 = slot floor). The plate glues into the 43 x 12 slot like the stock holder;
# the carrier slides on the arm and clamps with 2x M3x20 + washers + nuts behind the arm.
SLOT_X, SLOT_Y, HOLDER_T = 42.8, 12.0, 5.5
ARM_W, ARM_BOTTOM = 22.0, -32.0
SCREW_DX, M3_CLEAR = 6.5, 3.4
CARRIER_H, CARRIER_T = 12.0, 9.5
HUB_BORE_D, HUB_STEP_D, HUB_STEP_H = 6.0, 5.5, 3.0   # stock holder's bore: hex seats on the step
HUB_COLLAR = 2.3              # 12G hub: tube root sits this far below the hex seat
TUBE_L = 38.0
CARRIER_TOP = (4.0, -20.0)    # travel of the carrier's top edge, plate y
COVER_EDGE_PY = COVER_BOTTOM_Z - PANEL_CZ + PATTERN_CENTER[1]   # S1 cover bottom edge in plate y


def tip_py(carrier_top):
    return carrier_top - CARRIER_H + HUB_STEP_H - HUB_COLLAR - TUBE_L


def carrier_top_for(a_mm, dip=25.0):
    """Carrier top (plate y) that puts the tip `dip` below a nozzle `a_mm` under the cover edge."""
    return carrier_top_for_tip(COVER_EDGE_PY - a_mm - dip)


def carrier_top_for_tip(tip):
    return tip - tip_py(0.0)


def holder_plate():
    cx = SLOT_X / 2
    plate = unary_union([sbox(0, 0, SLOT_X, SLOT_Y), sbox(cx - ARM_W / 2, ARM_BOTTOM, cx + ARM_W / 2, SLOT_Y)])
    for sx in (cx - SCREW_DX, cx + SCREW_DX):
        y0, y1 = CARRIER_TOP[1] - CARRIER_H / 2, CARRIER_TOP[0] - CARRIER_H / 2
        plate = plate.difference(Point(sx, y0).buffer(M3_CLEAR / 2, 32).union(
            Point(sx, y1).buffer(M3_CLEAR / 2, 32)).convex_hull)
    return trimesh.creation.extrude_polygon(plate, height=HOLDER_T)   # prints flat as-is


def carrier():
    cx = SLOT_X / 2
    blk = trimesh.creation.box(bounds=[[cx - ARM_W / 2, -CARRIER_H, 0], [cx + ARM_W / 2, 0, CARRIER_T]])
    bz = 2.0 + HUB_BORE_D / 2   # 2 mm wall to the arm

    def bore_y(d, y0, y1):
        c = trimesh.creation.cylinder(radius=d / 2, height=y1 - y0, sections=64)
        c.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
        c.apply_translation([cx, (y0 + y1) / 2, bz])
        return c

    cuts = [bore_y(HUB_BORE_D, -CARRIER_H + HUB_STEP_H, 1), bore_y(HUB_STEP_D, -CARRIER_H - 1, 0)]
    for sx in (cx - SCREW_DX, cx + SCREW_DX):
        h = trimesh.creation.cylinder(radius=M3_CLEAR / 2, height=CARRIER_T + 2, sections=32)
        h.apply_translation([sx, -CARRIER_H / 2, CARRIER_T / 2])
        cuts.append(h)
    c = blk.difference(cuts, engine="manifold")
    c.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))  # bore vertical
    c.apply_translation([0, 0, -c.bounds[0][2]])
    return c


# --- FilaLoom depositor re-seated for 5x2 magnets (stock uses 8x2 in 8.49 pockets) ---
DEP = HERE.parent / "FilaLoom Components STLs" / "Yarn Depositor"
SEAT_D, SEAT_T, OLD_POCKET_D = 5.3, 2.1, 8.49
DOCK_PTS = [(8.97, 9.0), (44.0, 8.98), (26.5, 66.0)]   # x, y in the bodies' shared frame
# part, axis, pocket centres (other two axes, in order), face, inward sign, old depth, new seat?
DEPOSITOR_POCKETS = [
    ("Front Body", 2, DOCK_PTS, 108.5, +1, 1.1, True),    # glue face 108.5 (bumps reach 108.0); magnets live here
    ("Back Body", 2, DOCK_PTS, 69.0, -1, 1.5, False),     # matching recess, now solid
    ("Docking Bars", 0, [(19.03, 85.0), (63.06, 84.98)], 70.0, +1, 2.4, True),
]


def cyl_axis(axis, c2, d, a0, a1):
    c = trimesh.creation.cylinder(radius=d / 2, height=abs(a1 - a0), sections=64)
    if axis == 0:
        c.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
    elif axis == 1:
        c.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    pos = np.insert(np.array(c2, float), axis, (a0 + a1) / 2)
    c.apply_translation(pos)
    return c


def depositor_5x2(name, axis, pts, face, inward, old_depth, seat):
    m = trimesh.load(DEP / f"Depositor - {name}.stl")
    plugs = [cyl_axis(axis, p, OLD_POCKET_D + 0.4, face, face + inward * (old_depth + 0.2)) for p in pts]
    m = m.union(plugs, engine="manifold")
    if seat:
        m = m.difference([cyl_axis(axis, p, SEAT_D, face - inward * 1, face + inward * SEAT_T) for p in pts],
                         engine="manifold")
    return m


def back_cover():
    back = load_fixed(REF / "kobra s1 back 5x3 magnets.stl")
    face_y = back.bounds[0][1]
    return recut_magnets(back, BACK_CUPS, face_y, +1)


if __name__ == "__main__":
    out = HERE / "STL"
    out.mkdir(exist_ok=True)
    fp = faceplate()
    assert fp.is_watertight and fp.body_count == 1, (fp.is_watertight, fp.body_count)
    depth = FACE_Y - PARTING_Y
    assert abs(depth - 42.33) < 0.05, depth
    # print standing as mounted (cad dfm --candidates: ~4x less support than face-up); use a brim
    fp.apply_translation(-np.array([fp.centroid[0], fp.centroid[1], fp.bounds[0][2]]))
    fp.export(out / "S1 Faceplate - FilaLoom Magnetic 5x2.stl")
    bk = back_cover()
    assert bk.is_watertight and bk.body_count == 1, (bk.is_watertight, bk.body_count)
    bk.export(out / "S1 Back Cover - Magnetic 5x2.stl")
    print(f"faceplate depth {depth:.1f} mm (stock magnetic 56.9), lug +{LUG_H} mm; bounds {fp.bounds.round(1).tolist()}")
    tip_z = to_panel(0, NEEDLE_TIP_PY)[1]
    print(f"docked needle: axis {FACE_Y + NEEDLE_FWD:.1f} mm in front of the old cover face, "
          f"tip {COVER_BOTTOM_Z - tip_z:.1f} mm below the cover's bottom edge "
          f"(needle dip below nozzle = that minus the nozzle's drop below the cover edge)")

    for name, part in [("S1 Needle Holder - Adjustable Plate", holder_plate()), ("S1 Needle Holder - Carrier", carrier())]:
        assert part.is_watertight and part.body_count == 1, name
        part.export(out / f"{name}.stl")
    # tip moves 1:1 with the carrier; stock holder put the tip at plate y -37.3
    assert abs(tip_py(0.0) - (-49.3)) < 1e-9 and abs(tip_py(-20) - tip_py(0) + 20) < 1e-9
    assert abs(carrier_top_for_tip(tip_py(-7.5)) + 7.5) < 1e-9
    lo, hi = (COVER_EDGE_PY - 25 - tip_py(t) for t in CARRIER_TOP)
    print(f"adjustable holder: 25 mm dip works for nozzle {lo:.1f} to {hi:.1f} mm below the cover edge")
    print("A (nozzle below cover edge) -> carrier top below depositor bottom edge, for 25 mm needle dip:")
    for a in range(0, 16, 2):
        print(f"  A={a:2d} mm -> {-carrier_top_for(a):5.1f} mm")

    for spec in DEPOSITOR_POCKETS:
        part = depositor_5x2(*spec)
        assert part.is_watertight and part.body_count == 1, spec[0]
        part.export(out / f"Depositor - {spec[0]} 5x2.stl")
    print("depositor: Front Body, Back Body, Docking Bars re-seated for 5x2 magnets")
