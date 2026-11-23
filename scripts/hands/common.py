"""MuJoCo Menagerie's dexterous hands (LEAP, Allegro, Shadow; right hands) as MJCF text, with an optional free cube."""
import xml.etree.ElementTree as ET
from pathlib import Path

# name: (Menagerie directory, position of the free cube; the positions are guesses above each hand, not tuned)
HANDS = {
    "leap": ("leap_hand", "0.1 0 0.1"),
    "allegro": ("wonik_allegro", "0 0 0.25"),
    "shadow": ("shadow_hand", "0.35 0 0.1"),
}
CUBE = ('<body name="cube" pos="{pos}"><freejoint name="cube_free"/>'
        '<geom name="cube_geom" type="box" size="0.03 0.03 0.03" mass="0.1"/></body>')


def mjcf(menagerie, hand, cube=False, cube_pos=None):
    """The MJCF text of <menagerie>/<dir>/right_hand.xml with absolute asset directories (so it compiles from a
    string), and with a free 6 cm cube of 0.1 kg under the world when cube is set."""
    d, pos = HANDS[hand]
    base = Path(menagerie).resolve() / d
    root = ET.parse(base / "right_hand.xml").getroot()
    comp = root.find("compiler")
    for k in ("meshdir", "texturedir", "assetdir"):
        if comp is not None and comp.get(k):
            comp.set(k, str(base / comp.get(k)))
    if cube:
        root.find("worldbody").append(ET.fromstring(CUBE.format(pos=cube_pos or pos)))
    return ET.tostring(root, encoding="unicode")
