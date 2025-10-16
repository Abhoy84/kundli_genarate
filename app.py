# app.py
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import swisseph as swe
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, PathPatch
from matplotlib.path import Path
from timezonefinder import TimezoneFinder
from datetime import datetime
import pytz
import io
import os
import time
import requests
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import math
import numpy as np

app = Flask(__name__)
CORS(app)

# -----------------------------
# Geocoding helpers
# -----------------------------
def geocode_nominatim(place, retries=3, timeout=10):
    geolocator = Nominatim(user_agent="kundli_app_local")
    for attempt in range(retries):
        try:
            loc = geolocator.geocode(place, timeout=timeout)
            if loc:
                return {"lat": loc.latitude, "lon": loc.longitude, "display_name": loc.address}
        except (GeocoderTimedOut, GeocoderServiceError) as e:
            app.logger.warning(f"Nominatim attempt {attempt+1} failed: {e}")
            time.sleep(1 + attempt)
        except Exception as e:
            app.logger.exception("Unexpected geopy error")
            break
    return None

def get_location(place):
    return geocode_nominatim(place)

# -----------------------------
# Swiss Ephemeris helper
# -----------------------------
def safe_calc_longitude(jd_ut, planet_code):
    try:
        res = swe.calc_ut(jd_ut, planet_code)
        lon_val = res[0][0] if isinstance(res[0], (list, tuple)) else res[0]
        return float(lon_val) % 360.0
    except Exception as e:
        app.logger.exception(f"Failed to calc planet {planet_code}: {e}")
        return None

# -----------------------------
# Curved house shape helper (Bezier)
# -----------------------------
def curved_house_path(cx, cy, size, rotation=0.0, bulge=0.35):
    """
    Create a Path (closed) representing a curved diamond-like house centered at (cx,cy).
    `size` controls radius; `rotation` rotates the shape; `bulge` controls curvature (0..0.6)
    """
    # Basic diamond control points (top, right, bottom, left)
    pts = np.array([
        [0.0, size],
        [size, 0.0],
        [0.0, -size],
        [-size, 0.0],
    ])
    # Apply rotation
    c, s = math.cos(rotation), math.sin(rotation)
    R = np.array([[c, -s], [s, c]])
    pts = pts.dot(R.T) + np.array([cx, cy])

    # Build a cubic Bezier closed path using intermediate control points to bulge edges outward
    # For each edge from P[i] to P[i+1], create one control point near the midpoint offset outward
    verts = []
    codes = []

    n = len(pts)
    for i in range(n):
        p0 = pts[i]
        p1 = pts[(i+1) % n]
        # midpoint
        mid = 0.5 * (p0 + p1)
        # outward normal (pointing away from center (cx,cy))
        center = np.array([cx, cy])
        out_dir = mid - center
        norm = np.linalg.norm(out_dir)
        if norm == 0:
            out = np.array([0.0, 0.0])
        else:
            out = (out_dir / norm) * (size * bulge)
        # two control points for cubic: cp1 near p0 toward midpoint+out, cp2 near p1 toward midpoint+out
        cp1 = p0 + 0.35 * (mid + out - p0)
        cp2 = p1 + 0.35 * (mid + out - p1)

        if i == 0:
            # move to first point
            verts.append(tuple(p0))
            codes.append(Path.MOVETO)

        # add cubic curve (cp1, cp2, p1)
        verts.append(tuple(cp1))
        codes.append(Path.CURVE4)
        verts.append(tuple(cp2))
        codes.append(Path.CURVE4)
        verts.append(tuple(p1))
        codes.append(Path.CURVE4)

    path = Path(verts, codes)
    return path

# -----------------------------
# Draw North-Indian Kundli with curved shapes (style #4)
# -----------------------------
def draw_connected_north_kundli_bright(name, ascendant_deg, planets_dict, info_text=""):
    """
    Draws a North-Indian Kundli that matches the bright cream/orange/red sample geometry.
    Planet labels are formatted like: "Ve 12:18 Has" (short planet, deg:min, nakshatra short).
    Returns a BytesIO PNG (300 dpi).
    """
    import matplotlib.pyplot as plt
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch, FancyBboxPatch
    import numpy as np
    import math
    import io

    # --- Colors & fonts ---
    BG = "#FFF7D9"         # cream background
    ORANGE = "#ff6a00"     # outer orange stroke
    RED = "#cc1a00"        # inner red stroke & text
    DEEP = "#8b0000"       # footer darker red
    PLANET_TEXT = RED
    RASHI_TEXT = RED
    ASC_TEXT = RED

    fig, ax = plt.subplots(figsize=(8.5, 7.5), dpi=150)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis("off")

    # ---------------------------
    # Hardcoded geometry (matches sample)
    # ---------------------------
    # We'll create 12 curved diamond house shapes using consistent control points.
    # Centers chosen to match the sample geometry visually.
    slot_centers = {
        1: (0.0,  0.70),
        2: (-0.45, 0.45),
        3: (-0.75, 0.0),
        4: (-0.45, -0.45),
        5: (0.0, -0.70),
        6: (0.45, -0.45),
        7: (0.75, 0.0),
        8: (0.45, 0.45),
        9: (0.0, 0.22),
        10:(0.22, 0.0),
        11:(0.0, -0.22),
        12:(-0.22, 0.0),
    }

    slot_size = {
        1: 0.30, 2:0.22, 3:0.30, 4:0.22,
        5:0.30, 6:0.22, 7:0.30, 8:0.22,
        9:0.16, 10:0.16, 11:0.16, 12:0.16
    }

    # A function that returns a smooth curved diamond Path centered at (cx,cy)
    def curved_diamond_path(cx, cy, size, bulge=0.36, rotation=0.0):
        # diamond base points (top,right,bottom,left) relative to center
        pts = np.array([[0, size], [size, 0], [0, -size], [-size, 0]])
        # rotate
        c, s = math.cos(rotation), math.sin(rotation)
        R = np.array([[c, -s], [s, c]])
        pts = (pts @ R.T) + np.array([cx, cy])

        verts = []
        codes = []
        n = len(pts)
        for i in range(n):
            p0 = pts[i]
            p1 = pts[(i + 1) % n]
            mid = 0.5 * (p0 + p1)
            center = np.array([cx, cy])
            out = mid - center
            norm = np.linalg.norm(out)
            if norm == 0:
                outn = np.array([0.0, 0.0])
            else:
                outn = out / norm * (size * bulge)
            # cubic control points toward midpoint+outn
            cp1 = p0 + 0.36 * (mid + outn - p0)
            cp2 = p1 + 0.36 * (mid + outn - p1)
            if i == 0:
                verts.append(tuple(p0)); codes.append(Path.MOVETO)
            verts.append(tuple(cp1)); codes.append(Path.CURVE4)
            verts.append(tuple(cp2)); codes.append(Path.CURVE4)
            verts.append(tuple(p1)); codes.append(Path.CURVE4)
        return Path(verts, codes)

    # ---------------------------
    # Draw dual border frame (outer orange + inner red) with rounded corners
    # ---------------------------
    # large FancyBboxPatch for rounded external border (orange) and inner red stroke
    frame_pad = 1.06
    bbox = FancyBboxPatch(
        (-frame_pad, -0.95), 2*frame_pad, 1.9,
        boxstyle="round,pad=0.25,rounding_size=0.35",
        linewidth=9, edgecolor=ORANGE, facecolor=BG, zorder=0
    )
    ax.add_patch(bbox)
    bbox2 = FancyBboxPatch(
        (-frame_pad+0.02, -0.95+0.02), 2*(frame_pad-0.02), 1.9-0.04,
        boxstyle="round,pad=0.20,rounding_size=0.31",
        linewidth=4, edgecolor=RED, facecolor="none", zorder=1
    )
    ax.add_patch(bbox2)

    # ---------------------------
    # Draw houses (orange outer stroke + red inner stroke)
    # ---------------------------
    for s in range(1, 13):
        cx, cy = slot_centers[s]
        size = slot_size[s]
        # rotate tips to point inward roughly
        rot = math.atan2(-cy, -cx) if (cx != 0 or cy != 0) else 0.0
        path = curved_diamond_path(cx, cy, size, bulge=0.36, rotation=rot)
        # outer orange glossy stroke
        p_outer = PathPatch(path, facecolor=BG, edgecolor=ORANGE, linewidth=6.0, zorder=2)
        ax.add_patch(p_outer)
        # inner red stroke
        p_inner = PathPatch(path, facecolor=BG, edgecolor=RED, linewidth=1.8, zorder=3)
        ax.add_patch(p_inner)

    # cross connecting lines (thin red) to form the inner X and plus like sample
    ax.plot([-0.8, 0.8], [0, 0], color=RED, lw=1.6, zorder=4)
    ax.plot([0, 0], [-0.8, 0.8], color=RED, lw=1.6, zorder=4)
    ax.plot([-0.55, 0.55], [0.55, -0.55], color=RED, lw=1.2, zorder=4)
    ax.plot([-0.55, 0.55], [-0.55, 0.55], color=RED, lw=1.2, zorder=4)

    # ---------------------------
    # Map ascendant -> rashi numbers in north chart order
    # ---------------------------
    asc_sign = int(ascendant_deg // 30) + 1
    slot_sign = {}
    for slot in range(1, 13):
        # slot 1 gets asc_sign, slot 2 next sign, etc.
        slot_sign[slot] = ((asc_sign + (slot - 1) - 1) % 12) + 1

    # draw rashi numbers (bold red)
    for s, (cx, cy) in slot_centers.items():
        ax.text(cx, cy + slot_size[s] * 0.22, str(slot_sign[s]),
                color=RASHI_TEXT, fontsize=18, fontweight='bold', ha='center', va='center', zorder=6)

    # ---------------------------
    # Helpers for planet label formatting and nakshatra short names
    # ---------------------------
    planet_short = {
        "Sun": "Su", "Moon":"Mo", "Mercury":"Me", "Venus":"Ve",
        "Mars":"Ma", "Jupiter":"Ju", "Saturn":"Sa", "Rahu":"Ra",
        "Ketu":"Ke", "Uranus":"Ur", "Neptune":"Ne", "Pluto":"Pl"
    }

    # Full nakshatra names (27), short 3-letter approximations used in many kundli prints.
    nak_full = [
        "Ashwini","Bharani","Krittika","Rohini","Mrigashira","Ardra","Punarvasu","Pushya","Ashlesha",
        "Magha","Purva Phalguni","Uttara Phalguni","Hasta","Chitra","Swati","Vishakha","Anuradha",
        "Jyeshtha","Mula","Purva Ashadha","Uttara Ashadha","Shravana","Dhanishta","Shatabhisha",
        "Purva Bhadrapada","Uttara Bhadrapada","Revati"
    ]
    # 3-letter short forms chosen to match common printed abbreviations
    nak_short = ["Ash","Bha","Kri","Roh","Mrg","Ard","Pun","Pus","Ash","Mag","Pph","Uph","Has","Chi","Swa","Vis","Anu","Jye","Mul","Paa","Uaa","Shr","Dha","Sha","Pbh","Ubh","Rev"]

    def deg_min_str(deg_in_sign):
        # deg_in_sign is 0..30 float
        d = int(math.floor(deg_in_sign))
        m = int(round((deg_in_sign - d) * 60))
        # handle rounding up to 60
        if m == 60:
            d += 1
            m = 0
        return f"{d:02d}:{m:02d}"

    def get_nak_short(lon):
        # lon in degrees 0..360
        lon_mod = lon % 360.0
        idx = int(lon_mod / (360.0 / 27.0))  # 0..26
        idx = max(0, min(26, idx))
        return nak_short[idx]

    # ---------------------------
    # Place planets inside each house; left-aligned style similar to reference
    # ---------------------------
    # Build mapping of slot -> list of planet label strings
    slot_planets = {s: [] for s in range(1,13)}
    for pname, lon in planets_dict.items():
        if lon is None:
            continue
        sign = int(lon // 30) + 1
        target_slot = next((s for s, signn in slot_sign.items() if signn == sign), None)
        if target_slot is None:
            continue
        deg_in_sign = lon % 30
        degmin = deg_min_str(deg_in_sign)
        nak = get_nak_short(lon)
        short = planet_short.get(pname, pname[:2])
        label = f"{short} {degmin} {nak}"
        slot_planets[target_slot].append(label)

    # small offset positions for left-aligned multi-line planet lists per slot (tweak for look)
    for s in range(1,13):
        labels = slot_planets[s]
        if not labels:
            continue
        cx, cy = slot_centers[s]
        # choose alignment offsets to match printed style: left of center for many slots
        # We'll offset horizontally depending on quadrant
        if cx < -0.1:
            ha = 'left'
            x_off = cx - slot_size[s]*0.18
        elif cx > 0.1:
            ha = 'right'
            x_off = cx + slot_size[s]*0.18
        else:
            ha = 'center'
            x_off = cx
        # vertical start
        start_y = cy + slot_size[s] * 0.06
        for i, lab in enumerate(labels):
            y = start_y - i * 0.095
            ax.text(x_off, y, lab, color=PLANET_TEXT, fontsize=10, ha=ha, va='center', zorder=8)

    # Ascendant mark prominent
    asc_slot = next((s for s, signn in slot_sign.items() if signn == asc_sign), None)
    if asc_slot:
        acx, acy = slot_centers[asc_slot]
        ax.text(acx, acy - slot_size[asc_slot]*0.42, "As", color=ASC_TEXT, fontsize=14, fontweight='bold', ha='center', va='center', zorder=9)

    # Title and name (top center)
    ax.text(0, 0.94, "Janma Kundli", color=RED, fontsize=26, fontweight='bold', ha='center', va='center', zorder=10)
    ax.text(0, 0.88, name, color=RED, fontsize=12, ha='center', va='center', zorder=10)

    # Footer info (small)
    ax.text(0, -0.95, info_text, color=DEEP, fontsize=9, ha='center', va='center', zorder=10)

    ax.set_xlim(-1.12, 1.12)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect('equal', 'box')

    # Save to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor(), dpi=300)
    plt.close(fig)
    buf.seek(0)
    return buf




# -----------------------------
# Generate Kundli Endpoint
# -----------------------------
@app.route("/generate_kundli", methods=["POST"])
def generate_kundli():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "Missing JSON body"}), 400

        name = data.get("name", "Unknown")
        dob = data.get("dob")
        tob = data.get("tob")
        place = data.get("place")

        if not dob or not tob or not place:
            return jsonify({"error": "Please provide dob, tob and place fields"}), 400

        loc = get_location(place)
        if not loc:
            return jsonify({"error": "Could not geocode the place."}), 400

        lat = float(loc["lat"])
        lon = float(loc["lon"])
        display_name = loc.get("display_name", place)
        app.logger.info(f"Resolved place => {display_name} ({lat:.6f},{lon:.6f})")

        tf = TimezoneFinder()
        tzname = tf.timezone_at(lng=lon, lat=lat)
        if not tzname:
            return jsonify({"error": "Could not determine timezone"}), 400
        tz = pytz.timezone(tzname)

        # parse datetime
        dt = None
        parse_attempts = [
            "%d %b %Y %I:%M %p",
            "%d %B %Y %I:%M %p",
            "%Y-%m-%d %H:%M",
            "%d-%m-%Y %H:%M"
        ]
        combined = f"{dob} {tob}"
        for fmt in parse_attempts:
            try:
                dt = datetime.strptime(combined, fmt)
                break
            except ValueError:
                continue
        if dt is None:
            return jsonify({"error": f"Could not parse date/time: '{combined}'"}), 400

        local_dt = tz.localize(dt)
        utc_dt = local_dt.astimezone(pytz.utc)

        jd_ut = swe.julday(
            utc_dt.year, utc_dt.month, utc_dt.day,
            utc_dt.hour + utc_dt.minute / 60.0 + utc_dt.second / 3600.0
        )

        try:
            cusps, ascmc = swe.houses(jd_ut, lat, lon)
            ascendant = float(ascmc[0]) % 360.0
        except Exception:
            ascmc = swe.houses_ex(jd_ut, lat, lon, b'P')[1]
            ascendant = float(ascmc[0]) % 360.0

        planet_codes = [
            (swe.SUN, "Sun"), (swe.MOON, "Moon"), (swe.MERCURY, "Mercury"),
            (swe.VENUS, "Venus"), (swe.MARS, "Mars"), (swe.JUPITER, "Jupiter"),
            (swe.SATURN, "Saturn"), (swe.URANUS, "Uranus"),
            (swe.NEPTUNE, "Neptune"), (swe.PLUTO, "Pluto"),
            (swe.TRUE_NODE, "Rahu"),
        ]
        planets = {}
        for code, pname in planet_codes:
            lon_deg = safe_calc_longitude(jd_ut, code)
            planets[pname] = lon_deg

        info_text = f"Generated by AstroApp • {display_name.split(',')[0]} • {local_dt.strftime('%d %b %Y, %I:%M %p %Z')}"
        buf = draw_connected_north_kundli(name, ascendant, planets, info_text=info_text)
        return send_file(buf, mimetype='image/png')

    except Exception as e:
        app.logger.exception("Unexpected error in /generate_kundli")
        return jsonify({"error": "Internal server error", "detail": str(e)}), 500

# -----------------------------
# Run Flask
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
