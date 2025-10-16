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
def draw_connected_north_kundli(name, ascendant_deg, planets_dict, info_text=""):
    """
    Draws North-Indian Kundli in bright traditional style (cream background, red/orange borders).
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path
    import numpy as np
    import math
    import io

    # --- Color theme ---
    bg_color = "#fff6d5"        # light cream background
    outer_line = "#ff6a00"      # orange outline
    inner_line = "#cc0000"      # deep red border
    text_color = "#b30000"      # dark red for all text
    rashi_color = "#b30000"     # same for numbers
    asc_color = "#b30000"       # same color for ascendant label
    footer_color = "#800000"

    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    ax.axis("off")

    # ----------------------
    # Geometry of 12 houses
    # ----------------------
    outer_radius = 1.0
    inner_radius = 0.55
    small_radius = 0.32

    slot_centers = {
        1: (0.0,  inner_radius),
        2: (-small_radius,  small_radius),
        3: (-inner_radius,  0.0),
        4: (-small_radius, -small_radius),
        5: (0.0, -inner_radius),
        6: ( small_radius, -small_radius),
        7: ( inner_radius,  0.0),
        8: ( small_radius,  small_radius),
        9: (0.0,  small_radius*0.38),
        10:( small_radius*0.38, 0.0),
        11:(0.0, -small_radius*0.38),
        12:(-small_radius*0.38, 0.0),
    }

    slot_size = {
        1: 0.33, 2: 0.25, 3: 0.33, 4: 0.25,
        5: 0.33, 6: 0.25, 7: 0.33, 8: 0.25,
        9: 0.20, 10: 0.20, 11: 0.20, 12: 0.20
    }

    # ----------------------
    # Helper to draw curved diamond path
    # ----------------------
    def curved_diamond(cx, cy, size, bulge=0.32, rotation=0):
        pts = np.array([
            [0, size],
            [size, 0],
            [0, -size],
            [-size, 0]
        ])
        # rotate
        c, s = math.cos(rotation), math.sin(rotation)
        R = np.array([[c, -s], [s, c]])
        pts = pts @ R.T + np.array([cx, cy])
        verts, codes = [], []
        n = len(pts)
        for i in range(n):
            p0 = pts[i]
            p1 = pts[(i + 1) % n]
            mid = 0.5 * (p0 + p1)
            center = np.array([cx, cy])
            out = mid - center
            norm = np.linalg.norm(out)
            if norm == 0:
                outn = np.array([0, 0])
            else:
                outn = out / norm * (size * bulge)
            cp1 = p0 + 0.35 * (mid + outn - p0)
            cp2 = p1 + 0.35 * (mid + outn - p1)
            if i == 0:
                verts.append(tuple(p0))
                codes.append(Path.MOVETO)
            verts.append(tuple(cp1)); codes.append(Path.CURVE4)
            verts.append(tuple(cp2)); codes.append(Path.CURVE4)
            verts.append(tuple(p1)); codes.append(Path.CURVE4)
        return Path(verts, codes)

    # ----------------------
    # Draw all 12 houses
    # ----------------------
    for s in range(1, 13):
        cx, cy = slot_centers[s]
        path = curved_diamond(cx, cy, slot_size[s], bulge=0.33)
        # orange outer border
        ax.add_patch(PathPatch(path, facecolor=bg_color, edgecolor=outer_line, lw=5, zorder=1))
        # red inner border
        ax.add_patch(PathPatch(path, facecolor=bg_color, edgecolor=inner_line, lw=1.5, zorder=2))

    # ----------------------
    # Cross lines inside
    # ----------------------
    ax.plot([-inner_radius, inner_radius], [0, 0], color=inner_line, lw=1.2)
    ax.plot([0, 0], [-inner_radius, inner_radius], color=inner_line, lw=1.2)
    ax.plot([-inner_radius * 0.7, inner_radius * 0.7], [inner_radius * 0.7, -inner_radius * 0.7], color=inner_line, lw=1.0)
    ax.plot([-inner_radius * 0.7, inner_radius * 0.7], [-inner_radius * 0.7, inner_radius * 0.7], color=inner_line, lw=1.0)

    # ----------------------
    # Map signs and planets
    # ----------------------
    asc_sign = int(ascendant_deg // 30) + 1
    slot_sign = {}
    for slot in range(1, 13):
        slot_sign[slot] = ((asc_sign + slot - 1 - 1) % 12) + 1

    # Sign numbers
    for s, (cx, cy) in slot_centers.items():
        ax.text(cx, cy + slot_size[s] * 0.2, str(slot_sign[s]),
                color=rashi_color, fontsize=15, ha='center', va='center', fontweight='bold')

    # Planets by slot
    slot_planets = {s: [] for s in range(1, 13)}
    for pname, lon in planets_dict.items():
        if lon is None:
            continue
        sign = int(lon // 30) + 1
        target = next((s for s, val in slot_sign.items() if val == sign), None)
        if target:
            deg = lon % 30
            slot_planets[target].append(f"{pname} {deg:.1f}°")

    for s, labels in slot_planets.items():
        if not labels:
            continue
        cx, cy = slot_centers[s]
        for i, txt in enumerate(labels):
            ax.text(cx, cy - 0.07 * i, txt, color=text_color, fontsize=10,
                    ha='center', va='center')

    # Ascendant mark
    asc_slot = next((s for s, signn in slot_sign.items() if signn == asc_sign), None)
    if asc_slot:
        acx, acy = slot_centers[asc_slot]
        ax.text(acx, acy, "As", color=asc_color, fontsize=13, fontweight='bold', ha='center', va='center')

    # Title & footer
    ax.text(0, 1.12, "Janma Kundli", color=inner_line, fontsize=20, weight='bold', ha='center')
    ax.text(0, 1.05, name, color=inner_line, fontsize=12, ha='center')
    ax.text(0, -1.10, info_text, color=footer_color, fontsize=9, ha='center')

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal", "box")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", facecolor=bg_color, dpi=300)
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
