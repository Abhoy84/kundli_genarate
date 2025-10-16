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
    # Theme colors
    bg_color = "#070014"      # deep purple/black
    outer_line = "#ff9f1c"    # warm orange for shiny border
    inner_line = "#d78cff"    # purple accent
    inner_fill = "#12001a"    # slightly lighter inner fill
    text_color = "#ffffff"
    rashi_color = "#e9d7ff"
    asc_color = "#ffd86b"
    footer_color = "#d6c9ff"

    fig, ax = plt.subplots(figsize=(8,8), dpi=150)
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    ax.axis("off")

    # Positions roughly matching the reference curved-kundli
    outer_radius = 0.62
    inner_radius = 0.30
    big_size = 0.30
    small_size = 0.18

    slot_centers = {
        1: (0.0,  outer_radius),
        2: (-inner_radius,  inner_radius),
        3: (-outer_radius,  0.0),
        4: (-inner_radius, -inner_radius),
        5: (0.0, -outer_radius),
        6: ( inner_radius, -inner_radius),
        7: ( outer_radius,  0.0),
        8: ( inner_radius,  inner_radius),
        9: (0.0,  inner_radius*0.36),
        10:( inner_radius*0.36, 0.0),
        11:(0.0, -inner_radius*0.36),
        12:(-inner_radius*0.36, 0.0),
    }

    slot_size = {}
    for s in range(1,13):
        if s in (1,3,5,7):
            slot_size[s] = big_size
        elif s in (2,4,6,8):
            slot_size[s] = small_size
        else:
            slot_size[s] = small_size * 0.9

    # Draw the curved house shapes
    for s in range(1,13):
        cx, cy = slot_centers[s]
        size = slot_size[s]
        # set rotation so tip points inward
        rot = math.atan2(-cy, -cx)
        path = curved_house_path(cx, cy, size, rotation=rot, bulge=0.36 if s in (1,3,5,7) else 0.30)
        patch = PathPatch(path, facecolor=inner_fill, edgecolor=inner_line, linewidth=2.2, zorder=2)
        ax.add_patch(patch)
        # Outer glossy stroke (slightly offset stroke for that shiny orange border look)
        patch_outer = PathPatch(path, facecolor="none", edgecolor=outer_line, linewidth=4.0, alpha=0.95, zorder=1)
        ax.add_patch(patch_outer)

    # Decorative thick rounded frame (four arcs) to emulate reference border
    # left-right top-bottom arcs using bezier approximations via Path
    frame_pad = 0.96
    # Top arc (left to right)
    top_path = Path([
        (-frame_pad, 0.88),
        (-0.2, 1.08),
        (0.2, 1.08),
        (frame_pad, 0.88)
    ], [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
    top_patch = PathPatch(top_path, facecolor="none", edgecolor=outer_line, linewidth=6, zorder=0)
    ax.add_patch(top_patch)
    # Bottom arc
    bot_path = Path([
        (-frame_pad, -0.88),
        (-0.2, -1.08),
        (0.2, -1.08),
        (frame_pad, -0.88)
    ], [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
    bot_patch = PathPatch(bot_path, facecolor="none", edgecolor=outer_line, linewidth=6, zorder=0)
    ax.add_patch(bot_patch)
    # Left vertical arc
    left_path = Path([
        (-0.88, frame_pad),
        (-1.08, 0.2),
        (-1.08, -0.2),
        (-0.88, -frame_pad)
    ], [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
    left_patch = PathPatch(left_path, facecolor="none", edgecolor=outer_line, linewidth=6, zorder=0)
    ax.add_patch(left_patch)
    # Right vertical arc
    right_path = Path([
        (0.88, frame_pad),
        (1.08, 0.2),
        (1.08, -0.2),
        (0.88, -frame_pad)
    ], [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4])
    right_patch = PathPatch(right_path, facecolor="none", edgecolor=outer_line, linewidth=6, zorder=0)
    ax.add_patch(right_patch)

    # Cross-connection lines (thin)
    ax.plot([-0.86, 0.86], [0,0], color=inner_line, lw=1.2, alpha=0.9, zorder=3)
    ax.plot([0,0], [-0.86,0.86], color=inner_line, lw=1.2, alpha=0.9, zorder=3)
    ax.plot([-0.66, 0.66], [0.66, -0.66], color=inner_line, lw=1.0, alpha=0.9, zorder=3)
    ax.plot([-0.66, 0.66], [-0.66, 0.66], color=inner_line, lw=1.0, alpha=0.9, zorder=3)

    # Determine ascendant sign and map slots to rashi numbers
    asc_sign = int(ascendant_deg // 30) + 1
    slot_sign = {}
    for slot in range(1,13):
        slot_sign[slot] = ((asc_sign + (slot - 1) - 1) % 12) + 1

    # Draw rashi numbers in each curved house
    for slot, (cx, cy) in slot_centers.items():
        y_off = slot_size[slot] * 0.22
        ax.text(cx, cy + y_off, str(slot_sign[slot]),
                color=rashi_color, fontsize=18, weight='bold', ha='center', va='center', family='sans-serif', zorder=6)

    # Prepare planets per slot
    slot_planets = {s: [] for s in range(1,13)}
    for pname, lon in planets_dict.items():
        if lon is None:
            continue
        sign_of_planet = int(lon // 30) + 1
        target_slot = next((s for s, signn in slot_sign.items() if signn == sign_of_planet), None)
        if target_slot is None:
            continue
        deg_in_sign = lon % 30
        label = f"{pname} {deg_in_sign:.1f}°"
        slot_planets[target_slot].append(label)

    # Render planet labels inside houses
    for slot, labels in slot_planets.items():
        if not labels:
            continue
        cx, cy = slot_centers[slot]
        start_y = cy - slot_size[slot]*0.05
        spacing = 0.085 if len(labels) <= 2 else 0.075
        for i, txt in enumerate(labels):
            y = start_y - i * spacing
            fs = 11 if len(labels) <= 2 else 9
            ax.text(cx, y, txt, color=text_color, fontsize=fs, ha='center', va='center', family='sans-serif', zorder=8)

    # Mark Ascendant prominently
    asc_slot = next((s for s, signn in slot_sign.items() if signn == asc_sign), None)
    if asc_slot:
        acx, acy = slot_centers[asc_slot]
        ax.text(acx, acy - slot_size[asc_slot]*0.45, "As", color=asc_color, fontsize=14, weight='bold', ha='center', va='center', zorder=9)

    # Title and name
    ax.text(0, 0.98, "Janma Kundli", color="#f5e9ff", fontsize=22, weight='bold', ha='center', va='center', zorder=10)
    ax.text(0, 0.92, name, color="#f0e0ff", fontsize=14, ha='center', va='center', zorder=10)

    # Footer info
    footer = info_text if info_text else "Generated by AstroApp"
    ax.text(0, -0.98, footer, color=footer_color, fontsize=9, ha='center', va='center', alpha=0.95, zorder=10)

    # Final layout
    ax.set_xlim(-1.12, 1.12)
    ax.set_ylim(-1.12, 1.12)
    ax.set_aspect('equal', 'box')

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
