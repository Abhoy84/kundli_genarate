# app.py
from flask import Flask, request, jsonify, send_file
import swisseph as swe
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
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
    # Keep it simple: Nominatim fallback only; you may plug in Geoapify here if you want.
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
# Utility: create diamond polygon (4 points) centered at (cx,cy)
# -----------------------------
def diamond_polygon(cx, cy, size, rotation=0.0):
    # diamond points (up, right, down, left) then rotate
    pts = np.array([
        [0.0, size],
        [size, 0.0],
        [0.0, -size],
        [-size, 0.0],
    ])
    c, s = math.cos(rotation), math.sin(rotation)
    R = np.array([[c, -s], [s, c]])
    rotated = pts.dot(R.T) + np.array([cx, cy])
    return rotated.tolist()

# -----------------------------
# Draw connected North-Indian Kundli chart
# -----------------------------
def draw_connected_north_kundli(name, ascendant_deg, planets_dict):
    # Colors requested
    bg_color = "#0b0b1a"       # blacky purple background
    line_color = "#8e44ad"     # bluish-purple lines
    planet_color = "#ffffff"   # white planet names
    rashi_color = "#8b5a2b"    # brown rashi numbers
    title_color = "#9b59b6"    # light purple

    fig, ax = plt.subplots(figsize=(8,8))
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    ax.axis("off")

    # Geometry - connected diamond grid (approximate, tuned visually)
    # Outer big diamonds centers
    outer_radius = 0.60
    inner_radius = 0.30
    big_size = 0.28    # size for outer corner diamonds
    small_size = 0.18  # size for inner diamonds

    # We'll put diamonds in these 12 logical slots (physical positions):
    # Slot order (physical): 1 top, 2 top-left, 3 left, 4 bottom-left, 5 bottom, 6 bottom-right,
    # 7 right, 8 top-right, 9 center-top (small), 10 center-right, 11 center-bottom, 12 center-left
    slot_centers = {
        1: (0.0,  outer_radius),        # top
        2: (-inner_radius,  inner_radius), # top-left inner
        3: (-outer_radius,  0.0),       # left
        4: (-inner_radius, -inner_radius), # bottom-left inner
        5: (0.0, -outer_radius),        # bottom
        6: ( inner_radius, -inner_radius), # bottom-right inner
        7: ( outer_radius,  0.0),       # right
        8: ( inner_radius,  inner_radius),  # top-right inner
        9: (0.0,  inner_radius*0.38),   # center-top (small)
        10:( inner_radius*0.38, 0.0),   # center-right
        11:(0.0, -inner_radius*0.38),   # center-bottom
        12:(-inner_radius*0.38, 0.0),   # center-left
    }

    # Diamonds sizes: outer corner ones bigger to match connected look
    slot_size = {}
    for s in range(1,13):
        if s in (1,3,5,7):    # top, left, bottom, right - bigger outer diamonds
            slot_size[s] = big_size
        elif s in (2,4,6,8):  # inner corner diamonds
            slot_size[s] = small_size
        else:                 # central small diamonds
            slot_size[s] = small_size * 0.9

    # Rotation so diamond points towards center: compute angle to center
    slot_rotation = {}
    for s, (cx,cy) in slot_centers.items():
        slot_rotation[s] = math.atan2(-cy, -cx)  # point inward

    # Draw the diamonds (polygons) with an inner border to create the connected look
    for s in range(1,13):
        cx, cy = slot_centers[s]
        size = slot_size[s]
        rot = slot_rotation[s]
        pts = diamond_polygon(cx, cy, size, rot)
        poly = Polygon(pts, closed=True, edgecolor=line_color, facecolor=bg_color, linewidth=2.6)
        ax.add_patch(poly)
        # inner thin border for the "double-line" look
        inner_pts = diamond_polygon(cx, cy, size*0.86, rot)
        poly2 = Polygon(inner_pts, closed=True, edgecolor=line_color, facecolor=bg_color, linewidth=1.2)
        ax.add_patch(poly2)

    # Add the cross lines to ensure it visually matches connected grid
    ax.plot([-0.9, 0.9], [0,0], color=line_color, lw=1.6, alpha=0.95)
    ax.plot([0,0], [-0.9,0.9], color=line_color, lw=1.6, alpha=0.95)
    ax.plot([-0.7, 0.7], [0.7, -0.7], color=line_color, lw=1.2, alpha=0.95)
    ax.plot([-0.7, 0.7], [-0.7, 0.7], color=line_color, lw=1.2, alpha=0.95)

    # Compute ascendant sign number (1..12) where Aries=1 ... Pisces=12
    asc_sign = int(ascendant_deg // 30) + 1

    # Map physical slot -> sign number (rashi) using anticlockwise order starting from slot 1 (top)
    # slot 1 gets asc_sign, slot 2 gets asc_sign+1, etc.
    slot_sign = {}
    for slot in range(1,13):
        slot_sign[slot] = ((asc_sign + (slot - 1) - 1) % 12) + 1

    # Place Rashi number inside each diamond (brown)
    for slot, (cx, cy) in slot_centers.items():
        # Slight upward offset for top text placement
        y_off = slot_size[slot] * 0.22
        ax.text(cx, cy + y_off, str(slot_sign[slot]),
                color=rashi_color, fontsize=18, weight='bold', ha='center', va='center', family='sans-serif')

    # Prepare planets per slot based on their sign
    slot_planets = {s: [] for s in range(1,13)}
    for pname, lon in planets_dict.items():
        if lon is None:
            continue
        sign_of_planet = int(lon // 30) + 1
        # find slot with that sign (we built slot_sign map)
        target_slot = None
        for s, signn in slot_sign.items():
            if signn == sign_of_planet:
                target_slot = s
                break
        if target_slot is None:
            continue
        deg_in_sign = lon % 30
        # Use short names for common planets (optional)
        short = pname
        # label: name and degree
        label = f"{short} {deg_in_sign:.1f}°"
        slot_planets[target_slot].append(label)

    # Render planet texts inside diamonds, stacked. Use white color.
    for slot, labels in slot_planets.items():
        if not labels:
            continue
        cx, cy = slot_centers[slot]
        # start below the rashi number
        start_y = cy - slot_size[slot]*0.06
        line_spacing = 0.08 if len(labels) == 1 else 0.085
        for i, txt in enumerate(labels):
            y = start_y - i * line_spacing
            # smaller font for multiple planets
            fontsize = 10 if len(labels) <= 2 else 9
            ax.text(cx, y, txt, color=planet_color, fontsize=fontsize, ha='center', va='center', family='sans-serif')

    # Title
    ax.set_title(f"Kundli — {name}", color=title_color, pad=18, fontsize=20, weight='bold')

    # Layout limits to nicely crop
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect('equal', 'box')

    # Save to buffer
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
        dob = data.get("dob")   # e.g., "22 Oct 2003"
        tob = data.get("tob")   # e.g., "08:05 PM"
        place = data.get("place")

        if not dob or not tob or not place:
            return jsonify({"error": "Please provide dob, tob and place fields"}), 400

        # 1) Geocode
        loc = get_location(place)
        if not loc:
            return jsonify({"error": "Could not geocode the place. Try again or set GEOAPIFY_KEY."}), 400

        lat = float(loc["lat"])
        lon = float(loc["lon"])
        display_name = loc.get("display_name", place)
        app.logger.info(f"Resolved place => {display_name} ({lat:.6f},{lon:.6f})")

        # 2) Timezone
        tf = TimezoneFinder()
        tzname = tf.timezone_at(lng=lon, lat=lat)
        if not tzname:
            return jsonify({"error": "Could not determine timezone for the given place"}), 400
        tz = pytz.timezone(tzname)
        app.logger.info(f"Timezone: {tzname}")

        # 3) Parse datetime
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

        # Localize and convert to UTC
        local_dt = tz.localize(dt)
        utc_dt = local_dt.astimezone(pytz.utc)
        app.logger.info(f"Local datetime: {local_dt.isoformat()} UTC: {utc_dt.isoformat()}")

        # 4) Julian Day UT
        jd_ut = swe.julday(
            utc_dt.year, utc_dt.month, utc_dt.day,
            utc_dt.hour + utc_dt.minute / 60.0 + utc_dt.second / 3600.0
        )
        app.logger.info(f"Julian Day UT: {jd_ut}")

        # 5) Calculate Ascendant & Planets
        try:
            cusps, ascmc = swe.houses(jd_ut, lat, lon)
            ascendant = float(ascmc[0]) % 360.0
        except Exception:
            try:
                ascmc = swe.houses_ex(jd_ut, lat, lon, b'P')[1]
                ascendant = float(ascmc[0]) % 360.0
            except Exception as e:
                app.logger.exception("Failed to compute ascendant/houses")
                return jsonify({"error": "Failed to compute ascendant/houses", "detail": str(e)}), 500

        # Planet list to compute
        planet_codes = [
            (swe.SUN, "Sun"),
            (swe.MOON, "Moon"),
            (swe.MERCURY, "Mercury"),
            (swe.VENUS, "Venus"),
            (swe.MARS, "Mars"),
            (swe.JUPITER, "Jupiter"),
            (swe.SATURN, "Saturn"),
            (swe.URANUS, "Uranus"),
            (swe.NEPTUNE, "Neptune"),
            (swe.PLUTO, "Pluto"),
            (swe.TRUE_NODE, "Rahu"),
        ]
        planets = {}
        for code, pname in planet_codes:
            lon_deg = safe_calc_longitude(jd_ut, code)
            planets[pname] = lon_deg

        # 6) Draw connected North Indian Kundli chart
        buf = draw_connected_north_kundli(name, ascendant, planets)
        return send_file(buf, mimetype='image/png')

    except Exception as e:
        app.logger.exception("Unexpected error in /generate_kundli")
        return jsonify({"error": "Internal server error", "detail": str(e)}), 500

# -----------------------------
# Run Flask
# -----------------------------
if __name__ == "__main__":
    # Note: set debug=False in production
    app.run(host="0.0.0.0", port=8000, debug=True)
