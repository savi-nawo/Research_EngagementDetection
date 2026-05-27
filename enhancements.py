import json

class ZoneEngine:
    def __init__(self, zones_path):
        with open(zones_path, "r") as f:
            raw = json.load(f)
        self.zones = {
            name: [(int(x), int(y)) for x, y in pts]
            for name, pts in raw.items()
        }

    def _point_in_poly(self, x, y, poly):
        inside = False
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            if ((y1 > y) != (y2 > y)):
                xinters = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-6) + x1
                if x < xinters:
                    inside = not inside
        return inside

    def get_zone(self, cx, cy):
        for zone, poly in self.zones.items():
            if self._point_in_poly(cx, cy, poly):
                return zone
        return "unknown"


def zone_adjust_engagement(engaged, zone, dwell_frames):
    """
    IMPORTANT:
    - Does NOT change engagement detection logic
    - Only adjusts interpretation
    """
    if not engaged:
        return False

    # Only boost/reduce AFTER engaged=True
    if zone == "kiosk" and dwell_frames >= 45:   # ~1.5s at 30 FPS
        return True
    elif zone == "walkway":
        return False
    else:
        return engaged
