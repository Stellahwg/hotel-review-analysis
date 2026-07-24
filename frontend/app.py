import gradio as gr
import json
import os

# ── Palette (from dataviz skill reference) ──────────────────────────────────
C_POSITIVE   = "#1baf7a"   # aqua / good
C_NEGATIVE   = "#eb6834"   # orange / bad
C_NEUTRAL    = "#eda100"   # yellow / neutral
C_BAR_DARK   = "#8c5e4a"   # warm brown for complaint bars (matches screenshot)
C_BAR_LIGHT  = "#c9a48e"
C_SURFACE    = "#f5f2ee"   # warm off-white page
C_CARD       = "#ffffff"
C_TEXT_PRI   = "#1a1208"
C_TEXT_SEC   = "#6b5c4a"
C_TEXT_MUT   = "#9c8c7a"
C_GRIDLINE   = "#e8e2da"
C_BORDER     = "rgba(90,70,50,0.12)"

# ── Mock data ────────────────────────────────────────────────────────────────
STATS = {
    "total_reviews": 1842,
    "avg_rating": 3.9,
    "negative_rate": 23,
    "neutral_rate": 19,
    "sentiment_score": 72,
    "positive_pct": 58,
    "neutral_pct": 19,
    "negative_pct": 23,
    "mom_change": "+4.2%",
}

COMPLAINTS = [
    ("Noise levels",       312),
    ("Slow check-in",      278),
    ("Room cleanliness",   241),
    ("Wi-Fi quality",      198),
    ("Breakfast options",  165),
    ("Air conditioning",   134),
    ("Parking access",     109),
    ("Staff attitude",      88),
]

WORD_CLOUD = [
    ("noisy",       52, "negative"),
    ("location",    48, "positive"),
    ("comfortable", 44, "positive"),
    ("friendly",    38, "positive"),
    ("clean",       36, "positive"),
    ("overpriced",  34, "negative"),
    ("view",        32, "positive"),
    ("slow",        30, "negative"),
    ("spacious",    28, "positive"),
    ("breakfast",   26, "positive"),
    ("staff",       25, "positive"),
    ("expensive",   24, "negative"),
    ("helpful",     23, "positive"),
    ("wifi",        22, "negative"),
    ("parking",     20, "negative"),
    ("modern",      19, "positive"),
    ("pool",        18, "positive"),
    ("dirty",       15, "negative"),
    ("outdated",    14, "negative"),
    ("quiet",       13, "positive"),
]

REVIEWS = [
    {"initials": "ML", "name": "Maria L.",  "stars": 2, "date": "Jul 19", "sentiment": "Negative",
     "text": "The room was noisy all night — couldn't sleep. Staff tried to help but nothing changed."},
    {"initials": "JK", "name": "James K.",  "stars": 5, "date": "Jul 18", "sentiment": "Positive",
     "text": "Fantastic location, spotless room, breakfast was excellent. Will return without hesitation."},
    {"initials": "FO", "name": "Fatima O.", "stars": 3, "date": "Jul 17", "sentiment": "Neutral",
     "text": "Decent stay overall, though check-in took nearly 40 minutes. Wi-Fi kept dropping."},
    {"initials": "TV", "name": "Tomáš V.",  "stars": 4, "date": "Jul 16", "sentiment": "Positive",
     "text": "Comfortable bed, great view. Air conditioning unit was a bit loud at night."},
    {"initials": "SM", "name": "Sophie M.", "stars": 1, "date": "Jul 15", "sentiment": "Negative",
     "text": "Room was not ready at 4pm. Bathroom had visible mould near the shower."},
    {"initials": "KA", "name": "Kwame A.",  "stars": 5, "date": "Jul 14", "sentiment": "Positive",
     "text": "Pool area was immaculate, staff incredibly warm. Best hotel stay this year."},
    {"initials": "IH", "name": "Ingrid H.", "stars": 2, "date": "Jul 13", "sentiment": "Negative",
     "text": "Promised a city-view room, got a wall view. Complaints were ignored."},
    {"initials": "RL", "name": "Ravi L.",   "stars": 4, "date": "Jul 12", "sentiment": "Positive",
     "text": "Great central location and friendly front desk. Breakfast variety could be better."},
    {"initials": "NP", "name": "Nora P.",   "stars": 3, "date": "Jul 11", "sentiment": "Neutral",
     "text": "Average experience. Room was clean but facilities feel dated."},
    {"initials": "OB", "name": "Omar B.",   "stars": 1, "date": "Jul 10", "sentiment": "Negative",
     "text": "AC broke on first night. Maintenance came next day but it was unbearable."},
    {"initials": "CL", "name": "Claire L.", "stars": 5, "date": "Jul 9",  "sentiment": "Positive",
     "text": "Exceptional stay from start to finish. Room was pristine, staff superb."},
    {"initials": "HM", "name": "Hamid M.",  "stars": 3, "date": "Jul 8",  "sentiment": "Neutral",
     "text": "Parking was a nightmare. Otherwise the room and service were acceptable."},
]

TOPICS = [
    {"name": "Noise levels",     "category": "Facilities",   "positive_pct": 12, "mentions": 312},
    {"name": "Slow check-in",    "category": "Service",      "positive_pct": 18, "mentions": 278},
    {"name": "Room cleanliness", "category": "Housekeeping", "positive_pct": 22, "mentions": 241},
    {"name": "Wi-Fi quality",    "category": "Facilities",   "positive_pct": 15, "mentions": 198},
    {"name": "Breakfast options","category": "F&B",          "positive_pct": 38, "mentions": 165},
    {"name": "Air conditioning", "category": "Facilities",   "positive_pct": 20, "mentions": 134},
    {"name": "Parking access",   "category": "Facilities",   "positive_pct": 10, "mentions": 109},
    {"name": "Staff attitude",   "category": "Service",      "positive_pct": 55, "mentions":  88},
    {"name": "Pool area",        "category": "Facilities",   "positive_pct": 81, "mentions":  76},
    {"name": "Room view",        "category": "Rooms",        "positive_pct": 78, "mentions":  65},
    {"name": "Bed comfort",      "category": "Rooms",        "positive_pct": 72, "mentions":  58},
    {"name": "Location",         "category": "General",      "positive_pct": 90, "mentions": 201},
]

TREND_MONTHS   = ["Feb", "Mar", "Apr", "May", "Jun", "Jul"]
TREND_SCORES   = [70,     65,    68,    67,    69,    72  ]
TREND_POSITIVE = [52,     48,    54,    53,    56,    58  ]
TREND_NEGATIVE = [28,     32,    27,    26,    24,    23  ]
TREND_VOLUME   = [120,   180,   210,   260,   310,   420  ]

RADAR_CATEGORIES = ["Cleanliness", "Service", "Facilities", "Comfort", "Value", "Location"]
RADAR_VALUES     = [0.55, 0.60, 0.45, 0.72, 0.38, 0.90]

# ── CSS ──────────────────────────────────────────────────────────────────────
CSS = """
/* ── Reset & base ── */
* { box-sizing: border-box; margin: 0; padding: 0; }
body, .gradio-container { background: #f5f2ee !important; font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
.gradio-container { max-width: 100% !important; padding: 0 !important; }
footer { display: none !important; }

/* ── Sidebar ── */
#sidebar {
  background: #1a1208;
  color: #f5f2ee;
  min-height: 100vh;
  padding: 24px 0;
  display: flex;
  flex-direction: column;
}
#sidebar .brand { padding: 0 20px 24px; border-bottom: 1px solid rgba(255,255,255,0.08); }
#sidebar .brand-title { font-size: 15px; font-weight: 700; color: #f5f2ee; letter-spacing: 0.01em; }
#sidebar .brand-sub { font-size: 10px; font-weight: 600; letter-spacing: 0.12em; color: #9c8c7a; text-transform: uppercase; margin-top: 2px; }
#sidebar nav { padding: 16px 12px; flex: 1; }
.nav-item {
  display: flex; align-items: center; gap: 10px;
  padding: 9px 12px; border-radius: 6px;
  font-size: 13px; font-weight: 500; color: #c3b9aa; cursor: pointer;
  transition: background 0.15s;
  margin-bottom: 2px;
}
.nav-item:hover  { background: rgba(255,255,255,0.06); color: #f5f2ee; }
.nav-item.active { background: rgba(255,255,255,0.10); color: #f5f2ee; }
.nav-icon { font-size: 14px; width: 18px; text-align: center; }
#sidebar .sidebar-footer { padding: 16px 20px; border-top: 1px solid rgba(255,255,255,0.08); font-size: 11px; color: #6b5c4a; }

/* ── Main content ── */
#main-content { background: #f5f2ee; padding: 0; min-height: 100vh; }

/* ── Page header ── */
.page-header { padding: 24px 28px 0; display: flex; align-items: flex-start; justify-content: space-between; }
.page-header h1 { font-size: 22px; font-weight: 700; color: #1a1208; }
.page-header .sub { font-size: 12px; color: #9c8c7a; margin-top: 3px; }
.live-badge {
  display: inline-flex; align-items: center; gap: 6px;
  background: #f5f2ee; border: 1px solid #e8e2da; border-radius: 20px;
  padding: 4px 12px; font-size: 11px; color: #6b5c4a; font-weight: 500;
}
.live-badge::before { content: ""; width: 7px; height: 7px; border-radius: 50%; background: #1baf7a; display: inline-block; }

/* ── Stat cards ── */
.stat-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; padding: 16px 28px 0; }
.stat-card {
  background: #fff; border: 1px solid #e8e2da; border-radius: 10px;
  padding: 18px 20px; position: relative;
}
.stat-label { font-size: 10px; font-weight: 700; letter-spacing: 0.10em; text-transform: uppercase; color: #9c8c7a; margin-bottom: 8px; }
.stat-value { font-size: 26px; font-weight: 700; color: #1a1208; }
.stat-sub { font-size: 11px; color: #9c8c7a; margin-top: 4px; }
.stat-icon { position: absolute; top: 18px; right: 18px; font-size: 14px; opacity: 0.5; }

/* ── Two-up panel row ── */
.panel-row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; padding: 12px 28px 0; }
.panel-full { padding: 12px 28px 0; }

/* ── Generic card ── */
.card {
  background: #fff; border: 1px solid #e8e2da; border-radius: 10px;
  padding: 20px 22px;
}
.card-title { font-size: 13px; font-weight: 700; color: #1a1208; margin-bottom: 4px; }
.card-sub   { font-size: 11px; color: #9c8c7a; margin-bottom: 16px; }

/* ── Sentiment gauge ── */
.gauge-wrap { display: flex; flex-direction: column; align-items: center; }
.gauge-badge {
  align-self: flex-end; background: #f0ede8; border-radius: 20px;
  padding: 3px 10px; font-size: 11px; font-weight: 600; color: #1baf7a;
}
.gauge-score { font-size: 38px; font-weight: 700; color: #1baf7a; margin-top: 4px; }
.gauge-label { font-size: 10px; font-weight: 700; letter-spacing: 0.10em; text-transform: uppercase; color: #9c8c7a; }
.sentiment-row { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; margin-top: 16px; }
.sent-tile {
  background: #f5f2ee; border-radius: 8px; padding: 12px 10px; text-align: center;
}
.sent-pct  { font-size: 18px; font-weight: 700; }
.sent-name { font-size: 10px; font-weight: 600; letter-spacing: 0.07em; text-transform: uppercase; color: #9c8c7a; margin-top: 2px; }
.sent-pos  { color: #1baf7a; }
.sent-neu  { color: #eda100; }
.sent-neg  { color: #eb6834; }

/* ── Bar chart ── */
.bar-chart { width: 100%; }
.bar-row { display: flex; align-items: center; gap: 8px; margin-bottom: 10px; }
.bar-name { font-size: 11px; color: #6b5c4a; text-align: right; min-width: 130px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.bar-track { flex: 1; background: #f0ede8; border-radius: 3px; height: 12px; position: relative; }
.bar-fill  { height: 100%; border-radius: 3px; }
.bar-val   { font-size: 10px; color: #9c8c7a; min-width: 28px; }

/* ── Line chart (SVG) ── */
.line-chart-wrap { width: 100%; overflow: hidden; }

/* ── Reviews list ── */
.filter-row { display: flex; gap: 6px; margin-bottom: 14px; align-items: center; }
.filter-btn {
  padding: 5px 14px; border-radius: 20px; border: 1px solid #e8e2da;
  background: #f5f2ee; font-size: 12px; font-weight: 500; color: #6b5c4a; cursor: pointer;
}
.filter-btn.active { background: #1a1208; color: #fff; border-color: #1a1208; }
.review-card {
  background: #f5f2ee; border-radius: 8px; padding: 14px 16px; margin-bottom: 8px;
}
.review-header { display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }
.avatar {
  width: 34px; height: 34px; border-radius: 50%;
  background: #e8e2da; display: flex; align-items: center; justify-content: center;
  font-size: 11px; font-weight: 700; color: #6b5c4a; flex-shrink: 0;
}
.review-name  { font-size: 13px; font-weight: 600; color: #1a1208; }
.review-stars { font-size: 12px; color: #eda100; }
.review-date  { font-size: 11px; color: #9c8c7a; margin-left: auto; }
.review-text  { font-size: 12px; color: #52514e; line-height: 1.5; }
.badge {
  padding: 2px 9px; border-radius: 20px; font-size: 10px; font-weight: 600;
  margin-left: 8px;
}
.badge-pos { background: #e8f7f1; color: #1baf7a; }
.badge-neu { background: #fef8e6; color: #eda100; }
.badge-neg { background: #fdeee8; color: #eb6834; }

/* ── Topics grid ── */
.topics-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.topic-card  { background: #f5f2ee; border-radius: 8px; padding: 14px 16px; }
.topic-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 4px; }
.topic-name   { font-size: 13px; font-weight: 600; color: #1a1208; }
.topic-cat    { font-size: 10px; color: #9c8c7a; margin-bottom: 8px; }
.topic-bar-track { background: #e8e2da; border-radius: 3px; height: 5px; }
.topic-bar-fill  { height: 100%; border-radius: 3px; }
.topic-mentions  { font-size: 10px; color: #9c8c7a; text-align: right; margin-top: 4px; }
.pct-badge {
  padding: 2px 8px; border-radius: 20px; font-size: 10px; font-weight: 600;
}
.pct-low  { background: #fdeee8; color: #eb6834; }
.pct-mid  { background: #fef8e6; color: #eda100; }
.pct-high { background: #e8f7f1; color: #1baf7a; }

/* ── Category filter tabs ── */
.cat-tabs { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 16px; }
.cat-tab {
  padding: 4px 12px; border-radius: 20px; border: 1px solid #e8e2da;
  background: #f5f2ee; font-size: 11px; font-weight: 500; color: #6b5c4a; cursor: pointer;
}
.cat-tab.active { background: #1a1208; color: #fff; border-color: #1a1208; }

/* ── Trend tabs ── */
.trend-tabs { display: flex; gap: 6px; margin-bottom: 16px; }
.trend-tab {
  padding: 5px 14px; border-radius: 20px; border: 1px solid #e8e2da;
  background: #f5f2ee; font-size: 12px; font-weight: 500; color: #6b5c4a; cursor: pointer;
}
.trend-tab.active { background: #1a1208; color: #fff; border-color: #1a1208; }
.trend-stats { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-top: 14px; }
.trend-stat-card { background: #f5f2ee; border-radius: 8px; padding: 14px 16px; }
.trend-stat-label { font-size: 10px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: #9c8c7a; margin-bottom: 4px; }
.trend-stat-value { font-size: 18px; font-weight: 700; }
.trend-stat-sub   { font-size: 11px; color: #9c8c7a; margin-top: 2px; }
.ts-best  { color: #1baf7a; }
.ts-worst { color: #eb6834; }
.ts-grow  { color: #eda100; }

/* ── Reports ── */
.report-card {
  background: #fff; border: 1px solid #e8e2da; border-radius: 8px;
  padding: 18px 20px; display: flex; align-items: flex-start; gap: 16px;
  margin-bottom: 10px;
}
.report-icon {
  width: 38px; height: 38px; border-radius: 6px; display: flex; align-items: center;
  justify-content: center; font-size: 12px; font-weight: 700; flex-shrink: 0;
  letter-spacing: 0.03em;
}
.icon-pdf { background: #fdeee8; color: #eb6834; }
.icon-csv { background: #e8f7f1; color: #1baf7a; }
.report-title { font-size: 13px; font-weight: 600; color: #1a1208; margin-bottom: 3px; }
.report-desc  { font-size: 11px; color: #9c8c7a; line-height: 1.4; margin-bottom: 6px; }
.report-meta  { font-size: 11px; color: #9c8c7a; }
.report-dl    { margin-left: auto; padding: 6px 14px; border-radius: 6px; border: 1px solid #e8e2da; background: #fff; font-size: 11px; font-weight: 600; color: #1a1208; cursor: pointer; white-space: nowrap; }
.report-note  { background: #f5f2ee; border: 1px solid #e8e2da; border-radius: 8px; padding: 10px 16px; font-size: 11px; color: #9c8c7a; margin-bottom: 16px; }

/* ── Chat widget ── */
#chat-fab {
  position: fixed; bottom: 24px; right: 24px; z-index: 1000;
  width: 48px; height: 48px; border-radius: 50%; background: #1a1208;
  display: flex; align-items: center; justify-content: center;
  cursor: pointer; box-shadow: 0 4px 16px rgba(0,0,0,0.20);
  font-size: 20px; color: #fff;
}

/* ── Search bar (Reviews) ── */
.search-bar-wrap { margin-bottom: 12px; }
.search-bar {
  width: 100%; padding: 8px 14px 8px 36px; border: 1px solid #e8e2da; border-radius: 8px;
  background: #fff; font-size: 12px; color: #1a1208; outline: none;
}
.search-wrap { position: relative; }
.search-icon { position: absolute; left: 12px; top: 50%; transform: translateY(-50%); font-size: 13px; color: #9c8c7a; }

/* ── Spacer ── */
.spacer { height: 24px; }

/* Gradio overrides */
.gr-button { display: none !important; }
.gr-box, .gr-form { background: transparent !important; border: none !important; box-shadow: none !important; }
.gr-padded { padding: 0 !important; }
#component-0 { background: #f5f2ee !important; }

/* Chat */
.chat-panel {
  background: #fff; border-radius: 12px 12px 0 0;
  box-shadow: 0 -4px 32px rgba(0,0,0,0.12);
  overflow: hidden;
}
.chat-header {
  background: #1a1208; color: #fff; padding: 14px 18px;
  display: flex; justify-content: space-between; align-items: flex-start;
}
.chat-header-title { font-size: 15px; font-weight: 700; }
.chat-header-sub   { font-size: 11px; color: #9c8c7a; margin-top: 2px; }
.chat-messages { padding: 16px; min-height: 200px; max-height: 340px; overflow-y: auto; }
.msg-bubble {
  background: #f0ede8; border-radius: 12px; padding: 12px 14px;
  font-size: 13px; color: #1a1208; line-height: 1.5; max-width: 90%;
}
.msg-user { background: #1a1208; color: #fff; margin-left: auto; max-width: 80%; }
.msg-wrap { margin-bottom: 10px; }
.msg-wrap.user { display: flex; justify-content: flex-end; }
.chat-input-row { display: flex; gap: 8px; padding: 12px 16px; border-top: 1px solid #e8e2da; }
.chat-input {
  flex: 1; padding: 8px 14px; border: 1px solid #e8e2da; border-radius: 8px;
  font-size: 12px; outline: none; background: #fafaf8;
}
.chat-send {
  width: 36px; height: 36px; border-radius: 8px; background: #c9a48e;
  border: none; cursor: pointer; display: flex; align-items: center; justify-content: center;
  font-size: 14px; color: #fff;
}

/* layout */
.layout-root { display: grid; grid-template-columns: 160px 1fr; min-height: 100vh; }
"""


# ── HTML builders ────────────────────────────────────────────────────────────

def stars_html(n):
    return "★" * n + "☆" * (5 - n)


def bar_chart_html(items, max_val=None):
    if max_val is None:
        max_val = items[0][1] if items else 1
    rows = []
    for name, val in items:
        pct = val / max_val * 100
        shade = f"#{hex(int(0x8c + (0xc9 - 0x8c) * (1 - val / max_val)))[2:].zfill(2)}"
        # gradient dark→light based on rank
        ratio = 1 - val / max_val
        r = int(0x8c + (0xc9 - 0x8c) * ratio)
        g = int(0x5e + (0xa4 - 0x5e) * ratio)
        b = int(0x4a + (0x8e - 0x4a) * ratio)
        color = f"#{r:02x}{g:02x}{b:02x}"
        rows.append(f"""
        <div class="bar-row">
          <div class="bar-name">{name}</div>
          <div class="bar-track">
            <div class="bar-fill" style="width:{pct:.1f}%;background:{color};"></div>
          </div>
          <div class="bar-val">{val}</div>
        </div>""")
    return f'<div class="bar-chart">{"".join(rows)}</div>'


def gauge_svg(score):
    import math
    cx, cy, r = 90, 85, 60
    start_angle = -180
    end_angle = 0
    filled_angle = start_angle + (end_angle - start_angle) * (score / 100)

    def pt(angle, radius=r):
        rad = math.radians(angle)
        return cx + radius * math.cos(rad), cy + radius * math.sin(rad)

    sx, sy = pt(start_angle)
    ex, ey = pt(filled_angle)

    def arc(x1, y1, x2, y2, large, color, width):
        return f'<path d="M {x1:.1f} {y1:.1f} A {r} {r} 0 {large} 1 {x2:.1f} {y2:.1f}" stroke="{color}" stroke-width="{width}" fill="none" stroke-linecap="round"/>'

    bg_x, bg_y = pt(end_angle)
    large_bg = 1
    large_fg = 1 if (filled_angle - start_angle) > 180 else 0

    needle_angle = start_angle + (end_angle - start_angle) * (score / 100)
    nx, ny = pt(needle_angle, r - 4)

    return f"""<svg width="180" height="100" viewBox="0 0 180 100">
  {arc(sx, sy, bg_x, bg_y, large_bg, "#e8e2da", 12)}
  {arc(sx, sy, ex, ey, large_fg, "#1baf7a", 12)}
  <line x1="{cx}" y1="{cy}" x2="{nx:.1f}" y2="{ny:.1f}" stroke="#1a1208" stroke-width="2" stroke-linecap="round"/>
  <circle cx="{cx}" cy="{cy}" r="4" fill="#1a1208"/>
  <text x="{cx-r-6}" y="{cy+14}" font-size="9" fill="#9c8c7a" text-anchor="middle">0</text>
  <text x="{cx+r+6}" y="{cy+14}" font-size="9" fill="#9c8c7a" text-anchor="middle">100</text>
</svg>"""


def word_cloud_html(words):
    sizes = [52, 48, 44, 38, 36, 34, 32, 30, 28, 26, 25, 24, 23, 22, 20, 19, 18, 15, 14, 13]
    spans = []
    for i, (word, freq, sentiment) in enumerate(words):
        size = max(11, min(34, 11 + (freq - 13) * 0.52))
        color = "#eb6834" if sentiment == "negative" else "#6b8c6e"
        spans.append(
            f'<span style="font-size:{size:.0f}px;color:{color};margin:3px 5px;display:inline-block;font-weight:{"700" if size > 22 else "500"}">{word}</span>'
        )
    legend = (
        '<div style="display:flex;gap:14px;margin-bottom:10px;font-size:11px;align-items:center;">'
        '<span style="display:flex;align-items:center;gap:4px;"><span style="width:8px;height:8px;border-radius:50%;background:#6b8c6e;display:inline-block;"></span> positive</span>'
        '<span style="display:flex;align-items:center;gap:4px;"><span style="width:8px;height:8px;border-radius:50%;background:#eb6834;display:inline-block;"></span> negative</span>'
        '</div>'
    )
    return legend + '<div style="line-height:1.8">' + "".join(spans) + "</div>"


def review_card_html(r, show=True):
    if not show:
        return ""
    badge_cls = {"Positive": "badge-pos", "Neutral": "badge-neu", "Negative": "badge-neg"}[r["sentiment"]]
    return f"""
<div class="review-card">
  <div class="review-header">
    <div class="avatar">{r["initials"]}</div>
    <div>
      <div class="review-name">{r["name"]}<span class="badge {badge_cls}">{r["sentiment"]}</span></div>
      <div class="review-stars">{stars_html(r["stars"])}</div>
    </div>
    <div class="review-date">{r["date"]}</div>
  </div>
  <div class="review-text">{r["text"]}</div>
</div>"""


def topic_card_html(t):
    pct = t["positive_pct"]
    if pct < 30:
        badge_cls, bar_color = "pct-low", "#eb6834"
    elif pct < 60:
        badge_cls, bar_color = "pct-mid", "#eda100"
    else:
        badge_cls, bar_color = "pct-high", "#1baf7a"
    return f"""
<div class="topic-card">
  <div class="topic-header">
    <div>
      <div class="topic-name">{t["name"]}</div>
      <div class="topic-cat">{t["category"]}</div>
    </div>
    <span class="pct-badge {badge_cls}">{pct}% positive</span>
  </div>
  <div class="topic-bar-track">
    <div class="topic-bar-fill" style="width:{min(pct,100)}%;background:{bar_color};"></div>
  </div>
  <div class="topic-mentions">{t["mentions"]} mentions</div>
</div>"""


def line_chart_svg(months, values, color="#eda100", label="Score", w=540, h=220):
    pad_l, pad_r, pad_t, pad_b = 40, 20, 20, 30
    plot_w = w - pad_l - pad_r
    plot_h = h - pad_t - pad_b
    mn, mx = min(values) - 5, max(values) + 5

    def px(i):
        return pad_l + i / (len(months) - 1) * plot_w

    def py(v):
        return pad_t + (1 - (v - mn) / (mx - mn)) * plot_h

    # grid lines
    grids = ""
    for tick in range(int(mn // 20) * 20, int(mx) + 20, 20):
        if mn <= tick <= mx + 5:
            y = py(tick)
            grids += f'<line x1="{pad_l}" y1="{y:.1f}" x2="{w - pad_r}" y2="{y:.1f}" stroke="#e8e2da" stroke-width="1"/>'
            grids += f'<text x="{pad_l - 6}" y="{y + 4:.1f}" font-size="9" fill="#9c8c7a" text-anchor="end">{tick}</text>'

    # x labels
    xlabels = ""
    for i, m in enumerate(months):
        xlabels += f'<text x="{px(i):.1f}" y="{h - 4}" font-size="9" fill="#9c8c7a" text-anchor="middle">{m}</text>'

    # polyline
    points = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(values))
    line = f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>'

    # dots
    dots = ""
    for i, v in enumerate(values):
        dots += f'<circle cx="{px(i):.1f}" cy="{py(v):.1f}" r="4" fill="{color}" stroke="#fff" stroke-width="2"/>'

    return f"""<svg width="100%" viewBox="0 0 {w} {h}" style="overflow:visible">
  {grids}{xlabels}{line}{dots}
</svg>"""


def radar_svg(categories, values, w=340, h=280):
    import math
    n = len(categories)
    cx, cy, r = w / 2, h / 2 - 10, min(w, h) * 0.35
    angles = [math.radians(-90 + 360 / n * i) for i in range(n)]

    def pt(angle, radius):
        return cx + radius * math.cos(angle), cy + radius * math.sin(angle)

    # grid rings
    rings = ""
    for frac in [0.25, 0.5, 0.75, 1.0]:
        pts = " ".join(f"{pt(a, r * frac)[0]:.1f},{pt(a, r * frac)[1]:.1f}" for a in angles)
        rings += f'<polygon points="{pts}" fill="none" stroke="#e8e2da" stroke-width="1"/>'

    # axes
    axes = ""
    for a in angles:
        x, y = pt(a, r)
        axes += f'<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" stroke="#e8e2da" stroke-width="1"/>'

    # data polygon
    data_pts = " ".join(f"{pt(angles[i], r * v)[0]:.1f},{pt(angles[i], r * v)[1]:.1f}" for i, v in enumerate(values))
    poly = f'<polygon points="{data_pts}" fill="#eda10033" stroke="#eda100" stroke-width="2"/>'

    # labels
    labels = ""
    for i, (cat, a) in enumerate(zip(categories, angles)):
        lx, ly = pt(a, r + 20)
        labels += f'<text x="{lx:.1f}" y="{ly:.1f}" font-size="10" fill="#6b5c4a" text-anchor="middle" dominant-baseline="middle">{cat}</text>'

    return f"""<svg width="100%" viewBox="0 0 {w} {h}">
  {rings}{axes}{poly}{labels}
</svg>"""


# ── Page builders ─────────────────────────────────────────────────────────────

def build_overview():
    s = STATS
    stat_row = f"""
<div class="stat-row">
  <div class="stat-card"><div class="stat-label">Total Reviews</div><div class="stat-value">{s['total_reviews']:,}</div><div class="stat-sub">{s['mom_change']} vs last month</div><div class="stat-icon">↗</div></div>
  <div class="stat-card"><div class="stat-label">Avg Rating</div><div class="stat-value">{s['avg_rating']}/5</div><div class="stat-sub">last 30 days</div><div class="stat-icon">★</div></div>
  <div class="stat-card"><div class="stat-label">Negative Rate</div><div class="stat-value">{s['negative_rate']}%</div><div class="stat-sub">of all reviews</div><div class="stat-icon" style="color:#eb6834">↘</div></div>
  <div class="stat-card"><div class="stat-label">Neutral Rate</div><div class="stat-value">{s['neutral_rate']}%</div><div class="stat-sub">of all reviews</div><div class="stat-icon">—</div></div>
</div>"""

    gauge = gauge_svg(s["sentiment_score"])
    sentiment_panel = f"""
<div class="card">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
    <div class="card-title">Overall Sentiment</div>
    <span class="gauge-badge">+4.2% MoM</span>
  </div>
  <div class="gauge-wrap">
    {gauge}
    <div class="gauge-score">{s['sentiment_score']}</div>
    <div class="gauge-label">Sentiment Score</div>
  </div>
  <div class="sentiment-row">
    <div class="sent-tile"><div class="sent-pct sent-pos">{s['positive_pct']}%</div><div class="sent-name">Positive</div></div>
    <div class="sent-tile"><div class="sent-pct sent-neu">{s['neutral_pct']}%</div><div class="sent-name">Neutral</div></div>
    <div class="sent-tile"><div class="sent-pct sent-neg">{s['negative_pct']}%</div><div class="sent-name">Negative</div></div>
  </div>
</div>"""

    wc = word_cloud_html(WORD_CLOUD)
    wc_panel = f"""
<div class="card">
  <div class="card-title">Word Cloud</div>
  <div style="margin-top:12px">{wc}</div>
</div>"""

    bar = bar_chart_html(COMPLAINTS)
    complaints_panel = f"""
<div class="card">
  <div class="card-title">Top Complaints</div>
  <div class="card-sub">by mention frequency across all reviews</div>
  {bar}
</div>"""

    recent_reviews = "".join(
        f'<div class="review-card" data-sentiment="{r["sentiment"]}">'
        f'<div class="review-header"><div class="avatar">{r["initials"]}</div>'
        f'<div><div class="review-name">{r["name"]}</div>'
        f'<div class="review-stars">{"★"*r["stars"]}{"☆"*(5-r["stars"])}</div></div>'
        f'<div class="review-date">{r["date"]}</div></div>'
        f'<div class="review-text">{r["text"]}</div></div>'
        for r in REVIEWS[:8]
    )
    def make_filter_onclick(val):
        if val == "All":
            js = ("(function(btn){"
                  "var root=btn.closest('.card');"
                  "root.querySelectorAll('.filter-btn').forEach(function(b){b.classList.remove('active')});"
                  "btn.classList.add('active');"
                  "root.querySelectorAll('#recent-list .review-card').forEach(function(c){c.style.display=''});"
                  "})(this)")
        else:
            js = ("(function(btn,v){"
                  "var root=btn.closest('.card');"
                  "root.querySelectorAll('.filter-btn').forEach(function(b){b.classList.remove('active')});"
                  "btn.classList.add('active');"
                  "root.querySelectorAll('#recent-list .review-card').forEach(function(c){c.style.display=c.dataset.sentiment===v?'':'none'});"
                  "})(this,'" + val + "')")
        return js

    filter_btns = (
        '<div class="filter-row">'
        '<button class="filter-btn active" onclick="' + make_filter_onclick("All") + '">All</button>'
        '<button class="filter-btn" onclick="' + make_filter_onclick("Positive") + '">Positive</button>'
        '<button class="filter-btn" onclick="' + make_filter_onclick("Neutral") + '">Neutral</button>'
        '<button class="filter-btn" onclick="' + make_filter_onclick("Negative") + '">Negative</button>'
        '</div>'
    )
    reviews_panel = f"""
<div class="card">
  <div class="card-title">Recent Reviews</div>
  {filter_btns}
  <div id="recent-list">{recent_reviews}</div>
</div>"""

    return f"""
{stat_row}
<div class="panel-row" style="margin-top:12px;margin-left:28px;margin-right:28px;gap:12px;display:grid;grid-template-columns:1fr 1fr;">
  {sentiment_panel}
  {wc_panel}
</div>
<div class="panel-full" style="padding:12px 28px 0;">
  {complaints_panel}
</div>
<div class="panel-full" style="padding:12px 28px 24px;">
  {reviews_panel}
</div>"""


def build_reviews(filter_val="All", search=""):
    filtered = REVIEWS
    if filter_val != "All":
        filtered = [r for r in filtered if r["sentiment"] == filter_val]
    if search:
        s = search.lower()
        filtered = [r for r in filtered if s in r["name"].lower() or s in r["text"].lower()]

    cards = "".join(review_card_html(r) for r in filtered)
    sort_row = f"""
<div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
  <div class="search-wrap" style="flex:1;position:relative;">
    <span class="search-icon">🔍</span>
    <input class="search-bar" placeholder="Search reviews or authors…" value="{search}"/>
  </div>
  <select style="padding:7px 10px;border:1px solid #e8e2da;border-radius:8px;font-size:12px;background:#fff;color:#6b5c4a;">
    <option>All</option><option>Positive</option><option>Neutral</option><option>Negative</option>
  </select>
  <select style="padding:7px 10px;border:1px solid #e8e2da;border-radius:8px;font-size:12px;background:#fff;color:#6b5c4a;">
    <option>Sort: Date</option><option>Sort: Rating</option>
  </select>
</div>"""

    return f"""
<div style="padding:0 28px 24px;">
  <p style="font-size:12px;color:#9c8c7a;margin-bottom:14px;">All guest feedback · {len(REVIEWS):,} entries</p>
  {sort_row}
  <p style="font-size:11px;color:#9c8c7a;margin-bottom:10px;">{len(filtered)} reviews</p>
  {cards if cards else '<p style="color:#9c8c7a;font-size:12px;">No reviews match.</p>'}
</div>"""


def build_topics(category="All"):
    items = TOPICS if category == "All" else [t for t in TOPICS if t["category"] == category]
    cards = "".join(topic_card_html(t) for t in items)
    radar = radar_svg(RADAR_CATEGORIES, RADAR_VALUES)

    return f"""
<div style="padding:0 28px 24px;">
  <p style="font-size:12px;color:#9c8c7a;margin-bottom:14px;">Complaint &amp; praise category breakdown</p>
  <div class="topics-grid">{cards}</div>
  <div class="card" style="margin-top:14px;">
    <div class="card-title">Category Radar</div>
    <div style="margin-top:8px;">{radar}</div>
  </div>
</div>"""


def build_trends(tab="Sentiment Score"):
    if tab == "Sentiment Score":
        chart = line_chart_svg(TREND_MONTHS, TREND_SCORES, color="#eda100", label="Score")
        title = "Sentiment Score Over Time"
    elif tab == "Pos / Neg Split":
        chart = line_chart_svg(TREND_MONTHS, TREND_POSITIVE, color="#1baf7a", label="Positive %")
        title = "Positive % Over Time"
    else:
        chart = line_chart_svg(TREND_MONTHS, TREND_VOLUME, color="#2a78d6", label="Reviews")
        title = "Review Volume Over Time"

    stat_cards = f"""
<div class="trend-stats">
  <div class="trend-stat-card">
    <div class="trend-stat-label">Best Month</div>
    <div class="trend-stat-value ts-best">July 2026</div>
    <div class="trend-stat-sub">score: 72</div>
  </div>
  <div class="trend-stat-card">
    <div class="trend-stat-label">Worst Month</div>
    <div class="trend-stat-value ts-worst">March 2026</div>
    <div class="trend-stat-sub">score: 65</div>
  </div>
  <div class="trend-stat-card">
    <div class="trend-stat-label">Growth</div>
    <div class="trend-stat-value ts-grow">+120%</div>
    <div class="trend-stat-sub">Feb → Jul volume</div>
  </div>
</div>"""

    return f"""
<div style="padding:0 28px 24px;">
  <p style="font-size:12px;color:#9c8c7a;margin-bottom:14px;">Sentiment over time · Feb – Jul 2026</p>
  <div class="card">
    <div class="card-title">{title}</div>
    <div class="card-sub">February – July 2026</div>
    {chart}
  </div>
  {stat_cards}
</div>"""


def build_reports():
    reports = [
        {"type": "PDF", "title": "Monthly Sentiment Report",
         "desc": "Full breakdown of July 2026 sentiment scores, complaint categories, and guest feedback patterns.",
         "date": "Jul 23, 2026", "size": "2.4 MB"},
        {"type": "PDF", "title": "Top Complaints Analysis",
         "desc": "Deep-dive into the 8 most frequent complaint categories with root cause hypotheses and recommended actions.",
         "date": "Jul 20, 2026", "size": "1.1 MB"},
        {"type": "CSV", "title": "Sentiment Trend Data",
         "desc": "Raw monthly sentiment and volume data from February through July 2026, suitable for further analysis.",
         "date": "Jul 23, 2026", "size": "48 KB"},
        {"type": "CSV", "title": "Word Frequency Export",
         "desc": "Complete word frequency table with sentiment labels and co-occurrence data from all 1,842 reviews.",
         "date": "Jul 23, 2026", "size": "120 KB"},
        {"type": "PDF", "title": "Q2 Executive Summary",
         "desc": "One-page management summary of Q2 2026 review performance, key wins, and priority improvement areas.",
         "date": "Jun 30, 2026", "size": "540 KB"},
    ]
    cards = []
    for rp in reports:
        icon_cls = "icon-pdf" if rp["type"] == "PDF" else "icon-csv"
        cards.append(f"""
<div class="report-card">
  <div class="report-icon {icon_cls}">{rp["type"]}</div>
  <div style="flex:1">
    <div class="report-title">{rp["title"]}</div>
    <div class="report-desc">{rp["desc"]}</div>
    <div class="report-meta">{rp["date"]} · {rp["size"]}</div>
  </div>
  <button class="report-dl">⬇ Download</button>
</div>""")

    return f"""
<div style="padding:0 28px 24px;">
  <p style="font-size:12px;color:#9c8c7a;margin-bottom:14px;">Downloadable analytics exports</p>
  <div class="report-note">Reports are generated automatically from your review data. Downloads are simulated in this demo.</div>
  {"".join(cards)}
</div>"""



CHAT_WIDGET = ""  # injected via gr.Blocks js instead

CHAT_JS = """
() => {
  // inject chat widget into the main page (outside any iframe)
  if (document.getElementById('chat-fab')) return;

  const style = document.createElement('style');
  style.textContent = `
    #chat-fab{position:fixed;bottom:24px;right:24px;z-index:9999;width:50px;height:50px;border-radius:50%;background:#1a1208;border:none;cursor:pointer;box-shadow:0 4px 16px rgba(0,0,0,0.25);display:flex;align-items:center;justify-content:center;font-size:22px;color:#fff;}
    #chat-popup{position:fixed;bottom:86px;right:24px;z-index:9998;width:360px;background:#fff;border-radius:14px;box-shadow:0 8px 40px rgba(0,0,0,0.18);display:none;flex-direction:column;overflow:hidden;font-family:system-ui,-apple-system,sans-serif;}
    #chat-popup.open{display:flex;}
    #chat-head{background:#1a1208;color:#fff;padding:14px 16px;display:flex;justify-content:space-between;align-items:flex-start;}
    #chat-head .t{font-size:15px;font-weight:700;}
    #chat-head .s{font-size:11px;color:#9c8c7a;margin-top:2px;}
    #chat-close{background:none;border:none;color:#9c8c7a;font-size:18px;cursor:pointer;line-height:1;padding:0;}
    #chat-messages{padding:14px;height:280px;overflow-y:auto;display:flex;flex-direction:column;gap:8px;}
    .cm-bot,.cm-user{max-width:88%;padding:10px 13px;border-radius:12px;font-size:13px;line-height:1.45;}
    .cm-bot{background:#f0ede8;color:#1a1208;align-self:flex-start;}
    .cm-user{background:#1a1208;color:#fff;align-self:flex-end;}
    #chat-input-row{display:flex;gap:8px;padding:10px 12px;border-top:1px solid #e8e2da;}
    #chat-text{flex:1;padding:7px 12px;border:1px solid #e8e2da;border-radius:8px;font-size:12px;outline:none;background:#fafaf8;}
    #chat-send{width:34px;height:34px;border-radius:8px;background:#c9a48e;border:none;cursor:pointer;color:#fff;font-size:15px;display:flex;align-items:center;justify-content:center;}
  `;
  document.head.appendChild(style);

  document.body.insertAdjacentHTML('beforeend', `
    <button id="chat-fab" onclick="chatToggle()">💬</button>
    <div id="chat-popup">
      <div id="chat-head">
        <div><div class="t">Review Assistant</div><div class="s">AI · hotel analytics</div></div>
        <button id="chat-close" onclick="chatToggle()">✕</button>
      </div>
      <div id="chat-messages">
        <div class="cm-bot">Hello! I'm your hotel review assistant. Ask me anything about your review data — complaints, sentiment trends, guest feedback patterns.</div>
      </div>
      <div id="chat-input-row">
        <input id="chat-text" type="text" placeholder="Ask about complaints, sentiment…" onkeydown="if(event.key==='Enter')chatSend()"/>
        <button id="chat-send" onclick="chatSend()">➤</button>
      </div>
    </div>
  `);

  window.chatHistory = [];

  window.chatToggle = function() {
    document.getElementById('chat-popup').classList.toggle('open');
  };

  window.chatSend = async function() {
    const inp = document.getElementById('chat-text');
    const msg = inp.value.trim();
    if (!msg) return;
    inp.value = '';
    const box = document.getElementById('chat-messages');
    box.innerHTML += '<div class="cm-user">' + msg.replace(/</g,'&lt;') + '</div>';
    box.scrollTop = box.scrollHeight;
    const thinking = document.createElement('div');
    thinking.className = 'cm-bot'; thinking.textContent = '…';
    box.appendChild(thinking); box.scrollTop = box.scrollHeight;
    try {
      const r = await fetch('/chat', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({message: msg, history: window.chatHistory})
      });
      const d = await r.json();
      thinking.textContent = d.reply;
      window.chatHistory.push([msg, d.reply]);
    } catch(e) {
      thinking.textContent = '(Error reaching assistant)';
    }
    box.scrollTop = box.scrollHeight;
  };
}
"""


def full_page_html(active_page, page_content):
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>{CSS}</style>
</head>
<body style="background:#f5f2ee;margin:0;padding:0;">
  <div id="main-content" style="background:#f5f2ee;min-height:100vh;">
    <div class="page-header">
      <div>
        <h1>{active_page}</h1>
        <div class="sub">Grand Horizon Hotel · July 2026</div>
      </div>
      <span class="live-badge">Live data</span>
    </div>
    {page_content}
  </div>
</body>
</html>"""


# ── AI chat helpers ───────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a hotel review analytics assistant for Grand Horizon Hotel.
You have access to the following dashboard data for July 2026:
- Total reviews: 1,842 (+4.2% MoM)
- Average rating: 3.9/5
- Sentiment: 58% positive, 19% neutral, 23% negative
- Sentiment score: 72/100
- Top complaints: Noise levels (312), Slow check-in (278), Room cleanliness (241), Wi-Fi quality (198)
- Top topics by positivity: Location 90%, Pool area 81%, Room view 78%, Bed comfort 72%
- Worst topics: Parking access 10%, Noise levels 12%, Wi-Fi quality 15%
Answer concisely and helpfully about review trends, complaints, and guest feedback patterns."""


def chat_response(message, history):
    import anthropic
    client = anthropic.Anthropic()
    messages = []
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        if assistant_msg:
            messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": message})

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    return response.content[0].text


# ── Gradio app ────────────────────────────────────────────────────────────────

def render_overview():
    content = build_overview()
    return full_page_html("Overview", content)

def render_reviews(filter_val, search):
    content = build_reviews(filter_val, search)
    return full_page_html("Reviews", content)

def render_topics(category):
    content = build_topics(category)
    return full_page_html("Topics", content)

def render_trends(tab):
    content = build_trends(tab)
    return full_page_html("Trends", content)

def render_reports():
    content = build_reports()
    return full_page_html("Reports", content)


NAV_CSS = """
/* ── Nav column ── */
#nav-col { background: #1a1208 !important; min-height: 100vh; padding: 0 !important; gap: 0 !important; }
#nav-col > .form, #nav-col > div { background: transparent !important; border: none !important; box-shadow: none !important; gap: 0 !important; }
/* Brand header block */
#nav-brand { padding: 24px 20px 20px !important; border-bottom: 1px solid rgba(255,255,255,0.08) !important; }
#nav-brand p, #nav-brand strong { color: #ffffff !important; font-size: 15px !important; font-weight: 700 !important; line-height: 1.3 !important; }
/* Nav buttons */
.nav-btn, .nav-btn-active {
  width: 100% !important; border: none !important; border-radius: 6px !important;
  text-align: left !important; padding: 9px 12px !important; margin: 1px 0 !important;
  font-size: 13px !important; font-weight: 500 !important; cursor: pointer !important;
  box-shadow: none !important; display: flex !important; align-items: center !important; gap: 10px !important;
  transition: background 0.15s !important;
}
.nav-btn         { background: transparent !important; color: #c3b9aa !important; }
.nav-btn:hover   { background: rgba(255,255,255,0.06) !important; color: #f5f2ee !important; }
.nav-btn-active  { background: rgba(255,255,255,0.10) !important; color: #f5f2ee !important; }
/* Nav buttons padding container */
#nav-btns { padding: 12px !important; flex: 1 !important; }
#nav-btns > .form, #nav-btns > div { background: transparent !important; border: none !important; box-shadow: none !important; gap: 2px !important; }
/* Footer */
#nav-footer { padding: 16px 20px !important; border-top: 1px solid rgba(255,255,255,0.08) !important; }
#nav-footer p { color: #6b5c4a !important; font-size: 11px !important; }
/* Content column */
#content-col { padding: 0 !important; background: #f5f2ee !important; }
#content-col > .form, #content-col > div { background: transparent !important; border: none !important; box-shadow: none !important; }
/* Sub-filter rows (Reviews/Topics/Trends) */
#sub-filters { background: #f5f2ee !important; padding: 12px 28px 4px !important; align-items: center !important; flex-wrap: wrap !important; gap: 8px !important; }
#sub-filters > .form, #sub-filters > div { background: transparent !important; border: none !important; box-shadow: none !important; }
/* Radio pills */
#sub-filters .wrap { gap: 6px !important; flex-wrap: wrap !important; }
#sub-filters .wrap label {
  padding: 5px 14px !important; border-radius: 20px !important;
  border: 1px solid #e8e2da !important; background: #fff !important;
  font-size: 12px !important; font-weight: 500 !important; color: #6b5c4a !important;
  cursor: pointer !important; margin: 0 !important;
}
#sub-filters .wrap label:has(input:checked) {
  background: #1a1208 !important; color: #fff !important; border-color: #1a1208 !important;
}
#sub-filters .wrap input[type=radio] { display: none !important; }
/* Search box */
#sub-filters textarea, #sub-filters input[type=text] {
  border: 1px solid #e8e2da !important; border-radius: 8px !important;
  background: #fff !important; font-size: 12px !important; color: #1a1208 !important;
  padding: 6px 12px !important; box-shadow: none !important;
  min-height: unset !important; height: 34px !important; resize: none !important;
}
#sub-filters .block { background: transparent !important; border: none !important; padding: 0 !important; box-shadow: none !important; min-height: unset !important; }
/* Send button */
#sub-filters button.lg {
  background: #c9a48e !important; color: #fff !important; border: none !important;
  border-radius: 8px !important; font-size: 12px !important; padding: 6px 16px !important;
  height: 34px !important; min-width: unset !important; box-shadow: none !important;
}
/* Hide Gradio chrome */
.gradio-container { max-width: 100% !important; padding: 0 !important; }
footer { display: none !important; }
"""

PAGES = ["Overview", "Reviews", "Topics", "Trends", "Reports"]
NAV_ICONS = ["◻", "☰", "◈", "∿", "⬜"]



# ── Gradio app ────────────────────────────────────────────────────────────────


def page_content_html(page, review_filter="All", review_search="",
                       topic_cat="All", trend_tab="Sentiment Score"):
    if page == "Overview":
        return full_page_html("Overview", build_overview())
    elif page == "Reviews":
        return full_page_html("Reviews", build_reviews(review_filter, review_search))
    elif page == "Topics":
        return full_page_html("Topics", build_topics(topic_cat))
    elif page == "Trends":
        return full_page_html("Trends", build_trends(trend_tab))
    elif page == "Reports":
        return full_page_html("Reports", build_reports())
    return ""


def sub_filters_visible(page):
    return (
        gr.update(visible=(page == "Reviews")),   # review_filter
        gr.update(visible=(page == "Reviews")),   # review_search
        gr.update(visible=(page == "Topics")),    # topic_filter
        gr.update(visible=(page == "Trends")),    # trend_tab
    )


with gr.Blocks(css=CSS + NAV_CSS, title="Hotel Review Dashboard", js=CHAT_JS) as demo:

    current_page = gr.State("Overview")

    with gr.Row(elem_id="app-root"):

        # ── Sidebar ────────────────────────────────────────────────
        with gr.Column(scale=0, min_width=160, elem_id="nav-col"):
            gr.Markdown("**Hotel Review Dashboard**", elem_id="nav-brand")
            with gr.Column(elem_id="nav-btns"):
                nav_buttons = []
                for icon, label in zip(NAV_ICONS, PAGES):
                    btn = gr.Button(f"{icon}  {label}", elem_classes=["nav-btn-active" if label == "Overview" else "nav-btn"])
                    nav_buttons.append(btn)
            gr.Markdown("Grand Horizon Hotel\nJul 2026 · 1,842 reviews", elem_id="nav-footer")

        # ── Content ────────────────────────────────────────────────
        with gr.Column(scale=1, elem_id="content-col"):
            with gr.Row(elem_id="sub-filters"):
                review_filter = gr.Radio(["All","Positive","Neutral","Negative"], value="All",
                                         show_label=False, visible=False)
                review_search = gr.Textbox(placeholder="Search…", show_label=False,
                                           visible=False, scale=3)
                topic_filter  = gr.Radio(["All","Facilities","Service","Housekeeping","F&B","Rooms","General"],
                                          value="All", show_label=False, visible=False)
                trend_tab_sel = gr.Radio(["Sentiment Score","Pos / Neg Split","Review Volume"],
                                          value="Sentiment Score", show_label=False, visible=False)

            content_html = gr.HTML(page_content_html("Overview"))

    # ── Nav handlers ───────────────────────────────────────────────
    def make_nav_handler(target_page):
        def handler(rf, rs, tf, tt):
            html = page_content_html(target_page, rf, rs, tf, tt)
            vis  = sub_filters_visible(target_page)
            btn_updates = [
                gr.update(elem_classes=["nav-btn-active" if p == target_page else "nav-btn"])
                for p in PAGES
            ]
            return [target_page, html] + list(vis) + btn_updates
        return handler

    all_inputs  = [review_filter, review_search, topic_filter, trend_tab_sel]
    all_outputs = ([current_page, content_html] +
                   [review_filter, review_search, topic_filter, trend_tab_sel] +
                   nav_buttons)

    for btn, page in zip(nav_buttons, PAGES):
        btn.click(make_nav_handler(page), inputs=all_inputs, outputs=all_outputs)

    # ── Sub-filter handlers ────────────────────────────────────────
    def update_reviews(rf, rs):
        return gr.update(value=page_content_html("Reviews", review_filter=rf, review_search=rs))
    def update_topics(tf):
        return gr.update(value=page_content_html("Topics", topic_cat=tf))
    def update_trends(tt):
        return gr.update(value=page_content_html("Trends", trend_tab=tt))

    review_filter.change(update_reviews, inputs=[review_filter, review_search], outputs=content_html)
    review_search.change(update_reviews, inputs=[review_filter, review_search], outputs=content_html)
    topic_filter.change(update_topics,   inputs=topic_filter,  outputs=content_html)
    trend_tab_sel.change(update_trends,  inputs=trend_tab_sel, outputs=content_html)


# ── /chat API endpoint ─────────────────────────────────────────────────────
from fastapi import Request as FRequest
from fastapi.responses import JSONResponse

@demo.app.post("/chat")
async def chat_endpoint(request: FRequest):
    body = await request.json()
    message = body.get("message", "")
    history = body.get("history", [])
    try:
        reply = chat_response(message, history)
    except Exception as e:
        reply = f"(Error: {e})"
    return JSONResponse({"reply": reply})


if __name__ == "__main__":
    demo.launch()
