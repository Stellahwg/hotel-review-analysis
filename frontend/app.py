import gradio as gr
import json
import os
import re
from collections import Counter

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


# ── Analysis pipeline (mirrors notebooks/project.ipynb) ──────────────────────

_TOPIC_KEYWORDS = {
    "wifi":        ["wifi", "wi-fi", "internet", "connection"],
    "cleanliness": ["clean", "dirty", "filthy", "spotless", "hygiene", "tidy"],
    "breakfast":   ["breakfast", "buffet", "brunch"],
    "checkin":     ["check in", "check-in", "reception", "front desk", "arrival"],
    "noise":       ["noise", "noisy", "loud", "quiet", "sound"],
    "location":    ["location", "area", "walk", "distance", "metro", "station"],
    "room":        ["room", "bed", "bedroom", "shower", "bathroom", "toilet"],
    "staff":       ["staff", "service", "employee", "helpful", "friendly", "rude"],
}
_TOPIC_DISPLAY = {
    "wifi": "Wi-Fi quality", "cleanliness": "Room cleanliness",
    "breakfast": "Breakfast options", "checkin": "Check-in experience",
    "noise": "Noise levels", "location": "Location",
    "room": "Room quality", "staff": "Staff attitude",
}
_TOPIC_CATEGORY = {
    "wifi": "Facilities", "cleanliness": "Housekeeping", "breakfast": "F&B",
    "checkin": "Service", "noise": "Facilities", "location": "General",
    "room": "Rooms", "staff": "Service",
}
_STOP_WORDS = {
    "the", "and", "was", "for", "but", "were", "with", "not", "from", "there",
    "you", "have", "are", "like", "also", "bit", "very", "this", "that", "had",
    "all", "our", "can", "its", "been", "has", "about", "hotel", "one", "just",
    "only", "even", "would", "really", "stay", "they", "their", "what", "when",
    "will", "some", "than", "get", "got", "did", "too", "time", "room", "rooms",
}
_NEG_WORDS = {
    "noisy", "noise", "dirty", "slow", "rude", "broken", "bad", "terrible",
    "awful", "poor", "disappointing", "small", "cold", "hard", "expensive",
    "smelly", "crowded", "nothing", "unfortunately",
}


def compute_data():
    import pandas as pd
    from textblob import TextBlob

    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "reviews_clean.csv")
    df = pd.read_csv(csv_path)
    df["reviewed_at"] = pd.to_datetime(df["reviewed_at"], errors="coerce")
    df["review_text"] = df["review_text"].fillna("").str.strip().str.replace(r"\s+", " ", regex=True)

    # ── Sentiment (§3) ──────────────────────────────────────────────────────
    df["sentiment"] = df["review_text"].apply(lambda t: TextBlob(str(t)).sentiment.polarity)
    # Classify by rating, consistent with notebook §5 bad/good split
    def _label(row):
        if row["rating"] >= 9:   return "Positive"
        elif row["rating"] <= 6: return "Negative"
        else:                    return "Neutral"
    df["sent_label"] = df.apply(_label, axis=1)

    total = len(df)
    pos_pct = round(df["sent_label"].eq("Positive").sum() / total * 100)
    neg_pct = round(df["sent_label"].eq("Negative").sum() / total * 100)
    neu_pct = 100 - pos_pct - neg_pct
    avg_r5  = round(df["rating"].mean() / 2, 1)
    sentiment_score = round(df["rating"].mean() / 10 * 100)

    # ── Topic detection (§4) ────────────────────────────────────────────────
    def _detect(text):
        t = str(text).lower()
        return [k for k, kws in _TOPIC_KEYWORDS.items() if any(kw in t for kw in kws)]
    df["topics"] = df["review_text"].apply(_detect)

    # ── Per-topic aggregation + business impact score (§5) ──────────────────
    overall_avg = df["rating"].mean()
    topic_rows = []
    for key in _TOPIC_KEYWORDS:
        sub = df[df["topics"].apply(lambda x: key in x)]
        if len(sub) == 0:
            continue
        mentions     = len(sub)
        avg_r        = sub["rating"].mean()
        r_impact     = avg_r - overall_avg
        pos_t        = round(sub["sent_label"].eq("Positive").sum() / mentions * 100)
        avg_sent     = sub["sentiment"].mean()
        sent_sev     = avg_sent - df["sentiment"].mean()
        reach_pct    = mentions / total * 100
        freq_score   = min((mentions / 500) * 100, 100)
        sent_score   = abs(sent_sev) * 100
        impact_score = abs(r_impact) * 50
        reach_score  = min((reach_pct / 50) * 100, 100)
        biz_score    = impact_score * 0.40 + freq_score * 0.30 + reach_score * 0.20 + sent_score * 0.10
        topic_rows.append({
            "key": key, "mentions": mentions, "avg_rating": avg_r,
            "rating_impact": r_impact, "positive_pct": pos_t,
            "biz_score": biz_score,
        })

    # ── Word frequency (§4) ─────────────────────────────────────────────────
    all_text = " ".join(df["review_text"].dropna()).lower()
    words = re.findall(r"\b[a-z]{3,}\b", all_text)
    wc = Counter(w for w in words if w not in _STOP_WORDS)
    word_cloud = [
        (word, freq, "negative" if word in _NEG_WORDS else "positive")
        for word, freq in wc.most_common(25)
    ]

    # ── Monthly trends (§5 additional) ──────────────────────────────────────
    df["ym"] = df["reviewed_at"].dt.to_period("M")
    monthly = (
        df.groupby("ym")
        .agg(pos=("sent_label", lambda x: x.eq("Positive").sum()),
             neg=("sent_label", lambda x: x.eq("Negative").sum()),
             count=("sentiment", "count"),
             avg_r=("rating", "mean"))
        .reset_index()
    )
    monthly = monthly[monthly["count"] >= 10].sort_values("ym").reset_index(drop=True)
    # Pick 6 consecutive months with highest total volume
    best_start, best_vol = 0, 0
    for i in range(max(1, len(monthly) - 5)):
        v = monthly.iloc[i:i + 6]["count"].sum()
        if v > best_vol:
            best_vol, best_start = v, i
    best6 = monthly.iloc[best_start: best_start + 6]
    trend_months   = [str(r["ym"])[-5:-3].lstrip("0") + " '" + str(r["ym"])[2:4] for _, r in best6.iterrows()]
    # Use abbreviated month names
    _MN = {"01":"Jan","02":"Feb","03":"Mar","04":"Apr","05":"May","06":"Jun",
           "07":"Jul","08":"Aug","09":"Sep","10":"Oct","11":"Nov","12":"Dec"}
    trend_months   = [_MN[str(r["ym"])[-2:]] for _, r in best6.iterrows()]
    trend_scores   = [round(r["avg_r"] / 10 * 100) for _, r in best6.iterrows()]
    trend_positive = [round(r["pos"] / r["count"] * 100) for _, r in best6.iterrows()]
    trend_negative = [round(r["neg"] / r["count"] * 100) for _, r in best6.iterrows()]
    trend_volume   = [int(r["count"]) for _, r in best6.iterrows()]

    # Best/worst months for trend stat cards
    best_idx  = best6["avg_r"].idxmax()
    worst_idx = best6["avg_r"].idxmin()
    best_month_label  = f"{_MN[str(monthly.loc[best_idx,'ym'])[-2:]]} {str(monthly.loc[best_idx,'ym'])[:4]}"
    worst_month_label = f"{_MN[str(monthly.loc[worst_idx,'ym'])[-2:]]} {str(monthly.loc[worst_idx,'ym'])[:4]}"
    best_score  = round(monthly.loc[best_idx,  "avg_r"] / 10 * 100)
    worst_score = round(monthly.loc[worst_idx, "avg_r"] / 10 * 100)
    vol_first, vol_last = best6.iloc[0]["count"], best6.iloc[-1]["count"]
    vol_growth = f"{round((vol_last - vol_first) / vol_first * 100):+}%"

    # MoM volume change (last two qualifying months in full dataset)
    if len(monthly) >= 2:
        mom = round((monthly.iloc[-1]["count"] - monthly.iloc[-2]["count"]) / monthly.iloc[-2]["count"] * 100, 1)
        mom_change = f"{mom:+.0f}%"
    else:
        mom_change = "N/A"

    # ── Sample reviews (§D) ─────────────────────────────────────────────────
    reviews = []
    for label in ["Negative", "Positive", "Neutral"]:
        sub = df[df["sent_label"] == label].dropna(subset=["review_text", "reviewed_by"])
        sub = sub[
            sub["review_text"].str.len().between(60, 220) &
            ~sub["review_text"].str.contains(r"[<>{}]", regex=True)
        ]
        for _, row in sub.sample(min(4, len(sub)), random_state=42).iterrows():
            name   = str(row["reviewed_by"])
            parts  = name.split()
            inits  = (parts[0][0] + parts[1][0]).upper() if len(parts) >= 2 else name[:2].upper()
            stars  = max(1, min(5, round(row["rating"] / 2)))
            date   = row["reviewed_at"].strftime("%b %d") if pd.notna(row["reviewed_at"]) else "N/A"
            text   = row["review_text"][:180].rstrip()
            if len(row["review_text"]) > 180:
                text += "…"
            reviews.append({"initials": inits, "name": name[:18], "stars": stars,
                            "date": date, "sentiment": label, "text": text})

    # ── Assemble outputs ─────────────────────────────────────────────────────
    complaints = sorted(
        [{"name": _TOPIC_DISPLAY[r["key"]], "mentions": r["mentions"], "rating_impact": r["rating_impact"]}
         for r in topic_rows if r["rating_impact"] < 0],
        key=lambda x: x["mentions"], reverse=True
    )
    topics_list = sorted(topic_rows, key=lambda x: x["mentions"], reverse=True)

    # Radar: one value per axis category, averaged positive_pct within that category
    radar_cats = ["Cleanliness", "Service", "Comfort", "Breakfast", "Wi-Fi", "Location"]
    _radar_map  = {"Cleanliness": ["cleanliness"], "Service": ["staff", "checkin"],
                   "Comfort": ["room"], "Breakfast": ["breakfast"],
                   "Wi-Fi": ["wifi"], "Location": ["location"]}
    radar_vals = []
    topic_pct  = {r["key"]: r["positive_pct"] for r in topic_rows}
    for cat in radar_cats:
        keys = _radar_map[cat]
        vals = [topic_pct[k] / 100 for k in keys if k in topic_pct]
        radar_vals.append(round(sum(vals) / len(vals), 2) if vals else 0.5)

    return {
        "stats": {
            "total_reviews": total,
            "avg_rating": float(avg_r5),
            "negative_rate": neg_pct,
            "neutral_rate": neu_pct,
            "sentiment_score": sentiment_score,
            "positive_pct": pos_pct,
            "neutral_pct": neu_pct,
            "negative_pct": neg_pct,
            "mom_change": mom_change,
        },
        "complaints":    [(c["name"], c["mentions"]) for c in complaints],
        "word_cloud":    word_cloud,
        "reviews":       reviews,
        "topics":        [{"name": _TOPIC_DISPLAY[r["key"]], "category": _TOPIC_CATEGORY[r["key"]],
                           "positive_pct": r["positive_pct"], "mentions": r["mentions"]}
                          for r in topics_list],
        "trend_months":  trend_months,
        "trend_scores":  trend_scores,
        "trend_positive": trend_positive,
        "trend_negative": trend_negative,
        "trend_volume":  trend_volume,
        "radar_categories": radar_cats,
        "radar_values":  radar_vals,
        "best_month_label":  best_month_label,
        "worst_month_label": worst_month_label,
        "best_score":    best_score,
        "worst_score":   worst_score,
        "vol_growth":    vol_growth,
        "topic_rows":    topic_rows,
    }


# ── Compute data at startup ───────────────────────────────────────────────────
_data = compute_data()
STATS            = _data["stats"]
COMPLAINTS       = _data["complaints"]
WORD_CLOUD       = _data["word_cloud"]
REVIEWS          = _data["reviews"]
TOPICS           = _data["topics"]
TREND_MONTHS     = _data["trend_months"]
TREND_SCORES     = _data["trend_scores"]
TREND_POSITIVE   = _data["trend_positive"]
TREND_NEGATIVE   = _data["trend_negative"]
TREND_VOLUME     = _data["trend_volume"]
RADAR_CATEGORIES = _data["radar_categories"]
RADAR_VALUES     = _data["radar_values"]

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
    <span class="gauge-badge">+50% MoM</span>
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
    <div class="trend-stat-value ts-best">{_data['best_month_label']}</div>
    <div class="trend-stat-sub">score: {_data['best_score']}</div>
  </div>
  <div class="trend-stat-card">
    <div class="trend-stat-label">Worst Month</div>
    <div class="trend-stat-value ts-worst">{_data['worst_month_label']}</div>
    <div class="trend-stat-sub">score: {_data['worst_score']}</div>
  </div>
  <div class="trend-stat-card">
    <div class="trend-stat-label">Growth</div>
    <div class="trend-stat-value ts-grow">{_data['vol_growth']}</div>
    <div class="trend-stat-sub">{TREND_MONTHS[0]} → {TREND_MONTHS[-1]} volume</div>
  </div>
</div>"""

    return f"""
<div style="padding:0 28px 24px;">
  <p style="font-size:12px;color:#9c8c7a;margin-bottom:14px;">Sentiment over time · Apr – Sep 2019</p>
  <div class="card">
    <div class="card-title">{title}</div>
    <div class="card-sub">April – September 2019</div>
    {chart}
  </div>
  {stat_cards}
</div>"""


def build_reports():
    reports = [
        {"type": "PDF", "title": "Sentiment Analysis Report",
         "desc": "Full breakdown of 641 Booking.com reviews for Motel One Brussels — sentiment scores, complaint categories, and guest feedback patterns.",
         "date": "Jul 2021", "size": "2.1 MB"},
        {"type": "PDF", "title": "Top Complaints Analysis",
         "desc": "Deep-dive into the 4 topics with negative rating impact: Room quality, Check-in experience, Noise levels, and Wi-Fi quality.",
         "date": "Jul 2021", "size": "980 KB"},
        {"type": "CSV", "title": "Sentiment Trend Data",
         "desc": "Raw monthly sentiment and volume data from Aug 2018 through Feb 2020, suitable for further analysis.",
         "date": "Jul 2021", "size": "38 KB"},
        {"type": "CSV", "title": "Word Frequency Export",
         "desc": "Complete word frequency table with sentiment labels from all 641 reviews.",
         "date": "Jul 2021", "size": "95 KB"},
        {"type": "PDF", "title": "Business Impact Score Summary",
         "desc": "Priority ranking of all 8 topics by business impact score — combining rating impact, frequency, reach, and sentiment severity.",
         "date": "Jul 2021", "size": "410 KB"},
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
    #chat-popup{position:fixed;bottom:86px;right:24px;z-index:9998;width:440px;background:#fff;border-radius:14px;box-shadow:0 8px 40px rgba(0,0,0,0.18);display:none;flex-direction:column;overflow:hidden;font-family:system-ui,-apple-system,sans-serif;}
    #chat-popup.open{display:flex;}
    #chat-head{background:#1a1208;color:#fff;padding:14px 16px;display:flex;justify-content:space-between;align-items:flex-start;}
    #chat-head .t{font-size:15px;font-weight:700;}
    #chat-head .s{font-size:11px;color:#9c8c7a;margin-top:2px;}
    #chat-close{background:none;border:none;color:#9c8c7a;font-size:18px;cursor:pointer;line-height:1;padding:0;}
    #chat-messages{padding:14px;height:420px;overflow-y:auto;display:flex;flex-direction:column;gap:8px;}
    .cm-bot,.cm-user{max-width:88%;padding:10px 13px;border-radius:12px;font-size:13px;line-height:1.6;}
    .cm-bot{background:#f0ede8;color:#1a1208;align-self:flex-start;}
    .cm-bot strong{font-weight:700;}
    .cm-bot ul{margin:4px 0 4px 16px;padding:0;}
    .cm-bot li{margin:2px 0;}
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

  // Minimal markdown → HTML renderer
  window.mdToHtml = function(text) {
    return text
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      // headings → bold paragraph
      .replace(/^#{1,3} (.+)$/gm, '<strong>$1</strong>')
      // bold
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      // bullet lists
      .replace(/^[-*] (.+)$/gm, '<li>$1</li>')
      // wrap consecutive <li> in <ul>
      .replace(/(<li>.*<\/li>\n?)+/gs, m => '<ul>' + m + '</ul>')
      // line breaks
      .replace(/\n{2,}/g, '<br><br>')
      .replace(/\n/g, '<br>');
  };

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
      thinking.innerHTML = window.mdToHtml(d.reply);
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
        <div class="sub">Motel One Brussels · 2018 – 2021</div>
      </div>
      <span class="live-badge">Live data</span>
    </div>
    {page_content}
  </div>
</body>
</html>"""


# ── AI chat helpers ───────────────────────────────────────────────────────────
_problems   = [t for t in _data["topic_rows"] if t["rating_impact"] < 0]
_strengths  = [t for t in _data["topic_rows"] if t["rating_impact"] >= 0]
_prob_str   = ", ".join(f"{_TOPIC_DISPLAY[t['key']]} ({t['mentions']} mentions, {t['rating_impact']:+.2f} stars)"
                        for t in sorted(_problems, key=lambda x: x["mentions"], reverse=True))
_str_str    = ", ".join(f"{_TOPIC_DISPLAY[t['key']]} ({t['positive_pct']}% positive)"
                        for t in sorted(_strengths, key=lambda x: x["positive_pct"], reverse=True))

SYSTEM_PROMPT = f"""You are a hotel review analytics assistant for Motel One Brussels.
You have access to the following dashboard data from {STATS['total_reviews']} Booking.com reviews:
- Average rating: {STATS['avg_rating']}/5 (on a 10-point scale: {round(STATS['avg_rating'] * 2, 1)}/10)
- Sentiment: {STATS['positive_pct']}% positive, {STATS['neutral_pct']}% neutral, {STATS['negative_pct']}% negative
- Sentiment score: {STATS['sentiment_score']}/100
- Topics with negative rating impact (problems to fix): {_prob_str}
- Top strengths: {_str_str}
- Best month: {_data['best_month_label']} (score {_data['best_score']}), Worst month: {_data['worst_month_label']} (score {_data['worst_score']})
Reply in short, clear sentences. Use bullet points sparingly — only when listing 3+ items. Avoid large headers. Keep answers concise and actionable."""


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


with gr.Blocks(css=CSS + NAV_CSS, title="Hotel Review Dashboard") as demo:

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
            gr.Markdown("Motel One Brussels\n2018–2021 · 641 reviews", elem_id="nav-footer")

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


# ── /chat API endpoint + HTML injection middleware ─────────────────────────
from fastapi import FastAPI, Request as FRequest
from fastapi.responses import JSONResponse, HTMLResponse
from starlette.middleware.base import BaseHTTPMiddleware
import uvicorn

CHAT_SNIPPET = """
<script>
(function() {
  var STYLES = `
    #chat-fab{position:fixed;bottom:24px;right:24px;z-index:99999;width:50px;height:50px;border-radius:50%;background:#1a1208;border:none;cursor:pointer;box-shadow:0 4px 16px rgba(0,0,0,0.25);display:flex;align-items:center;justify-content:center;font-size:22px;color:#fff;}
    #chat-popup{position:fixed;bottom:86px;right:24px;z-index:99998;width:440px;background:#fff;border-radius:14px;box-shadow:0 8px 40px rgba(0,0,0,0.18);display:none;flex-direction:column;overflow:hidden;font-family:system-ui,-apple-system,sans-serif;}
    #chat-popup.open{display:flex;}
    #chat-head{background:#1a1208;color:#fff;padding:14px 16px;display:flex;justify-content:space-between;align-items:flex-start;}
    #chat-head .t{font-size:15px;font-weight:700;}
    #chat-head .s{font-size:11px;color:#9c8c7a;margin-top:2px;}
    #chat-close{background:none;border:none;color:#9c8c7a;font-size:18px;cursor:pointer;line-height:1;padding:0;}
    #chat-messages{padding:14px;height:420px;overflow-y:auto;display:flex;flex-direction:column;gap:8px;}
    .cm-bot,.cm-user{max-width:88%;padding:10px 13px;border-radius:12px;font-size:13px;line-height:1.6;}
    .cm-bot{background:#f0ede8;color:#1a1208;align-self:flex-start;}
    .cm-bot strong{font-weight:700;}
    .cm-bot ul{margin:4px 0 4px 16px;padding:0;}
    .cm-bot li{margin:2px 0;}
    .cm-user{background:#1a1208;color:#fff;align-self:flex-end;}
    #chat-input-row{display:flex;gap:8px;padding:10px 12px;border-top:1px solid #e8e2da;}
    #chat-text{flex:1;padding:7px 12px;border:1px solid #e8e2da;border-radius:8px;font-size:12px;outline:none;background:#fafaf8;}
    #chat-send{width:34px;height:34px;border-radius:8px;background:#c9a48e;border:none;cursor:pointer;color:#fff;font-size:15px;display:flex;align-items:center;justify-content:center;}
  `;

  var chatHistory = [];

  function mdToHtml(t) {
    return t.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
      .replace(/^#{1,3} (.+)$/gm,'<strong>$1</strong>')
      .replace(/\\*\\*(.+?)\\*\\*/g,'<strong>$1</strong>')
      .replace(/^[-*] (.+)$/gm,'<li>$1</li>')
      .replace(/(<li>[\\s\\S]*?<\\/li>\\n?)+/g, function(m){ return '<ul>'+m+'</ul>'; })
      .replace(/\\n{2,}/g,'<br><br>').replace(/\\n/g,'<br>');
  }

  function chatToggle() {
    document.getElementById('chat-popup').classList.toggle('open');
  }

  async function chatSend() {
    var inp=document.getElementById('chat-text'), msg=inp.value.trim();
    if(!msg) return; inp.value='';
    var box=document.getElementById('chat-messages');
    box.innerHTML+='<div class="cm-user">'+msg.replace(/</g,'&lt;')+'</div>';
    box.scrollTop=box.scrollHeight;
    var el=document.createElement('div'); el.className='cm-bot'; el.textContent='…';
    box.appendChild(el); box.scrollTop=box.scrollHeight;
    try {
      var r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({message:msg,history:chatHistory})});
      var d=await r.json();
      el.innerHTML=mdToHtml(d.reply);
      chatHistory.push([msg,d.reply]);
    } catch(e){ el.textContent='(Error reaching assistant)'; }
    box.scrollTop=box.scrollHeight;
  }

  function injectChat() {
    if (document.getElementById('chat-fab')) return;

    var style = document.createElement('style');
    style.textContent = STYLES;
    document.head.appendChild(style);

    var fab = document.createElement('button');
    fab.id = 'chat-fab';
    fab.innerHTML = '💬';
    fab.onclick = chatToggle;
    document.body.appendChild(fab);

    var popup = document.createElement('div');
    popup.id = 'chat-popup';
    popup.innerHTML = `
      <div id="chat-head">
        <div><div class="t">Review Assistant</div><div class="s">AI · hotel analytics</div></div>
        <button id="chat-close" onclick="chatToggle()">✕</button>
      </div>
      <div id="chat-messages">
        <div class="cm-bot">Hello! I'm your hotel review assistant. Ask me anything about the data — complaints, sentiment trends, guest feedback patterns.</div>
      </div>
      <div id="chat-input-row">
        <input id="chat-text" type="text" placeholder="Ask about complaints, sentiment…"/>
        <button id="chat-send">&#10148;</button>
      </div>`;
    document.body.appendChild(popup);

    document.getElementById('chat-text').addEventListener('keydown', function(e){ if(e.key==='Enter') chatSend(); });
    document.getElementById('chat-send').addEventListener('click', chatSend);

    window.chatToggle = chatToggle;
  }

  // Inject immediately and again after Gradio finishes rendering
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function(){ injectChat(); setTimeout(injectChat, 1500); });
  } else {
    injectChat();
    setTimeout(injectChat, 1500);
  }
})();
</script>
"""

class ChatInjectionMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        ct = response.headers.get("content-type", "")
        if "text/html" not in ct:
            return response
        body = b""
        async for chunk in response.body_iterator:
            body += chunk
        html = body.decode("utf-8")
        if "</body>" in html and "injectChat" not in html:
            html = html.replace("</body>", CHAT_SNIPPET + "</body>", 1)
        headers = dict(response.headers)
        headers.pop("content-length", None)
        headers.pop("Content-Length", None)
        return HTMLResponse(content=html, status_code=response.status_code, headers=headers)


app = FastAPI()

@app.post("/chat")
async def chat_endpoint(request: FRequest):
    body = await request.json()
    message = body.get("message", "")
    history = body.get("history", [])
    try:
        reply = chat_response(message, history)
    except Exception as e:
        reply = f"(Error: {e})"
    return JSONResponse({"reply": reply})

app = gr.mount_gradio_app(app, demo, path="/")
app.add_middleware(ChatInjectionMiddleware)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
