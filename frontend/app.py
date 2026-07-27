import gradio as gr
import json
import os
import re
from collections import Counter


# ── Analysis pipeline (mirrors notebooks/project_new2.ipynb) ─────────────────

# Display name + category lookup — translation table only.
# The actual topic list is read from analysis_results.json at startup.
# Unknown keys fall back to a title-cased version of the key.
_LDA_TOPIC_DISPLAY = {
    "room_bathroom":       "Room & Bathroom",
    "facilities_space":    "Facilities & Space",
    "location_transport":  "Location & Transport",
    "service_hospitality": "Service & Hospitality",
    "accommodation_type":  "Accommodation Type",
}
_LDA_TOPIC_CATEGORY = {
    "room_bathroom":       "Rooms",
    "facilities_space":    "Facilities",
    "location_transport":  "General",
    "service_hospitality": "Service",
    "accommodation_type":  "F&B",
}

def _topic_display(key):
    return _LDA_TOPIC_DISPLAY.get(key, key.replace("_", " ").title())

def _topic_category(key):
    return _LDA_TOPIC_CATEGORY.get(key, "General")

# Seed keywords for auto-labelling (exact match, mirrors notebook Cell 23)
_LABEL_SEEDS = {
    "location_transport":   {"walk","station","central","near","close","city","centre","metro","transport","located"},
    "facilities_space":     {"parking","facilities","coffee","bar","space","noisy","tea","spacious","outside","free"},
    "service_hospitality":  {"friendly","helpful","host","service","recommend","kind","owner","hosts","reception","super"},
    "accommodation_type":   {"restaurant","food","value","money","expensive","quality","price","dining","eat","menu"},
    "room_bathroom":        {"bed","bathroom","shower","small","comfortable","room","beds","bath","door","floor","hot","window"},
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
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    import joblib

    base = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

    # ── Step 1: Load Motel One Brussels reviews ──────────────────────────────
    csv_path = os.path.join(base, "reviews_clean.csv")
    df = pd.read_csv(csv_path)
    df["reviewed_at"] = pd.to_datetime(df["reviewed_at"], errors="coerce")
    df["review_text"] = df["review_text"].fillna("").str.strip().str.replace(r"\s+", " ", regex=True)

    # ── Step 2: VADER sentiment + rating-based label (notebook Cell 13) ─────
    analyzer = SentimentIntensityAnalyzer()
    df["overall_sentiment"] = df["review_text"].apply(
        lambda t: analyzer.polarity_scores(str(t))["compound"]
    )

    def _label(row):
        if row["rating"] >= 9:  return "Positive"
        elif row["rating"] < 6: return "Negative"
        else:                   return "Neutral"
    df["sent_label"] = df.apply(_label, axis=1)

    total = len(df)
    pos_pct = round(df["sent_label"].eq("Positive").sum() / total * 100)
    neg_pct = round(df["sent_label"].eq("Negative").sum() / total * 100)
    neu_pct = 100 - pos_pct - neg_pct
    avg_r5  = round(df["rating"].mean() / 2, 1)
    # VADER compound mean ranges -1 to +1; rescale to 0-100
    sentiment_score = round((df["overall_sentiment"].mean() + 1) / 2 * 100)

    # ── Step 3: Load saved LDA model + vectorizer (notebook Cell 23) ────────────
    results_dir = os.path.join(os.path.dirname(__file__), "..", "backend", "results")
    vectorizer = joblib.load(os.path.join(results_dir, "vectorizer.pkl"))
    lda_model  = joblib.load(os.path.join(results_dir, "lda_model.pkl"))

    # Auto-label topics using LABEL_SEEDS + greedy best-first (notebook Cell 24)
    feature_names = vectorizer.get_feature_names_out()
    topic_top_words = {}
    for i, comp in enumerate(lda_model.components_):
        top_idx = comp.argsort()[:-13:-1]  # top 12
        topic_top_words[i] = [feature_names[j] for j in top_idx]

    scores = {}
    for idx, words in topic_top_words.items():
        word_set = set(words)
        for name, seeds in _LABEL_SEEDS.items():
            scores[(idx, name)] = len(word_set & seeds)

    assigned_names = {}
    used_labels, used_indices = set(), set()
    for (idx, name), score in sorted(scores.items(), key=lambda x: -x[1]):
        if idx not in used_indices and name not in used_labels:
            assigned_names[idx] = name
            used_labels.add(name)
            used_indices.add(idx)
    for idx in range(5):
        if idx not in assigned_names:
            assigned_names[idx] = f"topic_{idx}"

    topic_keywords = {assigned_names[i]: topic_top_words[i][:15] for i in range(5)}

    # Transform reviews (inference only — no retraining)
    X_hotel = vectorizer.transform(df["review_text"].astype(str).tolist())
    doc_topic_dist = lda_model.transform(X_hotel)
    def _get_topics(probs):
        topics = [assigned_names[j] for j in range(5) if probs[j] > 0.15]
        return topics if topics else [assigned_names[probs.argmax()]]
    df["lda_topics"] = [_get_topics(doc_topic_dist[i]) for i in range(len(df))]

    # ── Steps 4–5: Load topic analysis from notebook output ──────────────────
    analysis_path = os.path.join(results_dir, "analysis_results.json")
    with open(analysis_path) as _f2:
        _analysis = json.load(_f2)

    # topic_keys comes from the notebook JSON — no hardcoding needed
    topic_keys = [t["topic"] for t in _analysis["topic_analysis"]]

    topic_rows = []
    for t in _analysis["topic_analysis"]:
        name = t["topic"]
        sub  = df[df["lda_topics"].apply(lambda x: name in x)]
        pos_t = round(sub["sent_label"].eq("Positive").sum() / len(sub) * 100) if len(sub) > 0 else 0
        biz = (abs(t["rating_impact"]) * 50 * 0.50
               + min(t["reach_pct"] / 50 * 100, 100) * 0.30
               + abs(t.get("sentiment_severity", 0)) * 100 * 0.20)
        topic_rows.append({
            "key":                  name,
            "mentions":             t["mentions"],
            "avg_rating":           t["avg_rating"],
            "rating_impact":        t["rating_impact"],
            "positive_pct":         pos_t,
            "avg_aspect_sentiment": t.get("avg_aspect_sentiment", 0.0),
            "biz_score":            biz,
        })

    # ── Step 6: Word frequency ───────────────────────────────────────────────
    all_text = " ".join(df["review_text"].dropna()).lower()
    words = re.findall(r"\b[a-z]{3,}\b", all_text)
    wc = Counter(w for w in words if w not in _STOP_WORDS)
    word_cloud = [
        (word, freq, "negative" if word in _NEG_WORDS else "positive")
        for word, freq in wc.most_common(25)
    ]

    # ── Step 7: Monthly trends ───────────────────────────────────────────────
    _MN = {"01":"Jan","02":"Feb","03":"Mar","04":"Apr","05":"May","06":"Jun",
           "07":"Jul","08":"Aug","09":"Sep","10":"Oct","11":"Nov","12":"Dec"}
    df["ym"] = df["reviewed_at"].dt.to_period("M")
    monthly = (
        df.groupby("ym")
        .agg(pos=("sent_label",       lambda x: x.eq("Positive").sum()),
             neg=("sent_label",       lambda x: x.eq("Negative").sum()),
             count=("overall_sentiment", "count"),
             avg_r=("rating",         "mean"))
        .reset_index()
    )
    monthly = monthly[monthly["count"] >= 10].sort_values("ym").reset_index(drop=True)
    best_start, best_vol = 0, 0
    for i in range(max(1, len(monthly) - 5)):
        v = monthly.iloc[i:i + 6]["count"].sum()
        if v > best_vol:
            best_vol, best_start = v, i
    best6 = monthly.iloc[best_start: best_start + 6]
    trend_months   = [_MN[str(r["ym"])[-2:]] for _, r in best6.iterrows()]
    trend_scores   = [round(r["avg_r"] / 10 * 100)        for _, r in best6.iterrows()]
    trend_positive = [round(r["pos"] / r["count"] * 100)  for _, r in best6.iterrows()]
    trend_negative = [round(r["neg"] / r["count"] * 100)  for _, r in best6.iterrows()]
    trend_volume   = [int(r["count"])                      for _, r in best6.iterrows()]
    years = sorted({str(r["ym"])[:4] for _, r in best6.iterrows()})
    trend_year_range = years[0] if len(years) == 1 else f"{years[0]}–{years[-1]}"

    best_idx  = best6["avg_r"].idxmax()
    worst_idx = best6["avg_r"].idxmin()
    best_month_label  = f"{_MN[str(monthly.loc[best_idx, 'ym'])[-2:]]} {str(monthly.loc[best_idx, 'ym'])[:4]}"
    worst_month_label = f"{_MN[str(monthly.loc[worst_idx,'ym'])[-2:]]} {str(monthly.loc[worst_idx,'ym'])[:4]}"
    best_score  = round(monthly.loc[best_idx,  "avg_r"] / 10 * 100)
    worst_score = round(monthly.loc[worst_idx, "avg_r"] / 10 * 100)
    vol_growth = f"{round((best6.iloc[-1]['count'] - best6.iloc[0]['count']) / best6.iloc[0]['count'] * 100):+}%"
    if len(monthly) >= 2:
        mom_val = round((monthly.iloc[-1]["count"] - monthly.iloc[-2]["count"]) / monthly.iloc[-2]["count"] * 100, 1)
        mom_change = f"{mom_val:+.0f}%"
    else:
        mom_change = "N/A"

    # Aspect sentiment per topic per month (for Trends → Aspect Sentiment tab)
    trend_aspect = {}
    best6_periods = [r["ym"] for _, r in best6.iterrows()]
    for name in topic_keys:
        monthly_aspect = []
        for period in best6_periods:
            sub = df[(df["ym"] == period) & df["lda_topics"].apply(lambda x: name in x)]
            if len(sub) > 0:
                sc = sub["overall_sentiment"].tolist()
                monthly_aspect.append(round(sum(sc) / len(sc), 3))
            else:
                monthly_aspect.append(None)
        trend_aspect[name] = monthly_aspect

    # ── Step 8: Pre-COVID risk analysis — load from notebook output ─────────
    risk_json_path = os.path.join(os.path.dirname(__file__), "..", "backend", "results", "lda_risk_analysis.json")
    risk_rows = []
    if os.path.exists(risk_json_path):
        with open(risk_json_path) as _f:
            _risk_data = json.load(_f)
        for r in _risk_data.get("risks", []):
            risk_rows.append({
                "key":              r["topic"],
                "display":          _topic_display(r["topic"]),
                "pct_point_change": round(r["pct_point_change"], 1),
                "rating_change":    round(r["rating_change"], 2),
                "risk_level":       r["risk_level"],
            })

    # ── Step 9: Sample reviews ───────────────────────────────────────────────
    # Sample reviews proportional to actual distribution, targeting ~30 total
    # 65% Positive → 20, 33% Neutral → 9, 3% Negative → 2 (floored to keep negatives visible)
    _sample_counts = {"Positive": 20, "Neutral": 9, "Negative": 2}
    reviews = []
    for label in ["Negative", "Positive", "Neutral"]:
        sub = df[df["sent_label"] == label].dropna(subset=["review_text", "reviewed_by"])
        sub = sub[
            sub["review_text"].str.len().between(60, 220) &
            ~sub["review_text"].str.contains(r"[<>{}]", regex=True)
        ]
        n = _sample_counts[label]
        for _, row in sub.sample(min(n, len(sub)), random_state=42).iterrows():
            name_   = str(row["reviewed_by"])
            parts   = name_.split()
            inits   = (parts[0][0] + parts[1][0]).upper() if len(parts) >= 2 else name_[:2].upper()
            stars   = max(1, min(5, round(row["rating"] / 2)))
            date    = row["reviewed_at"].strftime("%b %d") if pd.notna(row["reviewed_at"]) else "N/A"
            text    = row["review_text"][:180].rstrip()
            if len(row["review_text"]) > 180:
                text += "…"
            reviews.append({"initials": inits, "name": name_[:18], "stars": stars,
                            "date": date, "sentiment": label, "text": text})

    # ── Assemble outputs ─────────────────────────────────────────────────────
    # Actual data date range
    dates = df["reviewed_at"].dropna().sort_values()
    _MN2 = _MN  # reuse the month abbreviation table already defined above
    def _fmt_date(ts):
        return f"{_MN[ts.strftime('%m')]} {ts.strftime('%Y')}"
    data_date_range = f"{_fmt_date(dates.iloc[0])} – {_fmt_date(dates.iloc[-1])}"

    complaints = sorted(
        [{"name": _topic_display(r["key"]), "mentions": r["mentions"],
          "rating_impact": r["rating_impact"]}
         for r in topic_rows if r["rating_impact"] < 0],
        key=lambda x: x["mentions"], reverse=True
    )
    topics_list = sorted(topic_rows, key=lambda x: x["mentions"], reverse=True)

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
        "complaints":      [(c["name"], c["mentions"]) for c in complaints],
        "word_cloud":      word_cloud,
        "reviews":         reviews,
        "topics":          [{"key": r["key"],
                             "name": _topic_display(r["key"]),
                             "category": _topic_category(r["key"]),
                             "positive_pct": r["positive_pct"],
                             "mentions": r["mentions"],
                             "keywords": topic_keywords.get(r["key"], [])[:5]}
                            for r in topics_list],
        "data_date_range": data_date_range,
        "trend_year_range": trend_year_range,
        "trend_months":    trend_months,
        "trend_scores":    trend_scores,
        "trend_positive":  trend_positive,
        "trend_negative":  trend_negative,
        "trend_volume":    trend_volume,
        "trend_aspect":    trend_aspect,
        "best_month_label":  best_month_label,
        "worst_month_label": worst_month_label,
        "best_score":      best_score,
        "worst_score":     worst_score,
        "vol_growth":      vol_growth,
        "topic_rows":      topic_rows,
        "risk_rows":       risk_rows,
    }



# ── Compute data at startup ───────────────────────────────────────────────────
_data = compute_data()
STATS            = _data["stats"]
COMPLAINTS       = _data["complaints"]
WORD_CLOUD       = _data["word_cloud"]
REVIEWS          = _data["reviews"]
TOPICS           = _data["topics"]
DATA_DATE_RANGE  = _data["data_date_range"]
TREND_YEAR_RANGE = _data["trend_year_range"]
TREND_MONTHS     = _data["trend_months"]
TREND_SCORES     = _data["trend_scores"]
TREND_POSITIVE   = _data["trend_positive"]
TREND_NEGATIVE   = _data["trend_negative"]
TREND_VOLUME     = _data["trend_volume"]
TREND_ASPECT     = _data["trend_aspect"]
RISK_ROWS        = _data["risk_rows"]

# ── Warning alerts (derived from risk analysis) ───────────────────────────────
WARNINGS = [
    {"display": r["display"], "risk": r["risk_level"],
     "pp": r["pct_point_change"], "rating": r["rating_change"]}
    for r in RISK_ROWS if r["risk_level"] in ("HIGH", "MEDIUM")
]

_WARNINGS_JSON = json.dumps(WARNINGS)
_WARN_COUNT    = len(WARNINGS)

ALERTS_SNIPPET = f"""
<script>
(function() {{
  if (document.getElementById('alert-bell')) return;

  var WARNINGS = {_WARNINGS_JSON};
  var COUNT = {_WARN_COUNT};

  var style = document.createElement('style');
  style.textContent = `
    #alert-bell {{
      position:fixed; top:16px; right:76px; z-index:99999;
      width:38px; height:38px; border-radius:50%;
      background:#1a1208; border:none; cursor:pointer;
      box-shadow:0 2px 10px rgba(0,0,0,0.22);
      display:flex; align-items:center; justify-content:center;
      font-size:18px; color:#fff;
    }}
    #alert-badge {{
      position:absolute; top:-4px; right:-4px;
      background:#eb6834; color:#fff;
      border-radius:50%; width:18px; height:18px;
      font-size:10px; font-weight:700;
      display:flex; align-items:center; justify-content:center;
      border:2px solid #fff; pointer-events:none;
    }}
    #alert-dropdown {{
      position:fixed; top:62px; right:16px; z-index:99998;
      width:320px; background:#fff; border-radius:12px;
      box-shadow:0 8px 32px rgba(0,0,0,0.16);
      display:none; flex-direction:column; overflow:hidden;
      font-family:system-ui,-apple-system,sans-serif;
    }}
    #alert-dropdown.open {{ display:flex; }}
    #alert-header {{
      background:#1a1208; color:#fff;
      padding:12px 16px; font-size:14px; font-weight:700;
    }}
    #alert-header span {{ font-size:11px; color:#9c8c7a; font-weight:400; margin-left:6px; }}
    .alert-item {{
      padding:12px 16px; border-bottom:1px solid #f0ede8;
      display:flex; flex-direction:column; gap:4px;
    }}
    .alert-item:last-child {{ border-bottom:none; }}
    .alert-name {{ font-size:13px; font-weight:600; color:#1a1208; display:flex; align-items:center; gap:8px; }}
    .alert-badge-high {{ background:#fdeee8; color:#eb6834; border-radius:10px; padding:2px 8px; font-size:10px; font-weight:700; }}
    .alert-badge-med  {{ background:#fef8e6; color:#eda100; border-radius:10px; padding:2px 8px; font-size:10px; font-weight:700; }}
    .alert-detail {{ font-size:11px; color:#6b5c4a; display:flex; gap:14px; }}
    .alert-empty {{ padding:16px; font-size:12px; color:#9c8c7a; text-align:center; }}
  `;
  document.head.appendChild(style);

  var bell = document.createElement('button');
  bell.id = 'alert-bell';
  bell.innerHTML = '&#128276;' + (COUNT > 0 ? '<span id="alert-badge">' + COUNT + '</span>' : '');
  bell.title = COUNT + ' active warning' + (COUNT !== 1 ? 's' : '');
  document.body.appendChild(bell);

  var drop = document.createElement('div');
  drop.id = 'alert-dropdown';

  var header = '<div id="alert-header">&#9888; Warnings<span>Pre-COVID risk analysis</span></div>';
  var rows = '';
  if (COUNT === 0) {{
    rows = '<div class="alert-empty">No active warnings</div>';
  }} else {{
    WARNINGS.forEach(function(w) {{
      var badgeCls = w.risk === 'HIGH' ? 'alert-badge-high' : 'alert-badge-med';
      var ppArrow  = w.pp > 0 ? '&#8593;' : '&#8595;';
      var rArrow   = w.rating < 0 ? '&#8595;' : '&#8593;';
      var ppColor  = w.pp > 0 ? '#eb6834' : '#1baf7a';
      var rColor   = w.rating < 0 ? '#eb6834' : '#1baf7a';
      rows += '<div class="alert-item">' +
        '<div class="alert-name">' + w.display +
          '<span class="' + badgeCls + '">' + w.risk + '</span></div>' +
        '<div class="alert-detail">' +
          '<span style="color:' + ppColor + '">' + ppArrow + ' Mentions ' +
            (w.pp > 0 ? '+' : '') + w.pp.toFixed(1) + 'pp</span>' +
          '<span style="color:' + rColor + '">' + rArrow + ' Rating ' +
            (w.rating > 0 ? '+' : '') + w.rating.toFixed(2) + '</span>' +
        '</div>' +
      '</div>';
    }});
  }}
  drop.innerHTML = header + rows;
  document.body.appendChild(drop);

  bell.addEventListener('click', function(e) {{
    e.stopPropagation();
    drop.classList.toggle('open');
  }});
  document.addEventListener('click', function() {{
    drop.classList.remove('open');
  }});
  drop.addEventListener('click', function(e) {{ e.stopPropagation(); }});

  function injectAlerts() {{
    if (!document.getElementById('alert-bell')) {{
      document.body.appendChild(bell);
      document.body.appendChild(drop);
    }}
  }}
  setTimeout(injectAlerts, 1500);
}})();
</script>
"""


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


def review_card_html(r):
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
    keywords = t.get("keywords", [])
    pills = "".join(
        f'<span style="display:inline-block;background:#f0ede8;border-radius:10px;padding:2px 8px;font-size:10px;color:#6b5c4a;margin:2px 2px 0 0;">{k}</span>'
        for k in keywords
    )
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
  {f'<div style="margin-top:6px;">{pills}</div>' if pills else ''}
</div>"""


def line_chart_svg(months, values, color="#eda100", label="Score", w=540, h=220):
    pad_l, pad_r, pad_t, pad_b = 40, 20, 20, 30
    plot_w = w - pad_l - pad_r
    plot_h = h - pad_t - pad_b
    mn, mx = min(values) - 5, max(values) + 5

    def px(i):
        return pad_l + i / max(len(months) - 1, 1) * plot_w

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


def multi_line_chart_svg(months, series, w=540, h=220):
    """series: list of (label, color, [values_or_None])"""
    pad_l, pad_r, pad_t, pad_b = 44, 20, 20, 30
    plot_w = w - pad_l - pad_r
    plot_h = h - pad_t - pad_b
    all_vals = [v for _, _, vals in series for v in vals if v is not None]
    if not all_vals:
        return f'<svg width="100%" viewBox="0 0 {w} {h}"><text x="{w//2}" y="{h//2}" text-anchor="middle" font-size="11" fill="#9c8c7a">No data</text></svg>'
    mn, mx = min(all_vals) - 0.05, max(all_vals) + 0.05

    def px(i): return pad_l + i / max(len(months) - 1, 1) * plot_w
    def py(v): return pad_t + (1 - (v - mn) / (mx - mn)) * plot_h

    grids = ""
    for tick_n in range(5):
        tick_v = mn + (mx - mn) * tick_n / 4
        y = py(tick_v)
        grids += f'<line x1="{pad_l}" y1="{y:.1f}" x2="{w-pad_r}" y2="{y:.1f}" stroke="#e8e2da" stroke-width="1"/>'
        grids += f'<text x="{pad_l-4}" y="{y+3:.1f}" font-size="8" fill="#9c8c7a" text-anchor="end">{tick_v:.2f}</text>'

    xlabels = "".join(
        f'<text x="{px(i):.1f}" y="{h-4}" font-size="9" fill="#9c8c7a" text-anchor="middle">{m}</text>'
        for i, m in enumerate(months)
    )

    lines = ""
    legend = ""
    for si, (label, color, vals) in enumerate(series):
        # Connect only non-None points
        segments, seg = [], []
        for i, v in enumerate(vals):
            if v is not None:
                seg.append((i, v))
            else:
                if len(seg) > 1:
                    segments.append(seg)
                seg = []
        if len(seg) > 1:
            segments.append(seg)
        for seg in segments:
            pts = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in seg)
            lines += f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round" opacity="0.85"/>'
        for i, v in enumerate(vals):
            if v is not None:
                lines += f'<circle cx="{px(i):.1f}" cy="{py(v):.1f}" r="3" fill="{color}" stroke="#fff" stroke-width="1.5"/>'
        lx = pad_l + si * 110
        legend += f'<circle cx="{lx+5}" cy="{h-2}" r="4" fill="{color}"/>'
        legend += f'<text x="{lx+13}" y="{h+1}" font-size="9" fill="#6b5c4a">{label}</text>'

    return f'''<svg width="100%" viewBox="0 0 {w} {h+12}" style="overflow:visible">
  {grids}{xlabels}{lines}{legend}
</svg>'''



# ── Page builders ─────────────────────────────────────────────────────────────

def build_overview():
    s = STATS
    stat_row = f"""
<div class="stat-row">
  <div class="stat-card"><div class="stat-label">Total Reviews</div><div class="stat-value">{s['total_reviews']:,}</div><div class="stat-sub">{s['mom_change']} vs last month</div><div class="stat-icon">↗</div></div>
  <div class="stat-card"><div class="stat-label">Avg Rating</div><div class="stat-value">{s['avg_rating']}/5</div><div class="stat-sub">{DATA_DATE_RANGE}</div><div class="stat-icon">★</div></div>
  <div class="stat-card"><div class="stat-label">Negative Rate</div><div class="stat-value">{s['negative_rate']}%</div><div class="stat-sub">of all reviews</div><div class="stat-icon" style="color:#eb6834">↘</div></div>
  <div class="stat-card"><div class="stat-label">Neutral Rate</div><div class="stat-value">{s['neutral_rate']}%</div><div class="stat-sub">of all reviews</div><div class="stat-icon">—</div></div>
</div>"""

    gauge = gauge_svg(s["sentiment_score"])
    sentiment_panel = f"""
<div class="card">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
    <div class="card-title">Overall Sentiment</div>
    <span class="gauge-badge">{s['mom_change']} MoM</span>
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

    # Show ~8 reviews on overview: proportional mix across sentiments
    _overview_counts = {"Negative": 1, "Positive": 5, "Neutral": 2}
    _overview_reviews = []
    for _label, _n in _overview_counts.items():
        _overview_reviews += [r for r in REVIEWS if r["sentiment"] == _label][:_n]
    recent_reviews = "".join(
        f'<div class="review-card" data-sentiment="{r["sentiment"]}">'
        f'<div class="review-header"><div class="avatar">{r["initials"]}</div>'
        f'<div><div class="review-name">{r["name"]}</div>'
        f'<div class="review-stars">{"★"*r["stars"]}{"☆"*(5-r["stars"])}</div></div>'
        f'<div class="review-date">{r["date"]}</div></div>'
        f'<div class="review-text">{r["text"]}</div></div>'
        for r in _overview_reviews
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
    active_filters = []
    if filter_val != "All":
        active_filters.append(f"sentiment: <strong>{filter_val}</strong>")
    if search:
        active_filters.append(f"search: <strong>{search}</strong>")
    filter_note = (
        f'<p style="font-size:11px;color:#9c8c7a;margin-bottom:10px;">'
        f'{len(filtered)} of {len(REVIEWS)} reviews'
        + (f' · filtered by {", ".join(active_filters)}' if active_filters else "")
        + '</p>'
    )

    return f"""
<div style="padding:0 28px 24px;">
  <p style="font-size:12px;color:#9c8c7a;margin-bottom:14px;">All guest feedback · {len(REVIEWS):,} entries · use the filter controls above to search</p>
  {filter_note}
  {cards if cards else '<p style="color:#9c8c7a;font-size:12px;">No reviews match.</p>'}
</div>"""


def build_topics(category="All"):
    items = TOPICS if category == "All" else [t for t in TOPICS if t["category"] == category]
    cards = "".join(topic_card_html(t) for t in items)

    # Business impact score chart — unique info not shown in topic cards
    # Pull scores from topic_rows via _data
    score_map = {r["key"]: r["biz_score"] for r in _data["topic_rows"]}
    max_score = max(score_map.values()) if score_map else 1

    def _biz_chart(topics):
        if not topics:
            return '<p style="color:#9c8c7a;font-size:12px;">No topics in this category.</p>'
        rows = []
        for t in sorted(topics, key=lambda x: score_map.get(x.get("key", x["name"]), 0), reverse=True):
            key   = t.get("key", t["name"])
            score = score_map.get(key, 0)
            pct   = score / max_score * 100
            r_impact = next((r["rating_impact"] for r in _data["topic_rows"] if r["key"] == key), 0)
            color = "#eb6834" if r_impact < 0 else "#1baf7a"
            rows.append(f"""
<div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
  <div style="font-size:11px;color:#6b5c4a;text-align:right;min-width:150px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{t["name"]}</div>
  <div style="flex:1;background:#f0ede8;border-radius:4px;height:14px;">
    <div style="width:{pct:.1f}%;height:100%;background:{color};border-radius:4px;opacity:0.85;"></div>
  </div>
  <div style="font-size:11px;font-weight:700;color:{color};min-width:38px;">{score:.1f}</div>
</div>""")
        return '<div style="padding-top:4px;">' + "".join(rows) + "</div>"

    chart = _biz_chart(items)

    # Risk analysis table
    _risk_color = {"HIGH": "#eb6834", "MEDIUM": "#eda100", "LOW": "#1baf7a"}
    risk_rows_html = ""
    for r in RISK_ROWS:
        color = _risk_color.get(r["risk_level"], "#9c8c7a")
        arrow = "↑" if r["pct_point_change"] > 0 else "↓"
        rating_arrow = "↓" if r["rating_change"] < 0 else "↑"
        risk_rows_html += f"""
<tr>
  <td style="padding:8px 12px;font-size:12px;font-weight:600;color:#1a1208;">{r["display"]}</td>
  <td style="padding:8px 12px;font-size:12px;color:#6b5c4a;text-align:center;">{arrow} {abs(r["pct_point_change"]):.1f}pp</td>
  <td style="padding:8px 12px;font-size:12px;color:#6b5c4a;text-align:center;">{rating_arrow} {abs(r["rating_change"]):.2f}</td>
  <td style="padding:8px 12px;text-align:center;">
    <span style="background:{color}22;color:{color};border-radius:10px;padding:2px 10px;font-size:10px;font-weight:700;">{r["risk_level"]}</span>
  </td>
</tr>"""

    risk_card = f"""
<div class="card" style="margin-top:14px;">
  <div class="card-title">Pre-COVID Risk Analysis</div>
  <div class="card-sub">Topic trends {DATA_DATE_RANGE} · first vs second half (pre-COVID)</div>
  <table style="width:100%;border-collapse:collapse;">
    <thead>
      <tr style="border-bottom:1px solid #e8e2da;">
        <th style="padding:6px 12px;font-size:10px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#9c8c7a;text-align:left;">Topic</th>
        <th style="padding:6px 12px;font-size:10px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#9c8c7a;text-align:center;">Mention trend</th>
        <th style="padding:6px 12px;font-size:10px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#9c8c7a;text-align:center;">Rating Δ</th>
        <th style="padding:6px 12px;font-size:10px;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;color:#9c8c7a;text-align:center;">Risk</th>
      </tr>
    </thead>
    <tbody>{risk_rows_html}</tbody>
  </table>
</div>"""

    return f"""
<div style="padding:0 28px 24px;">
  <p style="font-size:12px;color:#9c8c7a;margin-bottom:14px;">LDA-discovered topics · aspect-based sentiment analysis</p>
  <div class="topics-grid">{cards}</div>
  <div class="card" style="margin-top:14px;">
    <div class="card-title">Business Impact Score</div>
    <div class="card-sub">Red = hurts ratings · Green = strength · Score: 50% rating impact + 30% reach + 20% sentiment</div>
    {chart}
  </div>
  {risk_card}
</div>"""


def build_trends(tab="Sentiment Score"):
    _topic_colors = {
        "room_bathroom":       "#eb6834",
        "facilities_space":    "#eda100",
        "location_transport":  "#1baf7a",
        "service_hospitality": "#2a78d6",
        "accommodation_type":  "#9b59b6",
    }

    if tab == "Sentiment Score":
        chart = line_chart_svg(TREND_MONTHS, TREND_SCORES, color="#eda100", label="Score")
        title = "Sentiment Score Over Time"
    elif tab == "Pos / Neg Split":
        chart = multi_line_chart_svg(TREND_MONTHS, [
            ("Positive %", "#1baf7a", TREND_POSITIVE),
            ("Negative %", "#eb6834", TREND_NEGATIVE),
        ])
        title = "Positive / Negative Split Over Time"
    elif tab == "Aspect Sentiment":
        series = [
            (_topic_display(name), _topic_colors.get(name, "#888"),
             TREND_ASPECT.get(name, [None]*6))
            for name in TREND_ASPECT
        ]
        chart = multi_line_chart_svg(TREND_MONTHS, series)
        title = "Aspect Sentiment by Topic Over Time"
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

    months_label = f"{TREND_MONTHS[0]} – {TREND_MONTHS[-1]} {TREND_YEAR_RANGE}"
    return f"""
<div style="padding:0 28px 24px;">
  <p style="font-size:12px;color:#9c8c7a;margin-bottom:14px;">Sentiment over time · {months_label}</p>
  <div class="card">
    <div class="card-title">{title}</div>
    <div class="card-sub">{months_label}</div>
    {chart}
  </div>
  {stat_cards}
</div>"""


def build_reports():
    reports = [
        {"type": "PDF", "title": "Sentiment Analysis Report",
         "desc": f"Full breakdown of {STATS['total_reviews']} Booking.com reviews for Motel One Brussels — sentiment scores, complaint categories, and guest feedback patterns.",
         "date": "Jul 2021", "size": "2.1 MB"},
        {"type": "PDF", "title": "Top Complaints Analysis",
         "desc": f"Deep-dive into the {len(COMPLAINTS)} topics with negative rating impact: {', '.join(c[0] for c in COMPLAINTS)}.",
         "date": "Jul 2021", "size": "980 KB"},
        {"type": "CSV", "title": "Sentiment Trend Data",
         "desc": f"Raw monthly sentiment and volume data from {DATA_DATE_RANGE}, suitable for further analysis.",
         "date": "Jul 2021", "size": "38 KB"},
        {"type": "CSV", "title": "Word Frequency Export",
         "desc": f"Complete word frequency table with sentiment labels from all {STATS['total_reviews']} reviews.",
         "date": "Jul 2021", "size": "95 KB"},
        {"type": "PDF", "title": "Business Impact Score Summary",
         "desc": f"Priority ranking of all {len(TOPICS)} LDA topics by business impact score — combining rating impact, frequency, reach, and aspect sentiment severity.",
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
        <div class="sub">Motel One Brussels · {DATA_DATE_RANGE}</div>
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
_prob_str   = ", ".join(f"{_topic_display(t['key'])} ({t['mentions']} mentions, {t['rating_impact']:+.2f} stars)"
                        for t in sorted(_problems, key=lambda x: x["mentions"], reverse=True))
_str_str    = ", ".join(f"{_topic_display(t['key'])} ({t['positive_pct']}% positive)"
                        for t in sorted(_strengths, key=lambda x: x["positive_pct"], reverse=True))
_risk_str   = ", ".join(f"{r['display']} ({r['risk_level']}, mention trend {r['pct_point_change']:+.1f}pp, rating {r['rating_change']:+.2f})"
                        for r in sorted(RISK_ROWS, key=lambda x: {"HIGH":0,"MEDIUM":1,"LOW":2}[x["risk_level"]]))

SYSTEM_PROMPT = f"""You are ReviewRadar, a hotel review analytics assistant for Motel One Brussels.
You have access to the following dashboard data from {STATS['total_reviews']} Booking.com reviews ({DATA_DATE_RANGE}).
Topics are discovered by LDA machine learning (5 topics). Sentiment uses Aspect-Based Sentiment Analysis (ABSA).

Key metrics:
- Average rating: {STATS['avg_rating']}/5 ({round(STATS['avg_rating'] * 2, 1)}/10)
- Sentiment: {STATS['positive_pct']}% positive, {STATS['neutral_pct']}% neutral, {STATS['negative_pct']}% negative
- Sentiment score: {STATS['sentiment_score']}/100
- Topics with negative rating impact: {_prob_str}
- Strengths: {_str_str}
- Best month: {_data['best_month_label']} (score {_data['best_score']}), Worst: {_data['worst_month_label']} (score {_data['worst_score']})
- Pre-COVID risk analysis: {_risk_str}

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


with gr.Blocks(css=CSS + NAV_CSS, title="ReviewRadar") as demo:

    current_page = gr.State("Overview")

    with gr.Row(elem_id="app-root"):

        # ── Sidebar ────────────────────────────────────────────────
        with gr.Column(scale=0, min_width=160, elem_id="nav-col"):
            gr.Markdown("**ReviewRadar**", elem_id="nav-brand")
            with gr.Column(elem_id="nav-btns"):
                nav_buttons = []
                for icon, label in zip(NAV_ICONS, PAGES):
                    btn = gr.Button(f"{icon}  {label}", elem_classes=["nav-btn-active" if label == "Overview" else "nav-btn"])
                    nav_buttons.append(btn)
            gr.Markdown(f"Motel One Brussels\n2018–2021 · {STATS['total_reviews']} reviews", elem_id="nav-footer")

        # ── Content ────────────────────────────────────────────────
        with gr.Column(scale=1, elem_id="content-col"):
            with gr.Row(elem_id="sub-filters"):
                review_filter = gr.Radio(["All","Positive","Neutral","Negative"], value="All",
                                         show_label=False, visible=False)
                review_search = gr.Textbox(placeholder="Search…", show_label=False,
                                           visible=False, scale=3)
                _topic_cats = ["All"] + sorted({t["category"] for t in TOPICS})
                topic_filter  = gr.Radio(_topic_cats,
                                          value="All", show_label=False, visible=False)
                trend_tab_sel = gr.Radio(["Sentiment Score","Pos / Neg Split","Review Volume","Aspect Sentiment"],
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
        return page_content_html("Reviews", review_filter=rf, review_search=rs)
    def update_topics(tf):
        return page_content_html("Topics", topic_cat=tf)
    def update_trends(tt):
        return page_content_html("Trends", trend_tab=tt)

    review_filter.change(update_reviews, inputs=[review_filter, review_search], outputs=content_html)
    review_search.change(update_reviews, inputs=[review_filter, review_search], outputs=content_html)
    topic_filter.change(update_topics,   inputs=[topic_filter],  outputs=content_html)
    trend_tab_sel.change(update_trends,  inputs=[trend_tab_sel], outputs=content_html)


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
        <div class="cm-bot">Hello! I'm the ReviewRadar assistant. Ask me anything about the data — complaints, sentiment trends, guest feedback patterns.</div>
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
        if "</body>" in html and "injectAlerts" not in html:
            html = html.replace("</body>", ALERTS_SNIPPET + "</body>", 1)
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
