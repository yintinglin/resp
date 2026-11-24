#!/usr/bin/env python3
"""Fetch Taiwan CDC respiratory surveillance data and write data.json.

Sources (NIDSS, no CORS — intended for a weekly GitHub Actions job, not
browser-side fetch from GitHub Pages):

- LARS influenza A/B/untyped:
  https://nidss.cdc.gov.tw/misc/lars?id=flu
- LARS key respiratory pathogens (influenza, SARS-CoV-2, RSV, Adeno,
  S. pneumoniae, Parainfluenza):
  https://nidss.cdc.gov.tw/misc/lars?id=uri
- NHI visits (門診, 門診+急診); 急診 = 門急診 − 門診:
  https://nidss.cdc.gov.tw/Cdcwnh/Cdcwnh01?disease=4  (類流感)
  https://nidss.cdc.gov.tw/Cdcwnh/Cdcwnh01?disease=11 (新冠)
- COVID 併發重症, by onset week:
  https://nidss.cdc.gov.tw/nndss/disease?id=19SC
- COVID 死亡, by date of death:
  https://nidss.cdc.gov.tw/nndss/deadmap?id=19SC

Never invent counts. If a required series cannot be parsed (for example
because the NIDSS page layout changed), keep the existing data.json and
exit with a non-zero status. A successful run that only changes
meta.retrieved_at also leaves the file untouched, so the weekly job
commits only when the epidemiological content actually changes.
"""

from __future__ import annotations

import json
import math
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import requests

BASE = "https://nidss.cdc.gov.tw"
UA = "Mozilla/5.0 (compatible; resp-weekly/1.0; +https://github.com/yintinglin)"
TZ = ZoneInfo("Asia/Taipei")
OUT = Path(__file__).resolve().parent / "data.json"

# Chart window: W21 of the latest year (or previous year if still before W21)
FORECAST_HORIZON = 4  # W+1 .. W+4; short enough to avoid a long dive-to-zero
MA_WINDOW = 3
HOLD_THRESHOLD = 0.05  # |WoW| < 5% → 持平
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": UA})


# ---------------------------------------------------------------------------
# CDC week (Sun–Sat; week 1 contains January 4)
# ---------------------------------------------------------------------------

def week1_sunday(year: int) -> date:
    jan4 = date(year, 1, 4)
    # Monday=0 ... Sunday=6 → days since Sunday
    return jan4 - timedelta(days=(jan4.weekday() + 1) % 7)


def week_sunday(year: int, week: int) -> date:
    return week1_sunday(year) + timedelta(weeks=week - 1)


def weeks_in_year(year: int) -> int:
    # If 31 Dec is still in this year's week numbering
    n = 52
    end = week_sunday(year, 53) + timedelta(days=6)
    # week 53 exists if its Sunday is still before next year's week 1
    if week_sunday(year, 53) < week1_sunday(year + 1):
        return 53
    return 52


def parse_year_week(token: str) -> tuple[int, int]:
    token = str(token).strip()
    if len(token) == 6 and token.isdigit():
        return int(token[:4]), int(token[4:])
    raise ValueError(f"unrecognised year-week: {token}")


def format_yw(year: int, week: int) -> str:
    return f"{year}{week:02d}"


def label_week(year: int, week: int, latest_year: int) -> str:
    """X-axis label: W21, or 27W02 when the year differs from latest_year."""
    if year == latest_year:
        return f"W{week:02d}"
    return f"{year % 100:02d}W{week:02d}"


def iter_weeks(start: tuple[int, int], end: tuple[int, int]):
    y, w = start
    while (y, w) <= end:
        yield y, w
        w += 1
        if w > weeks_in_year(y):
            y, w = y + 1, 1


def add_weeks(year: int, week: int, n: int) -> tuple[int, int]:
    y, w = year, week
    step = 1 if n >= 0 else -1
    for _ in range(abs(n)):
        w += step
        if w < 1:
            y -= 1
            w = weeks_in_year(y)
        elif w > weeks_in_year(y):
            y, w = y + 1, 1
    return y, w


# ---------------------------------------------------------------------------
# HTML / Highcharts parsing
# ---------------------------------------------------------------------------

def get(url: str, timeout: int = 40) -> str:
    r = SESSION.get(url, timeout=timeout)
    r.raise_for_status()
    return r.text


def parse_highcharts(html: str) -> list[dict]:
    results = []
    needle = "hcJson.push("
    idx = 0
    while True:
        i = html.find(needle, idx)
        if i < 0:
            break
        j = i + len(needle)
        if j >= len(html) or html[j] != "{":
            idx = j
            continue
        depth = 0
        k = j
        in_str = False
        esc = False
        while k < len(html):
            c = html[k]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        k += 1
                        break
            k += 1
        raw = html[j:k]
        idx = k
        raw2 = re.sub(r",\s*}", "}", raw)
        raw2 = re.sub(r",\s*]", "]", raw2).replace("undefined", "null")
        try:
            obj = json.loads(raw2)
        except json.JSONDecodeError:
            continue
        results.append(obj)
    return results


def series_by_name(charts: list[dict], wanted: str) -> tuple[list[str], list]:
    for ch in charts:
        for ser in ch.get("series") or []:
            if ser.get("name") == wanted:
                return list(ch.get("xAxis_categories") or []), list(ser.get("data") or [])
    return [], []


def to_week_map(cats: list[str], data: list) -> dict[tuple[int, int], int]:
    out = {}
    for c, v in zip(cats, data):
        if v is None:
            continue
        try:
            yw = parse_year_week(c)
        except ValueError:
            continue
        out[yw] = int(v)
    return out


def extract_form(html: str, form_id: str = "SearchForm") -> list[tuple[str, str]]:
    m = re.search(rf'<form[^>]*id="{form_id}"[^>]*>(.*?)</form>', html, re.S | re.I)
    if not m:
        raise RuntimeError("SearchForm not found")
    form = m.group(1)
    fields: list[tuple[str, str]] = []
    for tag in re.finditer(r"<input([^>]*)>", form, re.I):
        attrs = tag.group(1)
        name_m = re.search(r'\bname="([^"]+)"', attrs)
        typ_m = re.search(r'\btype="([^"]+)"', attrs)
        typ = (typ_m.group(1) if typ_m else "text").lower()
        val_m = re.search(r'\bvalue="([^"]*)"', attrs)
        checked = "checked" in attrs.lower()
        if not name_m:
            continue
        name = name_m.group(1)
        value = val_m.group(1) if val_m else ""
        if typ in ("checkbox", "radio"):
            if checked:
                fields.append((name, value))
        elif typ in ("submit", "button", "image"):
            continue
        else:
            fields.append((name, value))
    for m2 in re.finditer(r"<select([^>]*)>(.*?)</select>", form, re.S | re.I):
        attrs, body = m2.group(1), m2.group(2)
        name_m = re.search(r'\bname="([^"]+)"', attrs)
        if not name_m:
            continue
        name = name_m.group(1)
        sel = (
            re.search(r'<option[^>]*selected[^>]*value="([^"]*)"', body, re.I)
            or re.search(r'<option[^>]*value="([^"]*)"[^>]*selected', body, re.I)
            or re.search(r'<option[^>]*value="([^"]*)"', body, re.I)
        )
        fields.append((name, sel.group(1) if sel else ""))
    return fields


def national_visit_map(html: str, must_substr: str) -> dict[tuple[int, int], int]:
    """Extract 全國 就診人次 series whose chart title contains must_substr."""
    for ch in parse_highcharts(html):
        title = ch.get("title_text") or ""
        if "就診人次" not in title or must_substr not in title:
            continue
        cats = list(ch.get("xAxis_categories") or [])
        for ser in ch.get("series") or []:
            if ser.get("name") == "全國":
                return to_week_map(cats, ser.get("data") or [])
    raise RuntimeError(f"no national visit series for {must_substr!r}")


def fetch_nhi_visits(disease: int, case_type: str, case_text: str, title_key: str):
    """GET default 門診, or POST to switch 門診+急診, from Cdcwnh.

    NIDSS occasionally returns a chart-less page after a POST; retry a few times
    with a fresh GET (new anti-forgery token) before giving up.
    """
    import time

    url = f"{BASE}/Cdcwnh/Cdcwnh01?disease={disease}"
    last_err = None
    # NIDSS sometimes returns a 200 page whose charts did not switch ICD;
    # retry with a short backoff. A fresh session cookie/token each try.
    for attempt in range(6):
        if attempt:
            SESSION.cookies.clear()
        html = get(url)
        if case_type == "[1]":
            try:
                return national_visit_map(html, title_key)
            except RuntimeError as e:
                last_err = e
                time.sleep(2.0)
                continue
        fields = extract_form(html)
        new = []
        for n, v in fields:
            if n == "pty_case_type_value":
                new.append((n, case_type))
            elif n == "pty_case_type_text":
                new.append((n, case_text))
            else:
                new.append((n, v))
        r = SESSION.post(url, data=new, timeout=60)
        r.raise_for_status()
        try:
            return national_visit_map(r.text, title_key)
        except RuntimeError as e:
            last_err = e
            time.sleep(2.0)
    raise RuntimeError(f"{title_key} {case_text}: {last_err}")


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def pct_vs_prior_two_week_avg(current, prev, prev2) -> float | None:
    if current is None or prev is None or prev2 is None:
        return None
    avg = (prev + prev2) / 2.0
    if avg == 0:
        return None
    return (current - avg) / avg * 100.0


def round1(x: float | None) -> float | None:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return round(x + 0.0, 1)


def ma3(values: list[int | None]) -> list[float | None]:
    out: list[float | None] = []
    for i, v in enumerate(values):
        window = values[max(0, i - MA_WINDOW + 1) : i + 1]
        nums = [x for x in window if x is not None]
        if len(nums) < min(MA_WINDOW, i + 1) or not nums:
            out.append(None)
        else:
            # require 3 points once enough history exists
            if i + 1 >= MA_WINDOW and len(nums) < MA_WINDOW:
                out.append(None)
            else:
                out.append(sum(nums) / len(nums))
    return out


def trend_word(current: int | None, prev: int | None) -> str:
    if current is None or prev is None or prev == 0:
        return "不明"
    chg = (current - prev) / prev
    if abs(chg) < HOLD_THRESHOLD:
        return "持平"
    return "上升" if chg > 0 else "下降"


def _fill_internal_gaps(values: list):
    """Linear-interpolate internal missing weeks. Leading/trailing None stay None.

    Missing weeks are not dropped, so adjacent observed weeks are never glued
    together across a gap.
    """
    out = list(values)
    n = len(out)
    i = 0
    while i < n:
        if out[i] is not None:
            i += 1
            continue
        j = i
        while j < n and out[j] is None:
            j += 1
        left = i - 1
        right = j if j < n else None
        if left >= 0 and right is not None:
            lo = float(out[left])
            hi = float(out[right])
            span = right - left
            for k in range(i, j):
                w = (k - left) / span
                out[k] = lo + w * (hi - lo)
        i = j
    return out


def recent_contiguous_series(week_map: dict, latest: tuple[int, int], last_n: int = 16):
    """Last `last_n` calendar weeks ending at `latest`, gaps interpolated.

    Uses the full history in `week_map`, not just the chart display window.
    Returns a numeric list, or None if fewer than 4 usable points remain.
    """
    weeks = [add_weeks(*latest, i) for i in range(-(last_n - 1), 1)]
    raw = [week_map.get(yw) for yw in weeks]
    if sum(v is not None for v in raw) < 4:
        return None
    filled = _fill_internal_gaps(raw)
    if filled[-1] is None:
        return None
    start = 0
    for i, v in enumerate(filled):
        if v is None:
            start = i + 1
    series = filled[start:]
    if len(series) < 4:
        return None
    return [float(v) for v in series]


def damped_holt_forecast(week_map: dict, latest: tuple[int, int], horizon: int, last_n: int = 16):
    """Damped-trend Holt's method on log1p of the last `last_n` calendar weeks.

    Returns (point forecasts, lower, upper), each of length `horizon`.
    Intervals use one-step residual σ × √h × 1.645 on the log scale
    (approx. 90%), then expm1; they are uncalibrated rough bands.
    """
    series = recent_contiguous_series(week_map, latest, last_n)
    if series is None:
        return [None] * horizon, [None] * horizon, [None] * horizon
    ys = np.log1p(np.array(series, dtype=float))

    best = None
    alphas = np.linspace(0.2, 0.9, 8)
    betas = np.linspace(0.05, 0.5, 6)
    phis = np.linspace(0.7, 0.98, 6)
    for alpha in alphas:
        for beta in betas:
            for phi in phis:
                level = ys[0]
                trend = ys[1] - ys[0]
                sse = 0.0
                resid = []
                for t in range(1, len(ys)):
                    fc = level + phi * trend
                    err = ys[t] - fc
                    sse += err * err
                    resid.append(err)
                    prev_level = level
                    level = alpha * ys[t] + (1 - alpha) * (level + phi * trend)
                    trend = beta * (level - prev_level) + (1 - beta) * phi * trend
                if best is None or sse < best[0]:
                    best = (sse, alpha, beta, phi, level, trend, resid)

    sse, alpha, beta, phi, level, trend, resid = best
    sigma = float(np.std(resid, ddof=1)) if len(resid) > 2 else float(np.std(resid))
    z = 1.645
    fcs, los, his = [], [], []
    for h in range(1, horizon + 1):
        if abs(phi - 1) < 1e-9:
            growth = h * trend
        else:
            growth = trend * phi * (1 - phi**h) / (1 - phi)
        point_log = level + growth
        half = z * sigma * math.sqrt(h)
        fcs.append(max(0.0, float(np.expm1(point_log))))
        los.append(max(0.0, float(np.expm1(point_log - half))))
        his.append(max(0.0, float(np.expm1(point_log + half))))
    return fcs, los, his


def chart_start_week(latest: tuple[int, int]) -> tuple[int, int]:
    """Season window from W21, but always at least the latest 26 weeks."""
    year, week = latest
    season = (year, 21) if week >= 21 else (year - 1, 21)
    return min(season, add_weeks(year, week, -25))


def _week_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Calendar-week gap between two year-week pairs (absolute)."""
    if a == b:
        return 0
    if a > b:
        a, b = b, a
    n = 0
    cur = a
    while cur < b:
        cur = add_weeks(*cur, 1)
        n += 1
        if n > 200:
            break
    return n


def make_x_ticks(weeks: list, latest: tuple[int, int], step: int = 8, min_gap: int = 4) -> list[dict]:
    """Ticks from the actual chart week list: first, every `step`, latest, last.

    Every tick is guaranteed to fall inside `weeks`. Year prefix is added
    when the tick year differs from latest's year. A non-anchor step tick
    that sits fewer than `min_gap` weeks from an anchor (first / latest /
    last) is dropped so labels do not overlap (e.g. W45 next to W46).
    """
    if not weeks:
        return []
    anchors = {weeks[0], weeks[-1]}
    if latest in weeks:
        anchors.add(latest)
    wanted = set(anchors)
    for i in range(0, len(weeks), step):
        cand = weeks[i]
        if any(_week_distance(cand, a) < min_gap for a in anchors):
            continue
        wanted.add(cand)
    ticks = []
    for y, w in weeks:
        if (y, w) in wanted:
            ticks.append({
                "year": y,
                "week": w,
                "label": label_week(y, w, latest[0]),
            })
    return ticks


def build_pathogen_chart(week_map: dict, latest: tuple[int, int], chart_end: tuple[int, int], start=None):
    start = start or chart_start_week(latest)
    weeks = list(iter_weeks(start, chart_end))
    observed_end = latest
    # Forecast from the full history in week_map, not only the display window.
    future = [yw for yw in weeks if yw > observed_end]
    fcs, los, his = damped_holt_forecast(week_map, latest, horizon=len(future))
    ma = ma3([week_map.get(yw) if yw <= observed_end else None for yw in weeks])

    points = []
    fi = 0
    for i, (y, w) in enumerate(weeks):
        rec = {
            "year": y,
            "week": w,
            "year_week": format_yw(y, w),
            "label": label_week(y, w, latest[0]),
            "observed": week_map.get((y, w)) if (y, w) <= observed_end else None,
            "ma3": (round(ma[i], 2) if ma[i] is not None and (y, w) <= observed_end else None),
            "forecast": None,
            "forecast_lo": None,
            "forecast_hi": None,
        }
        if (y, w) > observed_end:
            rec["forecast"] = round(fcs[fi], 1) if fcs[fi] is not None else None
            rec["forecast_lo"] = round(los[fi], 1) if los[fi] is not None else None
            rec["forecast_hi"] = round(his[fi], 1) if his[fi] is not None else None
            fi += 1
        points.append(rec)
    return points, weeks


KEY_URI_PATHOGENS = (
    "Influenza",
    "SARS-CoV-2",
    "RSV",
    "Adeno",
    "S. pneumoniae",
    "Parainfluenza",
)

OTHER_SHARE_NOTE = (
    "LARS 未公布送驗總檢體數。佔比分母為同週六項重點病原"
    "（Influenza、SARS-CoV-2、RSV、Adeno、S. pneumoniae、Parainfluenza）陽性檢體合計。"
)


def count_delta(current, prev) -> int | None:
    """Week-over-week change in counts. 0 is a real value (unchanged)."""
    if current is None or prev is None:
        return None
    return int(current) - int(prev)


def share_pct(count, denominator) -> float | None:
    """Percentage of denominator, 1 decimal. None if the denominator is missing or 0."""
    if count is None or denominator is None or denominator == 0:
        return None
    return round(100.0 * float(count) / float(denominator), 1)


def pathogen_card(series: dict, latest: tuple[int, int], prev: tuple[int, int], denom: int | None) -> dict:
    current = value_at(series, latest)
    prior = value_at(series, prev)
    return {
        "count": current,
        "prior_week": prior,
        "delta": count_delta(current, prior),
        "share_of_key_positives_pct": share_pct(current, denom),
    }


def value_at(m: dict, yw: tuple[int, int]) -> int | None:
    return m.get(yw)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    unavailable: list[str] = []
    retrieved = datetime.now(TZ)
    sources = []

    # --- LARS flu (A/B/untyped) ---
    flu_html = get(f"{BASE}/misc/lars?id=flu")
    flu_charts = parse_highcharts(flu_html)
    flu_a = to_week_map(*series_by_name(flu_charts, "A"))
    flu_b = to_week_map(*series_by_name(flu_charts, "B"))
    flu_u = to_week_map(*series_by_name(flu_charts, "Untyped"))
    if not flu_a:
        raise RuntimeError("failed to parse LARS influenza A series")
    sources.append({
        "name": "疾管署 LARS 流感病毒陽性檢體",
        "url": f"{BASE}/misc/lars?id=flu",
        "update": "每週一、二清晨",
    })

    # --- LARS URI pathogens ---
    uri_html = get(f"{BASE}/misc/lars?id=uri")
    uri_charts = parse_highcharts(uri_html)
    uri = {}
    for name in ("Influenza", "SARS-CoV-2", "RSV", "Adeno", "S. pneumoniae", "Parainfluenza"):
        cats, data = series_by_name(uri_charts, name)
        uri[name] = to_week_map(cats, data)
        if not uri[name]:
            raise RuntimeError(f"failed to parse LARS series {name!r} (page layout may have changed)")
    sources.append({
        "name": "疾管署 LARS 重點呼吸道病原體陽性檢體",
        "url": f"{BASE}/misc/lars?id=uri",
        "update": "每週一、二清晨",
    })

    # Trend and weekly totals come from the URI Influenza series (longer
    # history, from 2025 W01). A/B/untyped from the flu page are only used
    # for the latest-week subtype line. Latest complete week is the last
    # week present on URI Influenza; that week must also exist on the
    # other required series.
    flu_total = uri["Influenza"]
    latest = max(flu_total)
    prev = add_weeks(*latest, -1)
    prev2 = add_weeks(*latest, -2)
    chart_end = add_weeks(*latest, FORECAST_HORIZON)

    flu_n = flu_total[latest]
    for name, series in uri.items():
        if latest not in series:
            raise RuntimeError(f"LARS {name} has no value for latest week {latest}")
    if latest not in flu_a or latest not in flu_b or latest not in flu_u:
        raise RuntimeError(f"LARS influenza A/B/Untyped missing latest week {latest}")
    typed_sum = flu_a.get(latest, 0) + flu_b.get(latest, 0) + flu_u.get(latest, 0)
    if typed_sum != flu_n:
        raise RuntimeError(
            f"LARS influenza total mismatch at {latest}: A+B+Untyped={typed_sum} vs URI Influenza={flu_n}"
        )

    covid_map = uri["SARS-CoV-2"]
    covid_n = value_at(covid_map, latest)
    covid_prev = value_at(covid_map, prev)
    if covid_n is None:
        raise RuntimeError(f"LARS SARS-CoV-2 has no value for latest week {latest}")

    flu_pct = round1(pct_vs_prior_two_week_avg(flu_n, flu_total.get(prev), flu_total.get(prev2)))
    covid_pct = round1(pct_vs_prior_two_week_avg(covid_n, covid_prev, value_at(covid_map, prev2)))

    flu_word = trend_word(flu_n, flu_total.get(prev))
    covid_word = trend_word(covid_n, covid_prev)
    # Clarify that 持平/上升/下降 is week-over-week, not the 2-week-average badge.
    headline = f"流感較上週{flu_word}；COVID 較上週{covid_word}。"

    flu_points, flu_weeks = build_pathogen_chart(flu_total, latest, chart_end)
    covid_points, _covid_weeks = build_pathogen_chart(covid_map, latest, chart_end)
    flu_fc_w_next = next((p for p in flu_points if p["forecast"] is not None), None)
    covid_fc_w_next = next((p for p in covid_points if p["forecast"] is not None), None)

    # --- NHI visits ---
    visits = {}
    try:
        ili_op = fetch_nhi_visits(4, "[1]", "門診", "類流感")
        ili_oped = fetch_nhi_visits(4, "[1][4]", "門診 + 急診", "類流感")
        cov_op = fetch_nhi_visits(11, "[1]", "門診", "新冠")
        cov_oped = fetch_nhi_visits(11, "[1][4]", "門診 + 急診", "新冠")
        # 急診 derived: 門急診 − 門診 (matches NIDSS arithmetic and the reference)
        ili_er = {k: ili_oped[k] - ili_op[k] for k in ili_oped if k in ili_op}
        cov_er = {k: cov_oped[k] - cov_op[k] for k in cov_oped if k in cov_op}

        def pack(m: dict, key: str):
            cur, p, p2 = value_at(m, latest), value_at(m, prev), value_at(m, prev2)
            if cur is None:
                unavailable.append(key)
            return {
                "count": cur,
                "pct_vs_prior_2w_avg": round1(pct_vs_prior_two_week_avg(cur, p, p2)),
                "prior_week": p,
                "prior2_week": p2,
            }

        visits = {
            "ili_oped": pack(ili_oped, "類流感門急診"),
            "ili_er": pack(ili_er, "類流感急診"),
            "covid_oped": pack(cov_oped, "COVID門急診"),
            "covid_er": pack(cov_er, "COVID急診"),
        }
        for key, rec in visits.items():
            if rec["count"] is None:
                raise RuntimeError(f"{key} missing for latest week {latest}")
            if rec["count"] < 0:
                raise RuntimeError(f"{key} is negative ({rec['count']}) at {latest}")
        sources.append({
            "name": "次級健保資料 門診及門診+急診就診人次（急診＝門急診−門診）",
            "url": f"{BASE}/Cdcwnh/Cdcwnh01?disease=4",
            "update": "每週一、二清晨",
        })
    except Exception as e:
        raise RuntimeError(f"健保就診資料擷取失敗（NIDSS 版型可能已改）：{e}") from e

    # --- COVID severe (onset week) & deaths (death week) ---
    severe_n = None
    death_n = None
    try:
        sc_html = get(f"{BASE}/nndss/disease?id=19SC")
        sc_charts = parse_highcharts(sc_html)
        sc_map = to_week_map(*series_by_name(sc_charts, "確定病例數"))
        severe_n = value_at(sc_map, latest)
        sources.append({
            "name": "新冠併發重症（依發病週）",
            "url": f"{BASE}/nndss/disease?id=19SC",
            "update": "法定傳染病統計每日上午 8:30 更新；本頁採與 LARS 同一完整週",
        })
        if severe_n is None:
            raise RuntimeError(f"COVID 重症 has no value for latest week {latest}")
    except Exception as e:
        raise RuntimeError(f"COVID 重症資料擷取失敗（NIDSS 版型可能已改）：{e}") from e

    try:
        d_html = get(f"{BASE}/nndss/deadmap?id=19SC")
        d_charts = parse_highcharts(d_html)
        d_map = to_week_map(*series_by_name(d_charts, "死亡病例數"))
        death_n = value_at(d_map, latest)
        sources.append({
            "name": "新冠併發重症死亡（依死亡週）",
            "url": f"{BASE}/nndss/deadmap?id=19SC",
            "update": "法定傳染病統計每日上午 8:30 更新；本頁採與 LARS 同一完整週",
        })
        if death_n is None:
            raise RuntimeError(f"COVID 死亡 has no value for latest week {latest}")
    except Exception as e:
        raise RuntimeError(f"COVID 死亡資料擷取失敗（NIDSS 版型可能已改）：{e}") from e

    # LARS URI publishes positive counts only, not total specimens tested.
    key_positive_total = 0
    for name in KEY_URI_PATHOGENS:
        n = value_at(uri.get(name) or {}, latest)
        if n is None:
            raise RuntimeError(f"LARS {name} has no value for latest week {latest}")
        key_positive_total += n

    other = {
        "rsv": pathogen_card(uri["RSV"], latest, prev, key_positive_total),
        "adenovirus": pathogen_card(uri["Adeno"], latest, prev, key_positive_total),
        "s_pneumoniae": pathogen_card(uri["S. pneumoniae"], latest, prev, key_positive_total),
        "parainfluenza": pathogen_card(uri["Parainfluenza"], latest, prev, key_positive_total),
        "denominator": {
            "key": "key_pathogen_positives",
            "label": "六項重點病原陽性檢體合計",
            "value": key_positive_total,
            "note": OTHER_SHARE_NOTE,
        },
    }
    for k, v in other.items():
        if k == "denominator":
            continue
        if v.get("count") is None:
            raise RuntimeError(f"LARS {k} has no value for latest week {latest}")

    latest_sun = week_sunday(*latest)
    latest_sat = latest_sun + timedelta(days=6)
    x_ticks = make_x_ticks(flu_weeks, latest)

    next_yw = add_weeks(*latest, 1)

    payload = {
        "meta": {
            "year": latest[0],
            "week": latest[1],
            "week_start": latest_sun.isoformat(),
            "week_end": latest_sat.isoformat(),
            "retrieved_at": retrieved.strftime("%Y-%m-%d %H:%M"),
            "timezone": "Asia/Taipei",
            "forecast_method": "阻尼趨勢 Holt 法（對數尺度，最近 16 週，往後 4 週）的探索性預測，陰影為未校準的粗估區間，並非 TimesFM",
            "forecast_not_timesfm": True,
            "wow_hold_threshold": HOLD_THRESHOLD,
            "pct_definition": "與前兩週平均相比",
            "er_definition": "急診人次 = 門診+急診 − 門診（次級健保）",
            "other_share_definition": OTHER_SHARE_NOTE,
            "cors": "NIDSS 未回傳 Access-Control-Allow-Origin，GitHub Pages 前端無法直接抓取，需週更 data.json",
            "unavailable": unavailable,
            "sources": sources,
            "links": {
                "lars": f"{BASE}/misc/lars?id=uri",
                "nhi": f"{BASE}/Cdcwnh/Cdcwnh01?disease=4",
                "covid_severe": f"{BASE}/nndss/disease?id=19SC",
                "covid_deaths": f"{BASE}/nndss/deadmap?id=19SC",
            },
        },
        "headline": headline,
        "flu": {
            "positives": flu_n,
            "a": flu_a.get(latest, 0),
            "b": flu_b.get(latest, 0),
            "untyped": flu_u.get(latest, 0),
            "pct_vs_prior_2w_avg": flu_pct,
            "prior_week": flu_total.get(prev),
            "trend_word": flu_word,
            "forecast_next_week_label": f"第 {next_yw[1]} 週",
            "forecast_next_week": (round(flu_fc_w_next["forecast"]) if flu_fc_w_next and flu_fc_w_next["forecast"] is not None else None),
            "forecast_next_week_lo": (round(flu_fc_w_next["forecast_lo"]) if flu_fc_w_next and flu_fc_w_next["forecast_lo"] is not None else None),
            "forecast_next_week_hi": (round(flu_fc_w_next["forecast_hi"]) if flu_fc_w_next and flu_fc_w_next["forecast_hi"] is not None else None),
            "points": flu_points,
        },
        "covid": {
            "positives": covid_n,
            "prior_week": covid_prev,
            "prior_week_number": prev[1],
            "pct_vs_prior_2w_avg": covid_pct,
            "trend_word": covid_word,
            "forecast_next_week_label": f"第 {next_yw[1]} 週",
            "forecast_next_week": (round(covid_fc_w_next["forecast"]) if covid_fc_w_next and covid_fc_w_next["forecast"] is not None else None),
            "forecast_next_week_lo": (round(covid_fc_w_next["forecast_lo"]) if covid_fc_w_next and covid_fc_w_next["forecast_lo"] is not None else None),
            "forecast_next_week_hi": (round(covid_fc_w_next["forecast_hi"]) if covid_fc_w_next and covid_fc_w_next["forecast_hi"] is not None else None),
            "points": covid_points,
        },
        "visits": visits,
        "covid_severe": {
            "cases_by_onset_week": severe_n,
            "deaths_by_death_week": death_n,
        },
        "other": other,
        "chart_x_ticks": x_ticks,
    }

    if unavailable:
        raise RuntimeError("required metrics unavailable: " + ", ".join(unavailable))

    if OUT.exists():
        try:
            previous = json.loads(OUT.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            previous = None
            print(f"warning: existing data.json is not valid JSON ({e}); it will be replaced if this run succeeds")
        if previous and isinstance(previous, dict):
            prev_meta = previous.get("meta") or {}
            if "year" in prev_meta and "week" in prev_meta:
                old_yw = (int(prev_meta["year"]), int(prev_meta["week"]))
                if latest < old_yw:
                    raise RuntimeError(
                        f"fetched week {latest[0]}-W{latest[1]} is older than "
                        f"existing {old_yw[0]}-W{old_yw[1]}; keeping the current file"
                    )
            if _payload_same_ignoring_retrieved(previous, payload):
                print(f"No epidemiological change (still {latest[0]} W{latest[1]}); kept {OUT}")
                _print_summary(payload, headline, flu_n, flu_pct, covid_n, covid_prev, covid_pct, visits, severe_n, death_n, other, latest_sun, latest_sat)
                return 0

    _atomic_write_json(OUT, payload)
    print(f"Wrote {OUT}")
    _print_summary(payload, headline, flu_n, flu_pct, covid_n, covid_prev, covid_pct, visits, severe_n, death_n, other, latest_sun, latest_sat)
    return 0


def _without_retrieved(obj: dict) -> dict:
    clone = json.loads(json.dumps(obj))
    if isinstance(clone.get("meta"), dict):
        clone["meta"].pop("retrieved_at", None)
    return clone


def _payload_same_ignoring_retrieved(old: dict, new: dict) -> bool:
    return _without_retrieved(old) == _without_retrieved(new)


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _print_summary(payload, headline, flu_n, flu_pct, covid_n, covid_prev, covid_pct, visits, severe_n, death_n, other, latest_sun, latest_sat) -> None:
    latest_y = payload["meta"]["year"]
    latest_w = payload["meta"]["week"]
    print(f"Latest complete week: {latest_y} W{latest_w} ({latest_sun}–{latest_sat})")
    print(f"Retrieved: {payload['meta']['retrieved_at']} Asia/Taipei")
    print(f"Headline: {headline}")
    print(
        f"Flu {flu_n} (A {payload['flu']['a']} · B {payload['flu']['b']} · 未分型 {payload['flu']['untyped']})  "
        f"{flu_pct}%  next-week fc {payload['flu']['forecast_next_week']}"
    )
    print(
        f"COVID {covid_n} (prev {covid_prev})  {covid_pct}%  "
        f"next-week fc {payload['covid']['forecast_next_week']}"
    )
    v = visits
    if v.get("ili_oped", {}).get("count") is not None:
        print(f"ILI 門急診 {v['ili_oped']['count']} {v['ili_oped']['pct_vs_prior_2w_avg']}%  急診 {v['ili_er']['count']} {v['ili_er']['pct_vs_prior_2w_avg']}%")
        print(f"COV 門急診 {v['covid_oped']['count']} {v['covid_oped']['pct_vs_prior_2w_avg']}%  急診 {v['covid_er']['count']} {v['covid_er']['pct_vs_prior_2w_avg']}%")
    print(f"Severe {severe_n}  deaths {death_n}")
    print(f"Other {other}")


def _self_test() -> int:
    """Tick and forecast unit tests (no network)."""
    failures = []

    def check(latest):
        start = chart_start_week(latest)
        end = add_weeks(*latest, FORECAST_HORIZON)
        weeks = list(iter_weeks(start, end))
        ticks = make_x_ticks(weeks, latest)
        weekset = set(weeks)
        for tk in ticks:
            yw = (tk["year"], tk["week"])
            if yw not in weekset:
                failures.append(f"tick {yw} not in chart weeks for latest={latest}")
        if ticks[0]["year"] != weeks[0][0] or ticks[0]["week"] != weeks[0][1]:
            failures.append(f"first tick != first week for latest={latest}")
        if ticks[-1]["year"] != weeks[-1][0] or ticks[-1]["week"] != weeks[-1][1]:
            failures.append(f"last tick != last week for latest={latest}")
        if latest in weekset and not any(tk["year"] == latest[0] and tk["week"] == latest[1] for tk in ticks):
            failures.append(f"latest {latest} missing from ticks")
        # year prefix when tick year differs
        for tk in ticks:
            if tk["year"] != latest[0] and not tk["label"].startswith(f"{tk['year'] % 100:02d}W"):
                failures.append(f"missing year prefix on {tk}")
        return weeks, ticks

    for latest in ((2026, 46), (2027, 3), (2027, 22)):
        weeks, ticks = check(latest)
        print(f"ticks latest={latest}: {[(tk['label']) for tk in ticks]}  (n_weeks={len(weeks)})")

    # Gap handling: do not glue across a hole
    latest = (2026, 46)
    week_map = {add_weeks(*latest, -i): 100 + i for i in range(0, 16)}
    del week_map[add_weeks(*latest, -3)]  # a hole
    series = recent_contiguous_series(week_map, latest, 16)
    if series is None or len(series) != 16:
        failures.append(f"expected 16 filled points, got {None if series is None else len(series)}")
    else:
        # the hole is 3 weeks before latest → index 12 (0..15, last is latest)
        hole = series[12]
        left, right = series[11], series[13]
        if not (min(left, right) <= hole <= max(left, right)):
            failures.append(f"interpolated hole {hole} not between {left} and {right}")

    # Window-only vs full history: a short chart window (W21–W22) still forecasts
    latest = (2026, 22)
    hist = {add_weeks(*latest, -i): 400 + i * 10 for i in range(0, 20)}
    pts, weeks = build_pathogen_chart(hist, latest, add_weeks(*latest, FORECAST_HORIZON))
    future = [p for p in pts if p["forecast"] is not None]
    if len(future) != FORECAST_HORIZON:
        failures.append(f"W22 should have {FORECAST_HORIZON} forecast weeks, got {len(future)}")
    if any(p["forecast"] is None for p in future):
        failures.append("W22 forecast contained None")

    # Log-scale forecast should stay positive on a declining series
    latest = (2026, 46)
    declining = {add_weeks(*latest, -i): max(1, 80 - i * 5) for i in range(0, 16)}
    fcs, los, his = damped_holt_forecast(declining, latest, horizon=4)
    if any(v is None or v < 0 for v in fcs):
        failures.append(f"declining log-Holt produced non-positive points: {fcs}")

    # Fail-safe helpers: compare ignores retrieved_at; atomic write leaves no tmp.
    a = {"meta": {"retrieved_at": "2026-11-24 13:27", "week": 46}, "flu": {"positives": 995}}
    b = {"meta": {"retrieved_at": "2026-11-24 14:00", "week": 46}, "flu": {"positives": 995}}
    c = {"meta": {"retrieved_at": "2026-11-24 13:27", "week": 46}, "flu": {"positives": 994}}
    if not _payload_same_ignoring_retrieved(a, b):
        failures.append("identical epidemiology should match when only retrieved_at differs")
    if _payload_same_ignoring_retrieved(a, c):
        failures.append("changing positives should not be treated as unchanged")
    if parse_highcharts("<html>no charts here</html>") != []:
        failures.append("junk HTML should yield no Highcharts series")

    import tempfile
    tmpdir = Path(tempfile.mkdtemp())
    out = tmpdir / "data.json"
    _atomic_write_json(out, a)
    if json.loads(out.read_text(encoding="utf-8")) != a:
        failures.append("atomic write did not persist the payload")
    leftovers = list(tmpdir.glob("*.tmp"))
    if leftovers:
        failures.append(f"atomic write left tmp files: {leftovers}")

    if count_delta(44, 44) != 0:
        failures.append("unchanged week should be delta 0, not None")
    if count_delta(44, 39) != 5 or count_delta(33, 47) != -14:
        failures.append("count_delta sign or magnitude is wrong")
    if count_delta(10, None) is not None or count_delta(None, 10) is not None:
        failures.append("count_delta should be None when a week is missing")
    if share_pct(44, 1215) != 3.6:
        failures.append(f"share_pct 44/1215 should be 3.6, got {share_pct(44, 1215)}")
    if share_pct(0, 1215) != 0.0:
        failures.append("share_pct of 0 should be 0.0")
    if share_pct(10, 0) is not None or share_pct(10, None) is not None:
        failures.append("share_pct should be None when denominator is missing or 0")
    card = pathogen_card({(2026, 46): 44, (2026, 45): 44}, (2026, 46), (2026, 45), 1215)
    if card != {"count": 44, "prior_week": 44, "delta": 0, "share_of_key_positives_pct": 3.6}:
        failures.append(f"pathogen_card fixture mismatch: {card}")

    if failures:
        for f in failures:
            print("FAIL:", f)
        return 1
    print("self-test: all passed")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("--self-test", "self-test"):
        sys.exit(_self_test())
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        if OUT.exists():
            print(f"Kept existing {OUT}", file=sys.stderr)
        else:
            print("No existing data.json to keep", file=sys.stderr)
        sys.exit(1)
