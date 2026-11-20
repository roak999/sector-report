"""「今日注目する銘柄（日本）」の毎日の記録と、その後の値動きの追跡。

- 記録：history/picks.json に、選出日ごとの3つのリスト（総合上位／上昇初動・ブレイク候補／タイミング良好）を保存する。
- 追跡：選出日の終値を基準に、その後の終値（最大12営業日分）とTOPIXを、画面用データにまとめる。
- 画面（report_template.html の stockFocusHtml）と同じ条件で選ぶ。条件を変えるときは両方そろえること。
- 非公開版（--private）では記録しない（保有銘柄が公開リポジトリに残らないようにするため）。
"""
import json
from pathlib import Path

import pandas as pd

HIST_PATH = Path(__file__).with_name("history") / "picks.json"
HORIZON = 10          # 追跡する営業日数（約2週間）
KEEP = HORIZON + 2    # 画面用に持つ終値の数（選出日＋翌日基準の計算に必要な分）
MAX_DAYS = 45         # 画面に出す選出日の最大数（古いものは記録に残るが表示しない）
LISTS = {"top": "総合評価の上位", "brk": "上昇初動・ブレイクアウト候補", "ent": "総合60点以上でタイミング良好"}


def make_picks(rows: list) -> dict:
    """rows は総合点の高い順。画面の「今日注目する銘柄」と同じ条件。"""
    top = rows[:5]
    brk = sorted((r for r in rows if 2 in r["tags"]), key=lambda r: -(r["entry"] or 0))[:5]
    ent = sorted((r for r in rows if r["entry"] is not None and r["total"] is not None and r["total"] >= 60),
                 key=lambda r: -r["entry"])[:5]
    f = lambda xs: [[r["code"], r["total"]] for r in xs]
    return {"top": f(top), "brk": f(brk), "ent": f(ent)}


def load_history(path: Path = HIST_PATH) -> dict:
    try:
        h = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(h, dict) and isinstance(h.get("days"), dict):
            return h
    except (OSError, ValueError):
        pass
    return {"version": 1, "days": {}}


def save_history(h: dict, path: Path = HIST_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h["days"] = dict(sorted(h["days"].items()))
    path.write_text(json.dumps(h, ensure_ascii=False, indent=1), encoding="utf-8")


def record(h: dict, date: str, picks: dict, source: str, overwrite_live: bool = True) -> bool:
    """date は YYYY-MM-DD。source は "live"（その日に実際に記録）か "replay"（過去分を後から再現）。
    実記録は再現で上書きしない。戻り値＝書き込んだか。"""
    old = h["days"].get(date)
    if old and old.get("source") == "live" and source != "live":
        return False
    if old and old.get("source") == "live" and not overwrite_live:
        return False
    h["days"][date] = {"source": source, "lists": picks}
    return True


def _uni_returns(prices: dict, universe: list, cal, days) -> dict:
    """代表銘柄の等金額平均の騰落率（%）。基準が選出日の終値（"0"）／翌営業日の終値（"1"）の2通り。"""
    cols = {}
    for t in universe:
        df = prices.get(t)
        if df is not None:
            cols[t] = df["Close"].reindex(cal).ffill(limit=3).reindex(days)
    P = pd.DataFrame(cols)
    out = {}
    for b in (0, 1):
        vals = []
        for k in range(HORIZON + 1):
            i = k + b
            if i >= len(P):
                vals.append(None)
                continue
            r = (P.iloc[i] / P.iloc[b] - 1).dropna()
            vals.append(round(float(r.mean() * 100), 3) if len(r) >= 20 else None)
        out[str(b)] = vals
    return out


def build_watch(h: dict, prices: dict, topix: pd.Series, names: dict, sectors: dict, universe: list = ()) -> dict:
    """画面用。選出日ごとに、選ばれた銘柄の「選出日以降の終値」とTOPIX・代表銘柄の平均を付ける。"""
    cal = topix.dropna().index
    out_days = []
    for date in sorted(h["days"], reverse=True)[:MAX_DAYS]:
        d = pd.Timestamp(date)
        if d not in cal:
            continue
        pos = cal.get_loc(d)
        days = cal[pos:pos + KEEP]
        entry = h["days"][date]
        tags = {}
        for k, lst in entry["lists"].items():
            for code, total in lst:
                tags.setdefault(code, {"lists": [], "total": total})["lists"].append(k)
        picks = []
        for code, info in tags.items():
            df = prices.get(code + ".T")
            if df is None:
                continue
            c = df["Close"].reindex(cal).ffill(limit=3).reindex(days)
            if pd.isna(c.iloc[0]):
                continue
            picks.append({"code": code, "name": names.get(code, code), "sector": sectors.get(code, ""),
                          "lists": info["lists"], "total": info["total"],
                          "c": [None if pd.isna(x) else round(float(x), 2) for x in c]})
        if not picks:
            continue
        tp = topix.reindex(days)
        out_days.append({"date": date, "source": entry.get("source", "live"),
                         "dates": [x.strftime("%Y-%m-%d") for x in days],
                         "topix": [None if pd.isna(x) else round(float(x), 2) for x in tp],
                         "uni": _uni_returns(prices, list(universe), cal, days) if universe else None,
                         "picks": picks})
    return {"horizon": HORIZON, "lists": LISTS, "days": out_days,
            "firstDate": min(h["days"]) if h["days"] else None,
            "liveDays": sum(1 for v in h["days"].values() if v.get("source") == "live"),
            "replayDays": sum(1 for v in h["days"].values() if v.get("source") == "replay")}
