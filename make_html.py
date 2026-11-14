"""スマホでも見られるHTMLレポートを書き出す。

使い方:
    python make_html.py                 # → report.html を作成（sectors.py の全グループ）
    python make_html.py --out docs/index.html
"""
import argparse
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

import data
from indicators import (CANDLE_BASICS, CANDLE_HELP, DEFAULT_WEIGHTS, INDICATORS, candle_help,
                        composite, describe_all, macd, score_all)
from sectors import GROUP_OPTIONS, REGIONS

HISTORY_DAYS = 60   # 推移ヒートマップの日数
CHART_DAYS = 120    # 詳細チャートの日数
TEMPLATE = Path(__file__).with_name("report_template.html")


def _num(v, nd=2):
    if v is None or pd.isna(v) or (isinstance(v, float) and math.isinf(v)):
        return None
    return round(float(v), nd)


def build_group(group: str, items: dict, prices: dict, weights: dict, volume_mode: str):
    sector_data = data.build_sector_data(items, prices, volume_mode)
    if not sector_data:
        return None, {}

    scores, descs, comp = {}, {}, {}
    for name, (ohlc, turnover) in sector_data.items():
        scores[name] = score_all(ohlc, turnover)
        descs[name] = describe_all(ohlc, turnover)
        comp[name] = composite(scores[name], weights)
    comp_df = pd.DataFrame(comp).dropna(how="all")

    # 今日（各銘柄の最新日。市場ごとに休日が違うので日付も持たせる）
    today = []
    for name in sector_data:
        c = comp[name].dropna()
        if c.empty:
            continue
        dt = c.index[-1]
        today.append({
            "name": name,
            "date": dt.strftime("%m/%d"),
            "total": _num(c.iloc[-1], 0),
            "scores": {k: _num(scores[name].at[dt, k], 0) for k in INDICATORS},
            "texts": {k: descs[name].at[dt, k] for k in INDICATORS},
        })
    if GROUP_OPTIONS.get(group, {}).get("sort", True):
        today.sort(key=lambda r: -(r["total"] if r["total"] is not None else -999))

    # 推移
    h = comp_df.tail(HISTORY_DAYS)
    history = {
        "dates": [d.strftime("%m/%d") for d in h.index],
        "rows": [{"name": r["name"], "values": [_num(v, 0) for v in h[r["name"]]]} for r in today],
    }

    # 詳細
    detail = {}
    for name, (ohlc, turnover) in sector_data.items():
        if name not in comp or comp[name].dropna().empty:
            continue
        cfg = items[name]
        c = ohlc["Close"]
        line, sig, hist = macd(c)
        ma25, ma75 = c.rolling(25).mean(), c.rolling(75).mean()
        d = ohlc.tail(CHART_DAYS)
        ix = d.index
        if cfg.get("volume") == "shares":
            vol_label, vol_vals = "出来高（百万株）", [_num(v / 1e6, 1) for v in turnover.reindex(ix)]
        else:
            vol_label, vol_vals = "売買代金（億円）", [_num(v / 1e8, 1) for v in turnover.reindex(ix)]
        candle_txt = descs[name]["ローソク足"].reindex(ix)
        candle_tip = []
        for dt, row, tx in zip(ix, d.itertuples(), candle_txt):
            title, body = candle_help(tx)
            candle_tip.append({"t": title, "b": body})
        last10 = []
        for dt in comp[name].dropna().tail(10).index[::-1]:
            last10.append({
                "date": dt.strftime("%m/%d"),
                "total": _num(comp[name].get(dt), 0),
                "scores": {k: _num(scores[name].at[dt, k], 0) for k in INDICATORS},
                "texts": {k: descs[name].at[dt, k] for k in INDICATORS},
            })
        detail[name] = {
            "group": group,
            "dates": [x.strftime("%Y-%m-%d") for x in ix],
            "o": [_num(v) for v in d["Open"]], "h": [_num(v) for v in d["High"]],
            "l": [_num(v) for v in d["Low"]], "c": [_num(v) for v in d["Close"]],
            "ma25": [_num(v) for v in ma25.loc[ix]], "ma75": [_num(v) for v in ma75.loc[ix]],
            "volLabel": vol_label, "vol": vol_vals,
            "macd": [_num(v, 3) for v in line.loc[ix]], "signal": [_num(v, 3) for v in sig.loc[ix]],
            "hist": [_num(v, 3) for v in hist.loc[ix]],
            "candle": candle_tip,
            "last10": last10,
        }

    sorted_ = GROUP_OPTIONS.get(group, {}).get("sort", True)
    return {"name": group, "sorted": sorted_, "today": today, "history": history}, detail


def build_payload(groups: list, weights: dict, volume_mode: str) -> dict:
    all_items = {}
    for g in groups:
        all_items.update(REGIONS[g])
    prices = data.load_prices(data.tickers_for(all_items))

    out_groups, detail = [], {}
    for g in groups:
        grp, det = build_group(g, REGIONS[g], prices, weights, volume_mode)
        if grp:
            out_groups.append(grp)
            detail.update(det)
    if not out_groups:
        raise SystemExit("株価データを取得できませんでした。ネット接続か銘柄コードを確認してください。")

    jst = timezone(timedelta(hours=9))
    return {
        "generated": datetime.now(jst).strftime("%Y-%m-%d %H:%M"),
        "indicators": INDICATORS,
        "weights": weights,
        "groups": out_groups,
        "detail": detail,
        "candleBasics": CANDLE_BASICS,
        "candleHelp": CANDLE_HELP,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="report.html")
    ap.add_argument("--groups", nargs="*", default=list(REGIONS), choices=list(REGIONS),
                    help="載せるグループ（省略時は全部）")
    ap.add_argument("--volume", default="構成銘柄の売買代金合計",
                    choices=["構成銘柄の売買代金合計", "ETF自体の出来高"])
    args = ap.parse_args()

    payload = build_payload(args.groups, DEFAULT_WEIGHTS, args.volume)
    html = TEMPLATE.read_text(encoding="utf-8").replace(
        "/*__DATA__*/null", json.dumps(payload, ensure_ascii=False).replace("</", "<\\/"))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"作成しました: {out.resolve()}（{', '.join(g['name'] for g in payload['groups'])}）")


if __name__ == "__main__":
    main()
