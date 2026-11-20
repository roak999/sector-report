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
from market import build_market, build_world
from names import NAMES
from stocks import TYPES, W_ENTRY, W_RISK, W_STOCK, W_TOTAL, analyze, load_watchlist
from sector_plus import build_sector_plus
from sectors import GROUP_OPTIONS, MARKET_INDEX, MARKETS, REGIONS, TOPIX_PROXY, UNIVERSE, US_UNIVERSE, WORLD

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


def build_payload(groups: list, weights: dict, volume_mode: str, private: bool = False,
                  record_history: bool = False) -> dict:
    all_items = {}
    for g in groups:
        all_items.update(REGIONS[g])
    universe = UNIVERSE if "日本（TOPIX-17業種）" in groups else {}
    jp_flat = sorted({t for ts in universe.values() for t in ts})
    world_tickers = {t for _, t, _ in WORLD}
    index_tickers = world_tickers | {t for m in MARKETS.values() for _, t in m["indices"]} \
        | {m["vi"][1] for m in MARKETS.values() if m.get("vi")}
    tickers = set(data.tickers_for(all_items)) | {MARKET_INDEX, TOPIX_PROXY} | index_tickers
    tickers |= set(jp_flat) | set(US_UNIVERSE)
    watch = load_watchlist(Path(__file__).with_name("watchlist.txt"))
    if private:   # 非公開の登録銘柄（ローカル専用ビルドのときだけ読む）
        pw = Path(__file__).with_name("watchlist_private.txt")
        if pw.exists():
            watch = watch + load_watchlist(pw)
    tickers |= {t for t, _, _ in watch}
    prices, failed = data.load_prices_ex(sorted(tickers))
    prices, stale = data.drop_stale(prices, MARKET_INDEX)
    if failed or stale:
        print(f"除外: 取得失敗 {len(failed)}件 {failed} / 更新停止 {len(stale)}件 {stale}")

    out_groups, detail = [], {}
    for g in groups:
        grp, det = build_group(g, REGIONS[g], prices, weights, volume_mode)
        if grp:
            out_groups.append(grp)
            detail.update(det)
    if not out_groups:
        raise SystemExit("株価データを取得できませんでした。ネット接続か銘柄コードを確認してください。")

    # ---- 市場分析・セクター拡張（失敗しても既存画面は出す） ----
    markets, sector_payload = {}, None
    world = {"rows": [], "chart": {"periods": []}}
    for mk, cfg in MARKETS.items():
        try:
            ut = jp_flat if cfg["universe"] == "jp" else US_UNIVERSE if cfg["universe"] == "us" else None
            markets[mk] = build_market(prices, cfg, ut)
        except Exception as e:
            print(f"市場分析（{cfg['label']}）の計算に失敗: {e!r}")
            markets[mk] = {"available": False, "label": cfg["label"], "error": f"計算エラー: {e}"}
    try:
        # 期間切替チャート用に、指数だけ11年分を別途取得（個別銘柄は2年分のまま）
        long_prices, _ = data.load_prices_ex(sorted(world_tickers), period="11y")
        world = build_world(prices, long_prices, WORLD, {mk: c["indices"][0][1] for mk, c in MARKETS.items()})
    except Exception as e:
        print(f"世界指数の計算に失敗: {e!r}")
    try:
        if universe and TOPIX_PROXY in prices:
            sd = data.build_sector_data(REGIONS["日本（TOPIX-17業種）"], prices, volume_mode)
            sector_payload = build_sector_plus(sd, prices, universe, prices[TOPIX_PROXY]["Close"])
    except Exception as e:
        print(f"セクター拡張の計算に失敗: {e!r}")

    # ---- 個別銘柄（代表銘柄 + watchlist.txt の登録銘柄） ----
    stock_rows, stock_detail, stock_missing = [], {}, []
    try:
        sector_of = {t: s for s, ts in UNIVERSE.items() for t in ts}
        targets = {t: (NAMES.get(t.replace(".T", "")), sector_of.get(t)) for t in jp_flat}
        for t, nm, sec in watch:
            base_name, base_sec = targets.get(t, (None, None))
            targets[t] = (nm or base_name or NAMES.get(t.replace(".T", "")), sec or base_sec or "未分類")
        sec_scores = {r["name"]: r["score"] for r in sector_payload["rows"]} if sector_payload else {}
        mkt_score = markets.get("jp", {}).get("score") if markets.get("jp", {}).get("available") else None
        stock_rows, stock_detail, stock_missing = analyze(
            prices, [(t, nm, sec) for t, (nm, sec) in targets.items()], prices[TOPIX_PROXY]["Close"], sec_scores, mkt_score)
        watch_codes = {t.replace(".T", "") for t, _, _ in watch}
        for r in stock_rows:
            r["inUniverse"] = r["code"] + ".T" in set(jp_flat)
            r["watch"] = r["code"] in watch_codes
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"個別銘柄の分析に失敗: {e!r}")
    stock_missing = sorted(set(stock_missing) | {t for t, _, _ in watch if t in failed})

    # ---- ウォッチ（注目銘柄の記録と追跡）。記録は公開ビルド（Actions か --record-history）のときだけ ----
    watch_payload = None
    try:
        import os
        import watch_history as WH
        hist = WH.load_history()
        if stock_rows and not private and (record_history or os.environ.get("GITHUB_ACTIONS") == "true") \
                and TOPIX_PROXY in prices:
            day = prices[TOPIX_PROXY].dropna(subset=["Close"]).index.max().strftime("%Y-%m-%d")
            if WH.record(hist, day, WH.make_picks(stock_rows), "live"):
                WH.save_history(hist)
                print(f"注目銘柄を記録しました: {day}")
        if TOPIX_PROXY in prices:
            nm = {**{k: v for k, v in NAMES.items()}, **{r["code"]: r["name"] for r in stock_rows}}
            sc = {r["code"]: r["sector"] for r in stock_rows}
            watch_payload = WH.build_watch(hist, prices, prices[TOPIX_PROXY]["Close"], nm, sc, jp_flat)
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"ウォッチの処理に失敗: {e!r}")

    # ---- バックテスト（過去約10年。代表銘柄のみ。登録銘柄・非公開銘柄は検証に含めない） ----
    backtest_payload = None
    try:
        from backtest import build_backtest
        if universe and TOPIX_PROXY in prices:
            bt_tickers = sorted(set(jp_flat) | {TOPIX_PROXY, MARKET_INDEX})
            bt_prices, _ = data.load_prices_ex(bt_tickers, period="11y")
            backtest_payload = build_backtest(
                bt_prices, universe, bt_prices[TOPIX_PROXY]["Close"],
                bt_prices[MARKET_INDEX]["Close"] if MARKET_INDEX in bt_prices else None)
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"バックテストの計算に失敗: {e!r}")

    latest = max((df.index.max() for df in prices.values()), default=None)
    quality = {
        "failed": failed, "stale": stale, "issues": data.ISSUES,
        "latestDate": latest.strftime("%Y-%m-%d") if latest is not None else None,
        "priceBasis": "Yahoo Finance の終値（株式分割・併合は調整済み／配当は未調整）。休場日・欠損は補完せず空欄のまま扱う",
    }
    jst = timezone(timedelta(hours=9))
    return {
        "markets": markets, "marketOrder": list(MARKETS), "world": world,
        "marketDefaults": {k: c["default_on"] for k, c in MARKETS.items()},
        "sectorPlus": sector_payload, "quality": quality, "backtest": backtest_payload, "watch": watch_payload,
        "stocks": stock_rows, "stockMissing": stock_missing, "_stockDetail": stock_detail,
        "types": TYPES, "stockWeights": {"stock": W_STOCK, "entry": W_ENTRY, "risk": W_RISK, "total": W_TOTAL},
        "generated": datetime.now(jst).strftime("%Y-%m-%d %H:%M") + (" 【非公開版・公開禁止】" if private else ""),
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
    ap.add_argument("--private", action="store_true", help="watchlist_private.txt も分析対象に含める（ローカル専用。出力を公開しないこと）")
    ap.add_argument("--record-history", action="store_true",
                    help="注目銘柄を history/picks.json に記録する（GitHub Actions では自動。ローカルでは通常不要）")
    args = ap.parse_args()

    payload = build_payload(args.groups, DEFAULT_WEIGHTS, args.volume, args.private, args.record_history)
    # 個別銘柄のチャート用データは、表示速度のため銘柄ごとの別JSONにして必要なときだけ読み込む
    detail = payload.pop("_stockDetail", {})
    data_dir = Path(args.out).parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for old in data_dir.glob("stock_*.json"):
        old.unlink()
    for code, d in detail.items():
        (data_dir / f"stock_{code}.json").write_text(json.dumps(d, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    html =TEMPLATE.read_text(encoding="utf-8").replace(
        "/*__DATA__*/null", json.dumps(payload, ensure_ascii=False).replace("</", "<\\/"))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"作成しました: {out.resolve()}（{', '.join(g['name'] for g in payload['groups'])}）")


if __name__ == "__main__":
    main()
