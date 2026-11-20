"""過去の「今日注目する銘柄」を、その日までのデータだけで再現して history/picks.json に入れる（初回に1回だけ使う）。

使い方:
    python watch_replay.py --days 10        # 直近10営業日分を再現（数分かかります）
    python watch_replay.py --days 10 --check  # 最新日を再現し、いまの画面の選出と一致するか確認するだけ

注意:
- 再現分は source="replay"。後から過去のデータで計算し直したもので、その日に実際に出ていた画面とは
  （株価の修正・構成銘柄の違いなどで）少し違う可能性があります。実記録（live）は上書きしません。
- 計算には、その日より後のデータは使いません（株価を切り詰めて、市場・セクター・個別銘柄を作り直します）。
- 保有銘柄などの非公開銘柄は含めません（watchlist.txt のみ）。
"""
import argparse
from pathlib import Path

import pandas as pd

import data
import watch_history as WH
from make_html import MARKET_INDEX, MARKETS, NAMES, REGIONS, TOPIX_PROXY, UNIVERSE, US_UNIVERSE, WORLD
from market import build_market
from sector_plus import build_sector_plus
from stocks import analyze, load_watchlist


def picks_at(prices: dict, day: pd.Timestamp, jp_flat: list, targets: list, volume_mode: str) -> dict:
    cut = {t: df.loc[:day] for t, df in prices.items()}
    cut, _ = data.drop_stale(cut, MARKET_INDEX)
    mk = build_market(cut, MARKETS["jp"], jp_flat)
    sd = data.build_sector_data(REGIONS["日本（TOPIX-17業種）"], cut, volume_mode)
    sp = build_sector_plus(sd, cut, UNIVERSE, cut[TOPIX_PROXY]["Close"])
    sec = {r["name"]: r["score"] for r in sp["rows"]}
    rows, _, _ = analyze(cut, targets, cut[TOPIX_PROXY]["Close"], sec, mk.get("score") if mk.get("available") else None,
                         detail_days=5)
    return WH.make_picks(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=10, help="何営業日さかのぼるか（最新日を含む）")
    ap.add_argument("--check", action="store_true", help="最新日だけ再現して表示（履歴ファイルは変更しない）")
    args = ap.parse_args()

    jp_flat = sorted({t for ts in UNIVERSE.values() for t in ts})
    world = {t for _, t, _ in WORLD}
    idx = world | {t for m in MARKETS.values() for _, t in m["indices"]} | {m["vi"][1] for m in MARKETS.values() if m.get("vi")}
    watch = load_watchlist(Path(__file__).with_name("watchlist.txt"))
    tickers = set(data.tickers_for(REGIONS["日本（TOPIX-17業種）"])) | {MARKET_INDEX, TOPIX_PROXY} | idx | set(jp_flat) | set(US_UNIVERSE) \
        | {t for t, _, _ in watch}
    prices, failed = data.load_prices_ex(sorted(tickers))
    prices, _ = data.drop_stale(prices, MARKET_INDEX)

    sector_of = {t: s for s, ts in UNIVERSE.items() for t in ts}
    tg = {t: (NAMES.get(t.replace(".T", "")), sector_of.get(t)) for t in jp_flat}
    for t, nm, sec in watch:
        b = tg.get(t, (None, None))
        tg[t] = (nm or b[0] or NAMES.get(t.replace(".T", "")), sec or b[1] or "未分類")
    targets = [(t, nm, sec) for t, (nm, sec) in tg.items()]

    cal = prices[TOPIX_PROXY]["Close"].dropna().index
    days = list(cal[-args.days:])
    hist = WH.load_history()
    if args.check:
        days = days[-1:]
    for d in days:
        p = picks_at(prices, d, jp_flat, targets, "構成銘柄の売買代金合計")
        key = d.strftime("%Y-%m-%d")
        if args.check:
            for k, lst in p.items():
                print(key, WH.LISTS[k], [c for c, _ in lst])
            continue
        ok = WH.record(hist, key, p, "replay")
        print(key, "記録" if ok else "実記録があるためスキップ", {k: [c for c, _ in v] for k, v in p.items()})
    if not args.check:
        WH.save_history(hist)
        print(f"保存しました: {WH.HIST_PATH}（{len(hist['days'])}日分）")


if __name__ == "__main__":
    main()
