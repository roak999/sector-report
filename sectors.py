"""対象セクターの定義。

price_ticker : 価格（ローソク足・MACDなど）に使う業種別ETF（NEXT FUNDS TOPIX-17）
members      : 出来高（売買代金）の代用に使う主要構成銘柄。自由に入れ替えてOK。
               業種別ETFは売買が少ないため、構成銘柄の売買代金合計を業種の出来高とみなす。
"""

JAPAN_17 = {
    "食品":               {"price_ticker": "1617.T", "members": ["2914.T", "2802.T", "2502.T"]},
    "エネルギー資源":     {"price_ticker": "1618.T", "members": ["5020.T", "1605.T", "5019.T"]},
    "建設・資材":         {"price_ticker": "1619.T", "members": ["1925.T", "1928.T", "1812.T"]},
    "素材・化学":         {"price_ticker": "1620.T", "members": ["4063.T", "4452.T", "3407.T"]},
    "医薬品":             {"price_ticker": "1621.T", "members": ["4502.T", "4568.T", "4519.T"]},
    "自動車・輸送機":     {"price_ticker": "1622.T", "members": ["7203.T", "7267.T", "6902.T"]},
    "鉄鋼・非鉄":         {"price_ticker": "1623.T", "members": ["5401.T", "5802.T", "5713.T"]},
    "機械":               {"price_ticker": "1624.T", "members": ["6367.T", "6301.T", "6273.T"]},
    "電機・精密":         {"price_ticker": "1625.T", "members": ["6758.T", "6501.T", "8035.T"]},
    "情報通信・サービス": {"price_ticker": "1626.T", "members": ["9984.T", "9432.T", "6098.T"]},
    "電力・ガス":         {"price_ticker": "1627.T", "members": ["9501.T", "9503.T", "9531.T"]},
    "運輸・物流":         {"price_ticker": "1628.T", "members": ["9022.T", "9020.T", "9101.T"]},
    "商社・卸売":         {"price_ticker": "1629.T", "members": ["8058.T", "8001.T", "8031.T"]},
    "小売":               {"price_ticker": "1630.T", "members": ["9983.T", "3382.T", "8267.T"]},
    "銀行":               {"price_ticker": "1631.T", "members": ["8306.T", "8316.T", "8411.T"]},
    "金融（除く銀行）":   {"price_ticker": "1632.T", "members": ["8766.T", "8591.T", "8604.T"]},
    "不動産":             {"price_ticker": "1633.T", "members": ["8801.T", "8802.T", "8830.T"]},
}

# 主要指数。指数には「構成銘柄の売買代金」が無いので、Yahooの出来高（株数）をそのまま使う。
# 出来高が配信されない指数（DAX・SENSEXなど）は、出来高の採点を自動的に外して総合スコアを出す。
INDICES = {
    "日経平均":          {"price_ticker": "^N225",  "members": [], "volume": "shares"},
    "TOPIX（ETF 1306）": {"price_ticker": "1306.T", "members": [], "volume": "turnover"},
    "NYダウ":            {"price_ticker": "^DJI",   "members": [], "volume": "shares"},
    "ナスダック総合":    {"price_ticker": "^IXIC",  "members": [], "volume": "shares"},
    "S&P500":            {"price_ticker": "^GSPC",  "members": [], "volume": "shares"},
    "DAX（ドイツ）":     {"price_ticker": "^GDAXI", "members": [], "volume": "shares"},
    "SENSEX（インド）":  {"price_ticker": "^BSESN", "members": [], "volume": "shares"},
}

# 画面での並び順。sort=False のグループはスコア順に並べ替えず、上の定義順のまま表示する
GROUP_OPTIONS = {
    "主要指数": {"sort": False},
}

REGIONS = {
    "日本（TOPIX-17業種）": JAPAN_17,
    "主要指数": INDICES,
    # 次のステップで追加:
    # "米国（SPDRセクター）": US_11,
    # "新興国（国別ETF）": EMERGING,
}
