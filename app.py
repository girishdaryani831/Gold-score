import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Gold Score", page_icon="🪙", layout="centered")


@st.cache_data(ttl=300)  # reuse data for 5 minutes, then fetch fresh prices
def get_ohlcv(ticker):
    df = yf.download(ticker, period="2y", interval="1d",
                     progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna(subset=["Close"])


@st.cache_data(ttl=6 * 3600)  # CFTC publishes weekly, so 6 hours is plenty
def get_cot():
    """COMEX gold, managed money (hedge funds), CFTC disaggregated report."""
    url = "https://publicreporting.cftc.gov/resource/72hh-3qpy.json"
    params = {
        "$where": "cftc_contract_market_code='088691'",
        "$order": "report_date_as_yyyy_mm_dd DESC",
        "$limit": 160,
        "$select": ("report_date_as_yyyy_mm_dd,"
                    "m_money_positions_long_all,m_money_positions_short_all"),
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    df["date"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"])
    df["net"] = (pd.to_numeric(df["m_money_positions_long_all"])
                 - pd.to_numeric(df["m_money_positions_short_all"]))
    return df.sort_values("date").reset_index(drop=True)


def soft(x):
    """Soft scaling: moderate moves score moderately, only extreme moves
    approach -1 or +1."""
    return float(np.tanh(x))


def rsi(s, p=14):
    d = s.diff()
    up = d.clip(lower=0).rolling(p).mean()
    dn = (-d.clip(upper=0)).rolling(p).mean()
    return 100 - 100 / (1 + up / dn)


st.title("🪙 Gold Score")
st.caption("Research tool only, not financial advice.")

timeframe = st.radio("Timeframe", ["Short", "Medium", "Long"],
                     index=1, horizontal=True)
WEIGHTS = {
    "Short": {"Technicals": 0.35, "Volume & flows": 0.30,
              "Macro proxies": 0.15, "Sentiment & positioning": 0.20},
    "Medium": {"Technicals": 0.30, "Volume & flows": 0.20,
               "Macro proxies": 0.30, "Sentiment & positioning": 0.20},
    "Long": {"Technicals": 0.15, "Volume & flows": 0.10,
             "Macro proxies": 0.55, "Sentiment & positioning": 0.20},
}[timeframe]

if st.button("Refresh data"):
    st.cache_data.clear()

try:
    gold_df = get_ohlcv("GC=F")            # gold futures
    usd = get_ohlcv("DX-Y.NYB")["Close"]   # US dollar index
    y10 = get_ohlcv("^TNX")["Close"]       # US 10-year yield (proxy)
except Exception as e:
    st.error(f"Could not load market data: {e}")
    st.stop()

gold = gold_df["Close"]
if len(gold) < 220 or len(usd) < 70 or len(y10) < 70:
    st.error("Not enough data came back from Yahoo Finance. Try Refresh in a minute.")
    st.stop()

# ---------- Pillar 1: technicals ----------
sma50 = gold.rolling(50).mean().iloc[-1]
sma200 = gold.rolling(200).mean().iloc[-1]
last = gold.iloc[-1]
rsi_now = rsi(gold).iloc[-1]
mom60 = gold.pct_change(60).iloc[-1]

tech = {
    "Price vs 200-day average": soft((last / sma200 - 1) / 0.10),
    "50-day vs 200-day average": soft((sma50 / sma200 - 1) / 0.06),
    # RSI above 70 = overbought (bearish), below 30 = oversold (bullish)
    "RSI (14)": soft((50 - rsi_now) / 15) if (rsi_now > 70 or rsi_now < 30)
                else soft((rsi_now - 50) / 25),
    "60-day momentum": soft(mom60 / 0.12),
}

# ---------- Pillar 2: volume & flows ----------
vol = {}
try:
    d = gold_df[gold_df["Volume"] > 0].dropna(subset=["Volume"])
    if len(d) >= 120:
        ret20 = d["Close"].pct_change(20).iloc[-1]
        rel_vol = (d["Volume"].rolling(20).mean().iloc[-1]
                   / d["Volume"].rolling(100).mean().iloc[-1])
        vol["20-day move, volume-weighted"] = soft(
            ret20 / 0.04 * float(np.clip(rel_vol, 0.6, 1.5)))
        direction = np.sign(d["Close"].diff())
        flow = ((direction * d["Volume"]).tail(20).sum()
                / d["Volume"].tail(20).sum())
        vol["On-balance volume flow (20d)"] = soft(flow / 0.25)
except Exception:
    pass

try:
    g = get_ohlcv("GLD")  # largest gold ETF: money-flow proxy for ETF demand
    g = g[g["Volume"] > 0]
    if len(g) >= 40:
        hl = (g["High"] - g["Low"]).replace(0, np.nan)
        mfm = ((g["Close"] - g["Low"]) - (g["High"] - g["Close"])) / hl
        cmf = ((mfm * g["Volume"]).tail(20).sum()
               / g["Volume"].tail(20).sum())
        vol["GLD ETF money flow (Chaikin, 20d)"] = soft(cmf / 0.15)
except Exception:
    pass

# ---------- Pillar 3: macro proxies (gold moves inversely to both) ----------
macro = {
    "US dollar trend (60d, inverse)": soft(-usd.pct_change(60).iloc[-1] / 0.05),
    "10-year yield trend (60d, inverse)": soft(-y10.pct_change(60).iloc[-1] / 0.15),
}

# ---------- Pillar 4: sentiment & positioning ----------
sent = {}
cot_date = None
try:
    cot = get_cot()
    if len(cot) >= 60:
        cot_date = cot["date"].iloc[-1].date()
        net = cot["net"]
        # 1) are hedge funds adding or cutting their net long? (trend-following)
        chg4 = net.iloc[-1] - net.iloc[-5]
        std4 = net.diff(4).std()
        sent["Hedge-fund net position, 4-week change"] = soft(chg4 / (1.5 * std4))
        # 2) crowding: only extreme positioning counts, and it counts against
        #    the crowd (very crowded long = bearish, very washed out = bullish)
        p = float(net.tail(156).rank(pct=True).iloc[-1])
        if p > 0.8:
            crowd = -(p - 0.8) / 0.2
        elif p < 0.2:
            crowd = (0.2 - p) / 0.2
        else:
            crowd = 0.0
        sent["Positioning crowding (3-year percentile)"] = float(crowd)
except Exception:
    pass

try:
    vix = get_ohlcv("^VIX")["Close"]
    if len(vix) >= 200:
        # fear above its yearly average is mildly supportive for gold
        sent["VIX fear gauge vs 1-year average"] = soft(
            (vix.iloc[-1] / vix.rolling(250, min_periods=200).mean().iloc[-1] - 1)
            / 0.30)
except Exception:
    pass

pillars = {"Technicals": tech, "Volume & flows": vol,
           "Macro proxies": macro, "Sentiment & positioning": sent}
pillars = {n: v for n, v in pillars.items() if v}   # drop a pillar with no data

w = {n: WEIGHTS[n] for n in pillars}
total_w = sum(w.values())
w = {n: x / total_w for n, x in w.items()}          # re-normalise to 100%

pillar_scores = {n: float(np.mean(list(v.values()))) for n, v in pillars.items()}
overall = sum(w[n] * pillar_scores[n] for n in pillars)
bullish_pct = 50 * (overall + 1)

all_scores = [s for v in pillars.values() for s in v.values()]
agree = float(np.mean([np.sign(s) == np.sign(overall) for s in all_scores]))
confidence = "High" if agree >= 0.75 else "Medium" if agree >= 0.5 else "Low"

# ---------- Display ----------
c1, c2, c3 = st.columns(3)
c1.metric("Bullish", f"{bullish_pct:.0f}%")
c2.metric("Confidence", confidence)
c3.metric("Gold price", f"{last:,.0f}")
st.progress(int(round(bullish_pct)))
st.caption(f"{agree * 100:.0f}% of indicators agree with the overall direction. "
           f"Price data as of {gold.index[-1].date()}.")

strong = [s for s in pillar_scores.values() if abs(s) > 0.15]
if any(s > 0 for s in strong) and any(s < 0 for s in strong):
    st.warning("Mixed signals: the pillars disagree. Read the reasons below "
               "before drawing conclusions.")

st.subheader("Pillars")
st.bar_chart(pd.Series({n: 50 * (s + 1) for n, s in pillar_scores.items()},
                       name="Bullish %"))
if len(pillars) == 4:
    st.caption("All four pillars are included. Official-statement tone "
               "(Fed, central banks) and real yields are still to come.")
else:
    st.caption(f"This score covers {len(pillars)} of 4 pillars "
               "(a pillar is skipped if its data failed to load).")

st.subheader("Why")
for name, inds in pillars.items():
    with st.expander(f"{name}: {50 * (pillar_scores[name] + 1):.0f}% bullish "
                     f"(weight {w[name] * 100:.0f}%)"):
        for k, v in inds.items():
            lean = "bullish" if v > 0.15 else "bearish" if v < -0.15 else "neutral"
            st.write(f"**{k}**: {v:+.2f} ({lean})")
        if name == "Sentiment & positioning" and cot_date:
            st.caption(f"Positioning data is from the CFTC report dated {cot_date}. "
                       "It is published weekly on Fridays and reflects "
                       "positions as of the prior Tuesday.")

st.subheader("Gold price with 50 and 200-day averages")
chart = pd.DataFrame({
    "Gold": gold,
    "50-day": gold.rolling(50).mean(),
    "200-day": gold.rolling(200).mean(),
}).dropna()
st.line_chart(chart)

with st.expander("How this score is calculated"):
    st.write(
        "Each indicator becomes a score between -1 (bearish) and +1 (bullish) "
        "using soft scaling, so moderate moves score moderately. Scores are "
        "averaged inside each pillar, then combined with the timeframe weights. "
        "Bullish % = 50 x (score + 1). Confidence reflects how many indicators "
        "agree. The ETF money-flow indicator is a proxy from GLD price and "
        "volume, not official ETF inflow data. Positioning uses the CFTC "
        "disaggregated report for COMEX gold (managed money). Weights and "
        "scaling are first guesses and have not been backtested yet."
    )
