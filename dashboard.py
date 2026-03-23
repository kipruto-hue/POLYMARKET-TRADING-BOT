"""Streamlit operator dashboard for the Polymarket bot."""

import time

import httpx
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="Polymarket Bot", layout="wide")
st.title("Polymarket Bot Dashboard")


def api_get(path: str):
    try:
        r = httpx.get(f"{API_BASE}{path}", timeout=5)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def api_post(path: str):
    try:
        r = httpx.post(f"{API_BASE}{path}", timeout=5)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


# --- Controls ---
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("KILL SWITCH", type="primary"):
        st.json(api_post("/kill"))
with col2:
    if st.button("Resume Trading"):
        st.json(api_post("/resume"))
with col3:
    if st.button("Reset Daily PnL"):
        st.json(api_post("/reset-daily"))

st.divider()

# --- Health & Metrics ---
health = api_get("/health")
if "error" not in health:
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    metrics = health.get("metrics", {})
    mcol1.metric("Uptime", f"{metrics.get('uptime_s', 0) // 60}m")
    mcol2.metric("Cycles", metrics.get("cycles", 0))
    mcol3.metric("Trades Placed", metrics.get("trades_placed", 0))
    mcol4.metric("Errors", metrics.get("errors", 0))

    dry = health.get("dry_run", True)
    st.info(f"Mode: {'DRY RUN' if dry else 'LIVE TRADING'}")
else:
    st.error(f"Bot unreachable: {health.get('error')}")

st.divider()

# --- Positions ---
st.subheader("Open Positions")
positions = api_get("/positions")
if isinstance(positions, dict) and "error" not in positions:
    if positions:
        import pandas as pd
        df = pd.DataFrame.from_dict(positions, orient="index")
        df.index.name = "token_id"
        st.dataframe(df, use_container_width=True)
    else:
        st.write("No open positions.")
else:
    st.write("Could not load positions.")

st.divider()

# --- Active Markets ---
st.subheader("Active Markets")
markets = api_get("/markets")
if isinstance(markets, list) and markets:
    import pandas as pd
    st.dataframe(pd.DataFrame(markets), use_container_width=True)
else:
    st.write("No markets loaded yet.")

# --- Auto-refresh ---
st.caption("Auto-refreshes every 30 seconds.")
time.sleep(0.1)
st.rerun() if st.button("Refresh Now") else None
