import streamlit as st
import pandas as pd
import numpy as np

from serving_client import ServingClient
from game_client import GameClient

def init_state():
    if "serving_client" not in st.session_state:
        st.session_state.serving_client = ServingClient(
            ip="serving",
            port=5000,
        )

    if "game_client" not in st.session_state:
        st.session_state.game_client = GameClient(
            serving_client=st.session_state.serving_client
        )

    if "current_game_id" not in st.session_state:
        st.session_state.current_game_id = None
    if "events_df" not in st.session_state:
        st.session_state.events_df = pd.DataFrame()
    if "game_meta" not in st.session_state:
        st.session_state.game_meta = {}
    if "last_ping_text" not in st.session_state:
        st.session_state.last_ping_text = "(never)"


def reset_game_state():
    st.session_state.events_df = pd.DataFrame()
    st.session_state.game_meta = {}
    st.session_state.last_ping_text = "(never)"

    if "game_client" in st.session_state:
        st.session_state.game_client.seen_event_ids.clear()


def compute_xg_and_score(df: pd.DataFrame):
    """
    Compute xG and actual score for home/away.

    Assumes df has:
      - 'is_home'   : True for home-team events, False for away-team
      - 'is_goal'   : 1 if goal, 0 otherwise
      - 'goal_prob' : model's predicted probability from ServingClient
    """
    if df.empty:
        return 0.0, 0.0, 0, 0

    home_df = df[df["is_home"] == True]
    away_df = df[df["is_home"] == False]

    xg_home = float(home_df["goal_prob"].sum())
    xg_away = float(away_df["goal_prob"].sum())
    score_home = int(home_df["is_goal"].sum())
    score_away = int(away_df["is_goal"].sum())

    return xg_home, xg_away, score_home, score_away


init_state()

st.title("NHL Live Game Predictor")

# -------------------------------------------------------------------------
# LIVE GAMES BANNER
# -------------------------------------------------------------------------
live_games = st.session_state.game_client.get_live_game_ids()

if live_games:
    games_str = "   •   ".join(live_games)

    st.markdown(
        f"""
        <div style="
            background-color:#161616;
            padding:12px 18px;
            border-radius:10px;
            margin-bottom:20px;
            border:1px solid #333;
        ">
            <marquee behavior="scroll" direction="left" scrollamount="4"
                     style="color:#0affc1; font-size:18px; font-weight:600;">
                🔴 LIVE GAMES: {games_str}
            </marquee>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        f"""
        <div style="
            background-color:#111;
            padding:10px;
            border-radius:8px;
            margin-bottom:15px;
            border:1px solid #333;
            color:#888;
            text-align:center;
        ">
            No live games right now.
        </div>
        """,
        unsafe_allow_html=True
    )

# -------------------------------------------------------------------
# SIDEBAR — Model Registry Inputs
# -------------------------------------------------------------------
with st.sidebar:
    st.subheader("Model Registry")

    model_options = ["distance", "angle_from_net", "distance_angle"]
    version_options = ["latest"]

    model_name = st.selectbox("Model", model_options, index=1)
    version = st.selectbox("Version", version_options, index=0)

    if st.button("Get model"):
        try:
            resp = st.session_state.serving_client.download_registry_model(
                workspace="default",
                model=model_name,
                version=version,
            )
            st.success(f"Model loaded: {resp}")
            # Reset seen events since we need to recompute predictions with the new model
            st.session_state.game_client.seen_event_ids.clear()
            st.session_state.events_df = pd.DataFrame()
            st.session_state.last_idx = None
            st.session_state.last_ping_text = "(model changed)"
            st.session_state.game_meta = {}
        except Exception as e:
            st.error(f"Error downloading model: {e}")

# -------------------------------------------------------------------
# CONTAINER 1 — Game ID Input + Ping
# -------------------------------------------------------------------
with st.container():
    game_id = st.text_input("Game ID", value="")

    ping_clicked = st.button("Ping game")

    if st.session_state.current_game_id != game_id:
        st.session_state.current_game_id = game_id
        reset_game_state()

    if ping_clicked:
        if not game_id:
            st.warning("Please enter a valid Game ID.")
        else:
            try:
                new_df = st.session_state.game_client.step(game_id)

                if not new_df.empty:
                    st.session_state.events_df = (
                        pd.concat([st.session_state.events_df, new_df])
                        .drop_duplicates(subset=["event_id"])
                        .reset_index(drop=True)
                    )

                    last_row = new_df.iloc[-1]
                    st.session_state.game_meta = {
                        "home_team": last_row.get("home_team", "Home team"),
                        "away_team": last_row.get("away_team", "Away team"),
                        "period": last_row.get("period", "?"),
                        "time_remaining": last_row.get("time_remaining", "??:??"),
                    }

                st.session_state.last_ping_text = "success"

            except Exception as e:
                st.session_state.last_ping_text = f"error: {e}"
                st.error(f"Error while pinging game: {e}")
    
    st.caption(f"Last ping: {st.session_state.last_ping_text}")

# -------------------------------------------------------------------
# CONTAINER 2 — Game Info + Predictions Summary
# -------------------------------------------------------------------
with st.container():
    meta = st.session_state.game_meta or {}
    home_team = meta.get("home_team", "Home team")
    away_team = meta.get("away_team", "Away team")
    period = meta.get("period", "?")
    time_remaining = meta.get("time_remaining", "??:??")

    xg_home, xg_away, score_home, score_away = compute_xg_and_score(
        st.session_state.events_df
    )

    st.markdown(
        f"### Game {st.session_state.current_game_id or 'N/A'}: "
        f"{home_team} vs {away_team}"
    )

    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        st.markdown(f"**Home Team:** {home_team}")
    with row1_col2:
        st.markdown(f"**Away Team:** {away_team}")

    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        st.markdown(f"**Period:** {period}")
    with row2_col2:
        st.markdown(f"**Time Left:** {time_remaining}")

    col_home, col_away = st.columns(2)
    with col_home:
        st.markdown(f"#### {home_team} xG (actual)")
        st.markdown(f"### {xg_home:.1f} ({score_home})")
        delta_home = score_home - xg_home
        st.metric(
            label="Difference",
            value="",
            delta=f"{delta_home:+.1f}",
            label_visibility="hidden",
        )

    with col_away:
        st.markdown(f"#### {away_team} xG (actual)")
        st.markdown(f"### {xg_away:.1f} ({score_away})")
        delta_away = score_away - xg_away
        st.metric(
            label="Difference",
            value="",
            delta=f"{delta_away:+.1f}",
            label_visibility="hidden",
        )

# -------------------------------------------------------------------
# CONTAINER 3 — Table of Events + Predictions
# -------------------------------------------------------------------
with st.container():
    st.markdown("### Events and Model Predictions")

    if st.session_state.events_df.empty:
        st.info("No shot events yet. Click **Ping game** to fetch new events.")
    else:
        st.dataframe(st.session_state.events_df, width='stretch')
