import streamlit as st
import pandas as pd
import numpy as np
from serving_client import ServingClient
from game_client import GameClient

def init_state():
    if "serving_client" not in st.session_state:
        st.session_state.serving_client = ServingClient(
            ip="127.0.0.1",
            port=5000,
        )
    if "game_client" not in st.session_state:
        st.session_state.game_client = GameClient()

    if "current_game_id" not in st.session_state:
        st.session_state.current_game_id = None
    if "last_idx" not in st.session_state:
        st.session_state.last_idx = None
    if "events_df" not in st.session_state:
        st.session_state.events_df = pd.DataFrame()
    if "game_meta" not in st.session_state:
        st.session_state.game_meta = {}
    if "last_ping_text" not in st.session_state:
        st.session_state.last_ping_text = "(never)"

init_state()

def reset_game_state():
    st.session_state.last_idx = None
    st.session_state.events_df = pd.DataFrame()
    st.session_state.game_meta = {}
    st.session_state.last_ping_text = "(never)"


def compute_xg_and_score(df: pd.DataFrame):
    """
    Compute xG and actual score for home/away.

    TODO: if your column names differ, change 'is_home', 'is_goal',
          and 'prediction' below.
    """
    if df.empty:
        return 0.0, 0.0, 0, 0

    home_df = df[df["is_home"] == True]
    away_df = df[df["is_home"] == False]

    xg_home = float(home_df["prediction"].sum())
    xg_away = float(away_df["prediction"].sum())
    score_home = int(home_df["is_goal"].sum())
    score_away = int(away_df["is_goal"].sum())

    return xg_home, xg_away, score_home, score_away

st.title("NHL Live Game Predictor")

# -----------------------------------------------------------------------------
# SIDEBAR — Model Registry Inputs
# -----------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Model Registry")

    # Placeholder options — replace with real ones later
    workspace_options = ["Workspace x", "Workspace y", "Workspace z"]
    model_options = ["Model x", "Model y", "Model z"]
    version_options = ["Version 1", "Version 2", "Version 3"]

    workspace = st.selectbox("Workspace", workspace_options, index=0)
    model_name = st.selectbox("Model", model_options, index=1)
    version = st.selectbox("Version", version_options, index=2)

    if st.button("Get model"):
        try:
            resp = st.session_state.serving_client.download_registry_model(
                workspace=workspace,
                model=model_name,
                version=version,
            )
            st.success(f"Model loaded: {resp}")
        except Exception as e:
            st.error(f"Error downloading model: {e}")

# -----------------------------------------------------------------------------
# CONTAINER 1 — Game ID Input
# -----------------------------------------------------------------------------
with st.container():

    game_id = st.text_input("", value="")

    ping_clicked = st.button("Ping game")

    if st.session_state.current_game_id != game_id:
        st.session_state.current_game_id = game_id
        reset_game_state()

    st.caption(f"Last ping: {st.session_state.last_ping_text}")

    if ping_clicked:
        try:
            # Call your GameClient; MUST return only *new* events
            X_new, new_last_idx, meta, events_df_new = (
                st.session_state.game_client.ping_game(
                    game_id=game_id,
                    last_idx=st.session_state.last_idx,
                    features=st.session_state.serving_client.features,
                )
            )

            st.session_state.last_idx = new_last_idx
            st.session_state.game_meta = meta or {}

            if not X_new.empty:
                # Call prediction service on new shots only
                preds_df = st.session_state.serving_client.predict(X_new)

                # Join predictions to new events
                events_df_new = events_df_new.join(preds_df)

                # Append to global events_df and drop duplicates by event id
                # TODO: change 'event_id' to whatever unique event identifier you use
                st.session_state.events_df = (
                    pd.concat([st.session_state.events_df, events_df_new])
                    .drop_duplicates(subset=["event_id"])
                    .reset_index(drop=True)
                )

            st.session_state.last_ping_text = "success"

        except Exception as e:
            st.session_state.last_ping_text = f"error: {e}"
            st.error(f"Error while pinging game: {e}")

# -----------------------------------------------------------------------------
# CONTAINER 2 — Game Info + Predictions Summary
# -----------------------------------------------------------------------------
with st.container():
    meta = st.session_state.game_meta or {}
    home_team = meta.get("home_team", "Home team")
    away_team = meta.get("away_team", "Away team")
    period = meta.get("period", "?")
    time_left = meta.get("time_left", "??:??")

    xg_home, xg_away, score_home, score_away = compute_xg_and_score(
        st.session_state.events_df
    )

    st.markdown(
        f"### Game {st.session_state.current_game_id}: "
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
        st.markdown(f"**Time Left:** {time_left}")

    col_home, col_away = st.columns(2)
    with col_home:
        st.markdown(f"#### {home_team} xG (actual)")
        st.markdown(f"### {xg_home:.1f} ({score_home})")
        st.caption(
            f"Δ = {xg_home:.1f} − {score_home} = {xg_home - score_home:+.1f}"
        )

    with col_away:
        st.markdown(f"#### {away_team} xG (actual)")
        st.markdown(f"### {xg_away:.1f} ({score_away})")
        st.caption(
            f"Δ = {xg_away:.1f} − {score_away} = {xg_away - score_away:+.1f}"
        )

# -----------------------------------------------------------------------------
# CONTAINER 3 — Table of Events + Predictions
# -----------------------------------------------------------------------------
with st.container():
    st.markdown("### Events and Model Predictions")

    if st.session_state.events_df.empty:
        st.info("No shot events yet. Click **Ping game** to fetch new events.")
    else:
        st.dataframe(st.session_state.events_df, use_container_width=True)