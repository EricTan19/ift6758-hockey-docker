from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np

EVENT_MAP = {"goal": "GOAL", "shot-on-goal": "SHOT"}

def parse_strength(code: str) -> str | None:
    """Decode NHL 4-digit situationCode to EVEN/PP/SH."""
    if not code or len(code) != 4:
        return None
    home = int(code[:2]) - 1
    away = int(code[2:]) - 1
    if home == away:
        return "EVEN"
    return "PP" if home > away else "SH"


def get_mapping_tables(payload: dict) -> tuple[dict, dict]:
    """Return player_name and team_name lookup tables from a game JSON."""
    player_name = {
        player["playerId"]: f"{player['firstName']['default']} {player['lastName']['default']}"
        for player in payload.get("rosterSpots", [])
    }

    team_name = {}
    for side in ("homeTeam", "awayTeam"):
        team = payload.get(side, {})
        if team:
            team_name[team["id"]] = {
                "name": f"{team['commonName']['default']}",
                "abbrev": team["abbrev"],
            }
    return player_name, team_name



def build_features(events: List[Dict], payload: Dict) -> pd.DataFrame:
    """
    Feature function used by GameClient.step.

    Args:
        events: list of *new* play dicts (subset of payload["plays"])
        payload: full game JSON

    Returns:
        DataFrame with all model features + meta.
    """
    player_name, team_name = get_mapping_tables(payload)

    home_id = payload.get("homeTeam", {}).get("id")
    away_id = payload.get("awayTeam", {}).get("id")

    plays = []
    for play in events:
        event = play.get("typeDescKey")
        if event not in EVENT_MAP:
            continue

        d = play.get("details", {}) or {}
        shooter_id = d.get("scoringPlayerId") or d.get("shootingPlayerId")
        goalie_id = d.get("goalieInNetId")
        team_id = d.get("eventOwnerTeamId")

        team_side = "HOME" if team_id == home_id else "AWAY"

        x = d.get("xCoord")
        y = d.get("yCoord")

        is_goal = 1 if EVENT_MAP.get(event) == "GOAL" else 0

        empty_net = int(goalie_id is None or goalie_id == 0)

        distance = None
        angle_from_net = None
        if x is not None and y is not None:
            distance = int(
                np.sqrt((89 - abs(x))**2 + (0 - abs(y))**2).round()
            )
            angle_from_net = float(
                np.degrees(np.arctan2(abs(y), 89 - x))
            )

        plays.append(
            {
                "event_id": play.get("eventId"),

                "period": play.get("periodDescriptor", {}).get("number"),
                "period_type": play.get("periodDescriptor", {}).get("periodType"),
                "time_remaining": play.get("timeRemaining"),

                "strength": parse_strength(play.get("situationCode")),
                "event_type": EVENT_MAP.get(event),

                "team_id": team_id,
                "team_name": team_name.get(team_id, {}).get("name"),
                "team_abbr": team_name.get(team_id, {}).get("abbrev"),
                "team_side": team_side,
                "is_home": (team_side == "HOME"),

                "shooter_id": shooter_id,
                "shooter_name": player_name.get(shooter_id),
                "goalie_id": goalie_id,
                "goalie_name": player_name.get(goalie_id),

                "shot_type": d.get("shotType"),

                "home_team": team_name.get(home_id, {}).get("name"),
                "away_team": team_name.get(away_id, {}).get("name"),

                "is_goal": is_goal,
                "empty_net": empty_net,
                "distance": distance,
                "angle_from_net": angle_from_net,
            }
        )

    return pd.DataFrame(plays)
