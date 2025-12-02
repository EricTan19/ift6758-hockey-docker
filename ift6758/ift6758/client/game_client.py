import logging
from typing import Callable, List, Dict

import requests
import pandas as pd

from serving_client import ServingClient
from features import build_features

logger = logging.getLogger(__name__)


class GameClient:

    def __init__(
        self,
        serving_client: ServingClient,
    ):

        self.serving_client = serving_client
        self.feature_fn = build_features
        self.seen_event_ids = set()

    def fetch_game_data(self, game_id: str) -> Dict:

        url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"
        logger.info(f"Fetching game data for game_id={game_id} from {url}")

        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def _extract_all_events(self, data: Dict) -> List[Dict]:

        return data.get("plays", [])

    def _get_event_id(self, event: Dict, fallback_idx: int) -> str:

        return str(event.get("eventId", fallback_idx))

    def get_new_events(self, data: Dict) -> List[Dict]:

        all_events = self._extract_all_events(data)
        new_events = []

        for idx, ev in enumerate(all_events):
            ev_id = self._get_event_id(ev, idx)
            if ev_id not in self.seen_event_ids:
                new_events.append(ev)

        logger.info(f"Found {len(new_events)} new events")
        return new_events

    def step(self, game_id: str) -> pd.DataFrame:

        game_data = self.fetch_game_data(game_id)
        new_events = self.get_new_events(game_data)

        if not new_events:
            logger.info("No new events to process.")
            return pd.DataFrame()

        X = self.feature_fn(new_events, game_data)

        preds_df = self.serving_client.predict(X)

        for idx, ev in enumerate(new_events):
            ev_id = self._get_event_id(ev, idx)
            self.seen_event_ids.add(ev_id)

        return preds_df
    
    # Bonus feature
    def get_live_game_ids(self) -> List[str]:
        url = "https://api-web.nhle.com/v1/scoreboard/now"

        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            games = data.get("games", [])
            live_ids = [
                str(g.get("id"))
                for g in games
                if g.get("gameState") == "LIVE"   # Only in-progress games
            ]

            logger.info(f"Live games found: {live_ids}")
            return live_ids

        except Exception as e:
            logger.error(f"Failed to fetch live games: {e}")
            return []
