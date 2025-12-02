import json
import requests
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class GameClient:
    def __init__(self):
        """
        Initialize anything required for your game client.
        You may want to store:
        - last processed event index
        - cached metadata
        - base NHL API URL
        """
        logger.info("Initializing GameClient")

        # TODO: store anything your client needs (API URL, state, etc.)
        pass

    def ping_game(self, game_id: str, last_idx: int, features: list):
        """
        Ping a live NHL game and return ONLY new events.

        MUST return (as per assignment Part 4):
            X_new            : DataFrame of NEW shot features ONLY
            new_last_idx     : updated last processed event index
            meta             : dict containing game metadata
            events_df_new    : df of raw NEW events (before model prediction)

        Args:
            game_id (str): NHL game ID, e.g., '2021020329'
            last_idx (int): index of last processed event
            features (list): list of model feature names
        
        Returns:
            tuple: (X_new, new_last_idx, meta, events_df_new)
        """

        raise NotImplementedError("TODO: implement ping_game()")

    def _fetch_game_data(self, game_id: str) -> dict:
        """
        Fetch the full game JSON from the NHL API.
        You may call this inside ping_game().

        Args:
            game_id (str): NHL game ID
        
        Returns:
            dict: Raw NHL API JSON
        """

        raise NotImplementedError("TODO: implement _fetch_game_data()")

    def _extract_metadata(self, game_json: dict) -> dict:
        """
        Extract useful metadata:
        - home team name
        - away team name
        - current period
        - time remaining
        - scores
        
        Args:
            game_json (dict): Raw NHL API feed
        
        Returns:
            dict: metadata dictionary
        """

        raise NotImplementedError("TODO: implement _extract_metadata()")

    def _extract_new_events(self, game_json: dict, last_idx: int) -> pd.DataFrame:
        """
        Extract ONLY newly occurred events since last_idx.

        Args:
            game_json (dict): Raw NHL API feed
            last_idx (int): Last processed event index
        
        Returns:
            pd.DataFrame: DataFrame of newly seen event dicts
        """

        raise NotImplementedError("TODO: implement _extract_new_events()")

    def _plays_to_features(self, events_df: pd.DataFrame, features: list) -> pd.DataFrame:
        """
        Convert raw events into the feature DataFrame used by the model.

        Args:
            events_df (DataFrame): Raw event data
            features (list): List of expected model features
        
        Returns:
            DataFrame: Feature dataframe aligned with model's expected columns
        """

        raise NotImplementedError("TODO: implement _plays_to_features()")
