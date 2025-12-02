import json
import requests
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class ServingClient:
    def __init__(self, ip: str = "127.0.0.1", port: int = 5000, features=None):
        self.base_url = f"http://{ip}:{port}"
        logger.info(f"Initializing client; base URL: {self.base_url}")

        if features is None:
            features = ["distance", "angle_from_net"]
        self.features = features

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.features is not None:
            missing = [f for f in self.features if f not in X.columns]
            if missing:
                raise ValueError(f"Missing required features in X: {missing}")
            X_payload = X[self.features].copy()
        else:
            X_payload = X.copy()

        url = f"{self.base_url}/predict"
        payload = X_payload.to_dict(orient="records")

        try:
            resp = requests.post(url, json=payload)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Error while calling prediction service: {e}")
            raise

        data = resp.json()  
        preds = data["predictions"]

        X_with_pred = X.copy()
        X_with_pred["goal_prob"] = preds
        return X_with_pred

    def logs(self) -> dict:
        url = f"{self.base_url}/logs"
        try:
            resp = requests.get(url)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Error while fetching logs: {e}")
            raise

        content_type = resp.headers.get("Content-Type", "")
        if "application/json" in content_type:
            try:
                return resp.json()
            except json.JSONDecodeError:
                return {"logs": resp.text}
        return {"logs": resp.text}

    def download_registry_model(self, workspace: str, model: str, version: str) -> dict:
        url = f"{self.base_url}/download_registry_model"
        payload = {
            "workspace": workspace,
            "model": model,
            "version": version,
        }

        try:
            resp = requests.post(url, json=payload)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.error(f"Error while downloading registry model: {e}")
            raise

        try:
            return resp.json()
        except json.JSONDecodeError:
            return {"raw_response": resp.text}
