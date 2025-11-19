"""
If you are in the same directory as this file (app.py), you can run run the app using gunicorn:
    
    $ gunicorn --bind 0.0.0.0:<PORT> app:app

gunicorn can be installed via:

    $ pip install gunicorn

"""
import os
from pathlib import Path
import logging
from flask import Flask, jsonify, request, abort

app = Flask(__name__)

import sklearn
import pandas as pd
import joblib
import wandb


import ift6758


LOG_FILE = os.environ.get("FLASK_LOG", "flask.log")





@app.before_first_request
def before_first_request():
    """
    Hook to handle any initialization before the first request (e.g. load model,
    setup logging handler, etc.)
    """
    # TODO: setup basic logging configuration
    # Setup logging
    logging.basicConfig(filename=LOG_FILE, level=logging.INFO)

    default_name = "distance"
    default_version = "latest"
    default_artifact = "logreg_distance_model"
    local_path = Path(f"{default_artifact}_{default_version}.pkl")

    # TODO: any other initialization before the first request (e.g. load default model)
    # Load default model: distance model
    
     # If local model exists → load it
    if local_path.exists():
        try:
            app.model = joblib.load(local_path)
            app.current_model_name = default_name
            app.current_model_version = default_version
            app.logger.info("Loaded default model from local file.")
            return
        except Exception as e:
            app.logger.error(f"Local default model exists but failed to load: {e}")

    # Otherwise download it using the SAME logic as /download_registry_model
    app.logger.warning("No default model found. Attempting automatic download...")

    try:
        # Initialize wandb client
        run = wandb.init(
            project="ift6758-shot-prediction",
            job_type="download-default",
            reinit=True
        )

        artifact = run.use_artifact(f"{default_artifact}:{default_version}", type="model")
        artifact_dir = artifact.download()

        # Find .pkl inside artifact directory
        pkl_files = list(Path(artifact_dir).rglob("*.pkl"))
        if not pkl_files:
            raise FileNotFoundError("Artifact contains no .pkl file")

        # Save locally for future runs
        joblib.dump(joblib.load(pkl_files[0]), local_path)

        # Load into app
        app.model = joblib.load(local_path)
        app.current_model_name = default_name
        app.current_model_version = default_version

        app.logger.info(f"Successfully downloaded and loaded default model {default_artifact}:{default_version}")

    except Exception as e:
        app.logger.error(f"Failed to automatically download default model: {e}")




@app.route("/logs", methods=["GET"])
def logs():
    """Reads data from the log file and returns them as the response"""
    
    # TODO: read the log file specified and return the data
    try:
        if not os.path.exists(LOG_FILE):
            response = {"logs": []}
        else:
            with open(LOG_FILE, "r") as f:
                lines = f.readlines()
            response = {"logs": lines}
    except Exception as e:
        app.logger.error(f"Error reading log file: {e}")
        abort(500, description="Could not read logs")

    return jsonify(response)  # response must be json serializable!


@app.route("/download_registry_model", methods=["POST"])
def download_registry_model():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/download_registry_model

    The comet API key should be retrieved from the ${COMET_API_KEY} environment variable.

    Recommend (but not required) json with the schema:

        {
            workspace: (required),
            model: (required),
            version: (required),
            ... (other fields if needed) ...
        }
    
    """
    # Get POST json data
    json = request.get_json()
    app.logger.info(json)

    workspace = json.get("workspace")
    model_name = json.get("model")
    version = json.get("version", "latest")

    if workspace is None or model_name is None:
        abort(400, description="workspace and model fields are required")

    # Map Flask model name → WandB artifact name
    artifact_map = {
        "distance": "logreg_distance_model",
        "angle_from_net": "logreg_angle_model",
        "distance_angle": "logreg_distance_angle_model",
    }

    if model_name not in artifact_map:
        abort(400, description=f"Invalid model name {model_name}")
        
    artifact_name = artifact_map[model_name]

    # TODO: check to see if the model you are querying for is already downloaded
    local_path = Path(f"{artifact_name}_{version}.pkl")
    
    # TODO: if yes, load that model and write to the log about the model change.  
    # eg: app.logger.info(<LOG STRING>)

    if local_path.exists():
        try:
            app.model = joblib.load(local_path)
            app.current_model_name = model_name
            app.current_model_version = version
            app.logger.info(f"Model already exists locally. Loaded {local_path}")
            return jsonify({"status": "success", "model": model_name, "version": version})
        except Exception as e:
            app.logger.error(f"Local model exists but could not be loaded: {e}")


    
    # TODO: if no, try downloading the model: if it succeeds, load that model and write to the log
    # about the model change. If it fails, write to the log about the failure and keep the 
    # currently loaded model
    try:

        run = wandb.init(project="ift6758-shot-prediction", job_type="download", reinit=True)

        artifact = run.use_artifact(f"{artifact_name}:{version}", type="model")
        artifact_dir = artifact.download()

        # find pkl file inside artifact directory
        pkl_files = list(Path(artifact_dir).rglob("*.pkl"))
        if not pkl_files:
            raise FileNotFoundError("Artifact contains no .pkl file")

        joblib.dump(joblib.load(pkl_files[0]), local_path)

        # load newly downloaded model
        app.model = joblib.load(local_path)
        app.current_model_name = model_name
        app.current_model_version = version

        app.logger.info(f"Downloaded and loaded model {artifact_name}:{version}")
        return jsonify({"status": "success", "model": model_name, "version": version})

    except Exception as e:
        app.logger.error(f"Failed to download model: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

    # Tip: you can implement a "CometMLClient" similar to your App client to abstract all of this
    # logic and querying of the CometML servers away to keep it clean here

    raise NotImplementedError("TODO: implement this endpoint")

    response = None

    app.logger.info(response)
    return jsonify(response)  # response must be json serializable!


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handles POST requests made to http://IP_ADDRESS:PORT/predict

    Returns predictions
    """
    # Get POST json data
    json = request.get_json()
    app.logger.info(json)

    if not hasattr(app, "model"):
        abort(503, description="No model loaded. Call /download_registry_model first.")


    # TODO:
    try:
        # Convert JSON → DataFrame
        X = pd.DataFrame.from_dict(json)

        # Determine expected feature list
        feature_map = {
            "distance": ["distance"],
            "angle_from_net": ["angle_from_net"],
            "distance_angle": ["distance", "angle_from_net"],
        }

        required = feature_map[app.current_model_name]

        # Validate missing columns
        missing = [c for c in required if c not in X.columns]
        if missing:
            abort(400, description=f"Missing required features: {missing}")

        X = X[required]

        # Predict probability (logistic regression)
        preds = app.model.predict_proba(X)[:, 1].tolist()

        response = {
            "model": app.current_model_name,
            "version": app.current_model_version,
            "predictions": preds
        }

        return jsonify(response)

    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        abort(500, description=str(e))