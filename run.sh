#!/bin/bash
NETWORK_NAME=ift6758-net
docker network create "${NETWORK_NAME}"
docker run -d --name serving --network "${NETWORK_NAME}" -p 5050:5000 -e WANDB_API_KEY="${WANDB_API_KEY}" ift6758/serving:latest
docker run -d --name streamlit --network "${NETWORK_NAME}" -p 8501:8501 -e WANDB_API_KEY="${WANDB_API_KEY}" ift6758/streamlit:latest