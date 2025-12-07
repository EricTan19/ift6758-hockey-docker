#!/bin/bash

docker build -t ift6758/serving:latest -f Dockerfile.serving .
docker build -t ift6758/streamlit:latest -f Dockerfile.streamlit .
