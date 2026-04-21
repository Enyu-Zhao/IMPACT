#!/bin/bash
set -e

IMAGE_NAME="impact"
IMAGE_TAG="latest"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

docker build -t ${IMAGE_NAME}:${IMAGE_TAG} -f "${SCRIPT_DIR}/Dockerfile" "${SCRIPT_DIR}"
echo "Built ${IMAGE_NAME}:${IMAGE_TAG}"
