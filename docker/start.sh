#!/bin/bash
set -e

IMAGE_NAME="impact"
IMAGE_TAG="latest"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="${SCRIPT_DIR}/.."

if ! docker image inspect ${IMAGE_NAME}:${IMAGE_TAG} &> /dev/null; then
    echo "Image not found. Building first..."
    bash "${SCRIPT_DIR}/build.sh"
fi

mkdir -p ~/.cache/curobo_torch_extensions
mkdir -p ~/.cache/huggingface

xhost + 2>/dev/null || true

docker run \
    --name impact \
    --privileged \
    --rm -it \
    -e NVIDIA_DISABLE_REQUIRE=1 \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    --gpus all \
    --network host \
    --env DISPLAY=$DISPLAY \
    --volume /tmp/.X11-unix:/tmp/.X11-unix \
    --mount type=bind,src="${REPO_ROOT}/curobo_content",dst=/pkgs/curobo/src/curobo/content \
    --mount type=bind,src="${REPO_ROOT}",dst=/pkgs/IMPACT \
    --mount type=bind,src="${HOME}/.cache/curobo_torch_extensions",dst=/root/.cache/torch_extensions \
    --mount type=bind,src="${HOME}/.cache/huggingface",dst=/root/.cache/huggingface \
    impact:latest \
    bash
