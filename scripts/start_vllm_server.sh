#!/bin/bash
# Start vLLM OpenAI-compatible server on GH200
#
# This starts the Qwen3 model as an OpenAI-compatible API server
# that can be called from another machine.
#
# Usage:
#   ./start_vllm_server.sh
#
# The server will listen on port 8000 and can be accessed at:
#   http://gh200:8000/v1/chat/completions

MODEL_PATH="/home/apasquale/models/Qwen3-235B-A22B-Instruct-2507-FP8"

# Check if vLLM is installed
if ! python -c "import vllm" 2>/dev/null; then
    echo "vLLM not installed. Installing..."
    pip install vllm
fi

echo "Starting vLLM server with Qwen3-235B..."
echo "Model: $MODEL_PATH"
echo "Port: 8000"
echo ""

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype auto \
    --max-model-len 8192 \
    --trust-remote-code \
    --enable-prefix-caching
