#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Cleanup any lingering processes ---
echo "Ensuring no stray servers are running..."
pkill -f uvicorn || true

# --- Static Analysis ---
echo "Running linter (ruff)..."
ruff check .

echo "Running static type checker (mypy)..."
mypy .

# --- Tests ---

# Create a directory for this test run's artifacts
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
export TEST_RUN_DIR="test_runs/$TIMESTAMP"
mkdir -p "$TEST_RUN_DIR"
echo "Test artifacts will be saved to: $TEST_RUN_DIR"

# Start the server in the background
echo "Starting Uvicorn server in the background..."
# Use the uvicorn from the virtual environment to ensure correct dependencies
./venv/bin/uvicorn app:app &
UVICORN_PID=$!

# A function to stop the server, which we'll call on exit
cleanup() {
    echo "Shutting down Uvicorn server (PID: $UVICORN_PID)..."
    # The [ -n "$UVICORN_PID" ] checks if the variable is not empty.
    # The ps -p checks if the process exists.
    if [ -n "$UVICORN_PID" ] && ps -p $UVICORN_PID > /dev/null; then
        kill $UVICORN_PID
    else
        echo "Server not running or PID not found."
    fi
}

# The 'trap' command ensures that the 'cleanup' function is called when the script exits,
# for any reason (e.g., success, failure, or interrupt).
trap cleanup EXIT

# Wait for the server to be ready
echo "Waiting for server to initialize..."
RETRY_COUNT=0
MAX_RETRIES=10
until nc -z 127.0.0.1 8000; do
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "Server failed to start after $MAX_RETRIES attempts."
        exit 1
    fi
    RETRY_COUNT=$((RETRY_COUNT+1))
    sleep 1
done

# Run pytest and store its exit code
echo "Running tests..."
./venv/bin/pytest
PYTEST_EXIT_CODE=$?

# The script will now exit, and the trap will automatically run the cleanup.
# We exit with the same code as pytest, so if tests fail, the script fails.
exit $PYTEST_EXIT_CODE
