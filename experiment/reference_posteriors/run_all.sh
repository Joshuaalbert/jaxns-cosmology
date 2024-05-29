#!/bin/bash

# Define an array to hold the PIDs of the background processes
declare -a pids

# Function to kill all background processes
cleanup() {
    echo "Cleaning up..."
    for pid in "${pids[@]}"; do
        kill -9 "$pid" 2>/dev/null
    done
}

# Trap INT and TERM signals and invoke cleanup
trap cleanup INT TERM

for model_name in CMB eggbox MSSM7 rastrigin rosenbrock spikeslab; do
  python main.py "$model_name" &
  pids+=("$!") # Store the PID of the background process
done

# Wait for all background processes to complete
wait

echo "All processes have completed."
