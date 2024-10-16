#!/bin/bash

# Check if at least one argument is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 command [args...]"
    exit 1
fi

# Extract the command and arguments
command="$1"
shift  # Shift off the first argument to get the rest as arguments

while true; do
	clear
    # Execute the command with its arguments
    "$command" "$@"
    
    # Sleep for 0.5 seconds
    sleep 1
done
