#!/bin/bash
# Fantasy Football Data Collection Job Setup
# This script sets up automated data collection until October 23rd

# Create the main runner script
cat > run_fantasy_collection.sh << 'EOF'
#!/bin/bash

# Fantasy Football Data Collection Runner
# Runs until October 23rd, 2025

LOG_FILE="logs/fantasy_collection_$(date +%Y%m%d).log"
ERROR_LOG="logs/fantasy_collection_errors.log"

# Create logs directory if it doesn't exist
mkdir -p logs
mkdir -p data

# Function to log messages
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to check if we should continue running (until Oct 23rd)
should_continue() {
    current_date=$(date +%Y%m%d)
    end_date="20250823"
    
    if [ "$current_date" -le "$end_date" ]; then
        return 0  # Continue
    else
        return 1  # Stop
    fi
}

# Main execution function
run_collection() {
    log_message "Starting fantasy data collection..."
    
    # Run R script first
    log_message "Running R script (fantasy-collector.R)..."
    if Rscript fantasy-collector.R >> "$LOG_FILE" 2>> "$ERROR_LOG"; then
        log_message "R script completed successfully"
    else
        log_message "ERROR: R script failed - check $ERROR_LOG"
        return 1
    fi
    
    # Run Python script
    log_message "Running Python script (mergin-data.py)..."
    if python mergin-data.py >> "$LOG_FILE" 2>> "$ERROR_LOG"; then
        log_message "Python script completed successfully"
    else
        log_message "ERROR: Python script failed - check $ERROR_LOG"
        return 1
    fi
    
    log_message "Data collection cycle completed"
    return 0
}

# Main loop
main() {
    log_message "Fantasy data collection job started - will run until October 23rd, 2025"
    
    while should_continue; do
        if run_collection; then
            log_message "Collection successful. Waiting for next run..."
        else
            log_message "Collection failed. Waiting before retry..."
        fi
        
        # Wait 6 hours between runs (adjust as needed)
        sleep 21600
    done
    
    log_message "Reached end date (October 23rd). Stopping data collection job."
}

# Handle script termination
cleanup() {
    log_message "Received termination signal. Cleaning up..."
    exit 0
}

trap cleanup SIGTERM SIGINT

# Start the main process
main
EOF

# Make the runner script executable
chmod +x run_fantasy_collection.sh

# Create systemd service file (for Linux systems)
cat > fantasy-collector.service << 'EOF'
[Unit]
Description=Fantasy Football Data Collector
After=network.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=$(pwd)
ExecStart=$(pwd)/run_fantasy_collection.sh
Restart=on-failure
RestartSec=300
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
