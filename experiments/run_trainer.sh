#!/bin/bash

# Define the project directory and virtualenv path
PROJECT_DIR="/home/janikbischoff/Documents/projects/slapstack-battery-management"
VENV_PATH="/home/janikbischoff/.virtualenvs/slapstack-battery-management/bin/activate"

# Change to the experiments directory
cd ${PROJECT_DIR}/experiments

# Activate the virtual environment
source ${VENV_PATH}

# Set up Python path for the required modules
export PYTHONPATH=${PYTHONPATH}:${PROJECT_DIR}/1_environment/slapstack/
export PYTHONPATH=${PYTHONPATH}:${PROJECT_DIR}/2_control/slapstack-controls/

# Run the Python script with nohup as root
sudo bash -c "
source ${VENV_PATH}
export PYTHONPATH=${PYTHONPATH}
nohup python ./trainer.py > trainer.txt 2>&1 &
echo \$! > process.pid
"

echo "Process started. PID saved in process.pid"