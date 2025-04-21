#!/bin/bash

# Enable conda commands in this script
source /opt/conda/etc/profile.d/conda.sh

# Activate your environment
conda activate meteo473_sp25

# Navigate to your project directory
cd /courses/meteo473_sp25/group5/

# Run your Python script
python Milestone2Master.py

# I set up the cron Job to run my script.sh file at midnight every day

# to run Script open a new terminal and copy and paste the following line
#meteo473/groupwork/group5/script.sh