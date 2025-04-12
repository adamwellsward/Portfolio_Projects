#!/bin/bash

#SBATCH --time=8:00:00   # walltime
#SBATCH --ntasks=4   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=32000M   # memory per CPU core
#SBATCH -J "auto-harmonizer"   # job name
#SBATCH --mail-user=matthew.merrill.ward@gmail.com   # email address
#SBATCH --mail-type=FAIL
#SBATCH --qos=standby

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
source py_harmonizer/bin/activate

# Define a maximum number of retries
MAX_RETRIES=5
RETRY_COUNT=0

# Run the program and check for success
while [[ $RETRY_COUNT -lt $MAX_RETRIES ]]
do
    python search_states_filter_one_config.py --n-chords "2" --n-melody "0" --t-prior "20.0" --e-prior "150.0" --dataset "All" --keys "major" --filter-train-set "False" --filter-val-set "True"

    # Check if the job was preempted or failed
    JOB_STATUS=$(squeue -h -j $SLURM_JOB_ID -o %T)  # Check the job status using squeue
    if [[ "$JOB_STATUS" == "PD" || "$JOB_STATUS" == "ST" || "$JOB_STATUS" == "CG" ]]; then
        # If the job is pending, stopped, or canceled, increment retry count
        echo "Job preempted or failed. Retrying... ($((RETRY_COUNT+1))/$MAX_RETRIES)"
        RETRY_COUNT=$((RETRY_COUNT+1))
        # Resubmit the job
        sbatch $0
        exit 0
    else
        # Exit if job completed successfully
        exit 0
    fi
done

# If the job exceeds max retries, send a notification
echo "Job failed after $MAX_RETRIES retries." | mail -s "Job Failure Notification" matthew.merrill.ward@gmail.com
