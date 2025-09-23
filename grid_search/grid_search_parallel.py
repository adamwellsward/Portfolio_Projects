import numpy as np
import subprocess
import itertools
import os

def get_bash_script(n_chords, n_melody, t_prior, e_prior):
    return f"""#!/bin/bash

#SBATCH --time=8:00:00   # walltime
#SBATCH --ntasks=4   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=32000M   # memory per CPU core
#SBATCH -J "auto-harmonizer"   # job name
#SBATCH --mail-user=matthew.merrill.ward@gmail.com   # email address
#SBATCH --mail-type=END
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
    python search_states_one_config.py --n_chords {n_chords} --n_melody {n_melody} --t_prior {t_prior} --e_prior {e_prior}

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
"""

def create_jobs(grids):
    # Create the directory for the scripts if it doesn't exist
    results_dir = './grid_search'
    scripts_dir = './grid_search_scripts'
    os.makedirs(scripts_dir, exist_ok=True)

    for grid in grids:
        n_chords_range = grid['n_chords']
        n_melody_range = grid['n_melody']
        t_prior_range = grid['t_prior']
        e_prior_range = grid['e_prior']

        for n_chords, n_melody, t_prior, e_prior in itertools.product(
            n_chords_range, 
            n_melody_range, 
            t_prior_range, 
            e_prior_range
        ):
            # Check if results were already calculated
            results_path = os.path.join(results_dir, f'{n_chords}_{n_melody}_{t_prior}_{e_prior}.txt')
            if os.path.exists(results_path):
                continue

            bash_script = get_bash_script(n_chords, n_melody, t_prior, e_prior)

            # Create the script file in the grid_search_scripts directory
            script_filename = f"{scripts_dir}/{n_chords}_{n_melody}_{t_prior}_{e_prior}.sh"
            with open(script_filename, 'w') as script_file:
                script_file.write(bash_script)

            # Make the script executable
            subprocess.run(["chmod", "+x", script_filename])

            # Submit job to slurm, redirect error to the void so the script ends  
            subprocess.Popen(["sbatch", script_filename], stderr=subprocess.DEVNULL)
    
    return

if __name__ == '__main__':
    grids = [
        {
            'n_chords': [1],
            'n_melody': [0],
            't_prior': np.linspace(50, 4000, 5, dtype=float),
            'e_prior': np.linspace(50, 4000, 5, dtype=float)
        },
        {
            'n_chords': [1],
            'n_melody': [1],
            't_prior': np.linspace(0, 30, 5, dtype=float),
            'e_prior': np.linspace(0, 30, 5, dtype=float)
        },
        {
            'n_chords': [2],
            'n_melody': [0],
            't_prior': np.linspace(20, 300, 5, dtype=float),
            'e_prior': np.linspace(0, 300, 5, dtype=float)
        },
        {
            'n_chords': [2],
            'n_melody': [1],
            't_prior': np.linspace(0, 10, 5, dtype=float),
            'e_prior': np.linspace(0, 10, 5, dtype=float)
        },
        {
            'n_chords': [2],
            'n_melody': [2],
            't_prior': np.linspace(0, 2, 5, dtype=float),
            'e_prior': np.linspace(0, 2, 5, dtype=float)
        }
    ]

    create_jobs(grids)

