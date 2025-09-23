import numpy as np
import subprocess
import itertools
import os

def get_bash_script(n_chords, n_melody, t_prior, e_prior, dataset, keys, filter_train, filter_val):
    return f'''#!/bin/bash

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
    python search_states_filter_one_config.py --n-chords "{n_chords}" --n-melody "{n_melody}" --t-prior "{t_prior}" --e-prior "{e_prior}" --dataset "{dataset}" --keys "{keys}" --filter-train-set "{filter_train}" --filter-val-set "{filter_val}"

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
'''

def create_jobs(grids):
    # Create the directory for the scripts if it doesn't exist
    results_dir = './grid_search'
    scripts_dir = './grid_search_scripts'
    os.makedirs(scripts_dir, exist_ok=True)

    for grid in grids:
        for n_chords, n_melody, t_prior, e_prior, dataset, keys, filter_train, filter_val in itertools.product(
            grid['n_chords'],
            grid['n_melody'],
            grid['t_prior'],
            grid['e_prior'],
            grid['dataset'],
            grid['keys'],
            grid['filter_train_set'],
            grid['filter_val_set']
        ):
            # Check if results were already calculated
            fname = f"{n_chords} {n_melody} {t_prior} {e_prior} {dataset} {keys} {filter_train} {filter_val}"
            results_path = os.path.join(results_dir, f"{fname}.txt")
            if os.path.exists(results_path):
                continue

            # Generate bash script with all flags
            bash_script = get_bash_script(
                n_chords, n_melody, t_prior, e_prior,
                dataset, keys, filter_train, filter_val
            )

            # Create the script file in the grid_search_scripts directory
            script_filename = f"{scripts_dir}/{fname}.sh"
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
            'e_prior': np.linspace(50, 4000, 5, dtype=float),
            'dataset': ['All', 'ABC Notation', 'JSB Chorales', 'The Session', 'FolkWiki', 'Nottingham'],
            'keys': ['All', 'minor', 'major', 'mixolydian', 'dorian', 'phrygian', 'lydian'],
            'filter_train_set': [True, False],
            'filter_val_set': [True, False]
        },
        {
            'n_chords': [1],
            'n_melody': [1],
            't_prior': np.linspace(0, 30, 5, dtype=float),
            'e_prior': np.linspace(0, 30, 5, dtype=float),
            'dataset': ['All', 'ABC Notation', 'JSB Chorales', 'The Session', 'FolkWiki', 'Nottingham'],
            'keys': ['All', 'minor', 'major', 'mixolydian', 'dorian', 'phrygian', 'lydian'],
            'filter_train_set': [True, False],
            'filter_val_set': [True, False]
        },
        {
            'n_chords': [2],
            'n_melody': [0],
            't_prior': np.linspace(20, 300, 5, dtype=float),
            'e_prior': np.linspace(0, 300, 5, dtype=float),
            'dataset': ['All', 'ABC Notation', 'JSB Chorales', 'The Session', 'FolkWiki', 'Nottingham'],
            'keys': ['All', 'minor', 'major', 'mixolydian', 'dorian', 'phrygian', 'lydian'],
            'filter_train_set': [True, False],
            'filter_val_set': [True, False]
        },
        {
            'n_chords': [2],
            'n_melody': [1],
            't_prior': np.linspace(0, 10, 5, dtype=float),
            'e_prior': np.linspace(0, 10, 5, dtype=float),
            'dataset': ['All', 'ABC Notation', 'JSB Chorales', 'The Session', 'FolkWiki', 'Nottingham'],
            'keys': ['All', 'minor', 'major', 'mixolydian', 'dorian', 'phrygian', 'lydian'],
            'filter_train_set': [True, False],
            'filter_val_set': [True, False]
        },
        {
            'n_chords': [2],
            'n_melody': [2],
            't_prior': np.linspace(0, 2, 5, dtype=float),
            'e_prior': np.linspace(0, 2, 5, dtype=float),
            'dataset': ['All', 'ABC Notation', 'JSB Chorales', 'The Session', 'FolkWiki', 'Nottingham'],
            'keys': ['All', 'minor', 'major', 'mixolydian', 'dorian', 'phrygian', 'lydian'],
            'filter_train_set': [True, False],
            'filter_val_set': [True, False]
        }
    ]

    create_jobs(grids)

