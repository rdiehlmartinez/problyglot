# EXAMPLE DEMONSTRATING HOW TO CONTINUOUSLY CALL ON CSD3 TO RUN NEVER ENDING JOBS

Checkout the main file boop.py; we launch csd3 by calling ```sbatch slurm_submit.peta4``` which in turn calls boop.py until it has counted up to 500.
