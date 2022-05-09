# Working example demonstrating how to run never-ending jobs on CSD3

Main logic lives in file boop.py

We launch csd3 by calling ```sbatch slurm_submit.peta4-skylake``` (if you are not paula's student change the billing info obviously) which in turn calls boop.py until it has counted up to some number. The program only lives for 90 seconds, and 60 seconds before termination CSD3 sends SIGINT to the main process which in turn has to save whatever the number is that is currently has counted up to, and calls on CSD3 to spawn a new job.
