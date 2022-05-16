import time 
import pickle
import signal
import os
import subprocess

count = 0
CHECKPOINT_PATH = "latest-checkpoint.pt"

def handler(signum, frame):
    print("Saving checkpoint")
    save()
    restart()
    exit(1)

signal.signal(signal.SIGINT, handler)

def save():
    global count
    fp = open(CHECKPOINT_PATH, 'wb')
    pickle.dump(count, fp)

def restart():
    subprocess.run(["sbatch", "slurm_submit.peta4-skylake"])

# Updating our 'model'
def main():
    global count
    print("initializing model")
    if os.path.exists(CHECKPOINT_PATH):
        print("reading in checkpoint")
        with open(CHECKPOINT_PATH, 'rb') as f:
            count = pickle.load(f)
    else:
        print("training from scratch")

    print("training model")
    while True: 
        print(f"model state: {count}")
        count += 1
        time.sleep(0.1)

        if count >= 500:
            print("DONE TRAINING")
            exit()

if __name__ == '__main__':
    main()