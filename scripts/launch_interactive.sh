#sintr --qos=INTR -A BUTTERY-SL3-GPU --exclusive -t 1:0:0  -p pascal
sintr --qos=INTR -A BUTTERY-SL3-GPU --gres=gpu:1 -t 1:0:0  -p pascal -N 1
