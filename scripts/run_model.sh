cd ..
source env/bin/activate
CMD="python run_model.py $1"

eval $CMD
