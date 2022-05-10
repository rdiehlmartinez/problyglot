cd ..
source env/bin/activate
CMD="python run_model.py $@"

eval $CMD
