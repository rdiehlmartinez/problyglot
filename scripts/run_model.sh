cd ..
source env/bin/activate

while getopts m: flag
do
    case "${flag}" in
        m) mode=${OPTARG};;
    esac
done

CMD="python run_model.py $1"

mode=${mode:-"train"}
if [[ $mode == "train" ]]; then
	CMD+=" --train-model"
elif [[ $mode == "eval" ]]; then
	CMD+=" --eval-model"
else 
	echo "Invalid model mode flag: $mode (should be either 'train' or 'eval')"
fi 

eval $CMD
