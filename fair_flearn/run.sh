python3  -u main.py --dataset='vehicle' --optimizer=$1  \
            --learning_rate=0.01 \
            --num_rounds=30 \
            --eval_every=1 \
            --clients_per_round=10 \
            --batch_size=32 \
            --t=0.1 \
            --q=$2 \
            --model='svm' \
            --sampling=2  \
            --num_epochs=1 \
            --data_partition_seed=$3 \
            --log_interval=30 \
            --static_step_size=0 \
            --track_individual_accuracy=0 \
            --output=$4



