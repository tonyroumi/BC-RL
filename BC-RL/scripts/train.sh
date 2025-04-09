DATASET_ROOT='dataset/'

python3 train.py $DATASET_ROOT --dataset carla --train-towns 1 3  --val-towns 2 \
    --train-weathers 0 1 2 3 4 5   --val-weathers 6 7 \
    --model DEFINEMODELHERE --sched cosine --epochs 25 --warmup-epochs 5 --lr 0.0005 --batch-size 16  -j 16 --no-prefetcher --eval-metric l1_error \
    --opt adamw --opt-eps 1e-8 --weight-decay 0.05  \
    --scale 0.9 1.1 --saver-decreasing --clip-grad 10 --freeze-num -1 \
    --with-backbone-lr --backbone-lr 0.0002 \
    --multi-view --with-lidar --multi-view-input-size 3 128 128 \
    --experiment interfuser_baseline \
    --pretrained