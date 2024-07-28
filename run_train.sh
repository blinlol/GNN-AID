cd GraphNAS
stdbuf -i0 -o0 -e0 python -m debugpy --listen 3342 graphnas/main.py \
--mode train \
--search_mode micro \
--submanager_log_file .txt \
--dataset Citeseer \
> out \
>> err &