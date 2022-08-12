python main.py \
--dataset_directory /dataset/ps2.0/training/ \
--labels_directory /dataset/ps2.0/ps_json_label/training \
--batch_size 20 \
--data_loading_workers 10 \
> FIX_BUG0725.txt 2>&1 &
