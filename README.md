# pytorch-Centernet-for-synthetic-data
Centernet training on synthetic data(combine objects with background)


1. train
python main_patcher.py  --model_name="centernet" --backbone="dla34" --task="ctdetp" --data_dir="directroy that patch or background image existed " --hide_data_time --batch_size 64 --subdivision 4 --master_batch 1 --lr 1.25e-4 --gpus 0 --num_workers 2  --num_epochs 1000 

2. demo
python demo.py 
