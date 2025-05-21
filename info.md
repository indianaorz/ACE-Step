python preprocess.py --input_dataset_path data/dataset --checkpoint_dir checkpoints --audio_root_path data/ --output_cache_path cache


python trainer.py --dataset_path cache --lora_config_path data/lora_config.json --exp_name ost --checkpoint_dir checkpoints --precision="32-true"


python -m http.server 6789 -b 0.0.0.0

python app_radio_loop.py --w exps/logs/2025-05-20_13-49-18_lofi/ --p 5001 --crossfade 0
