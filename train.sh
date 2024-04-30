meta_path=PROJECT_PATH

CUDA_VISIBLE_DEVICES='0' python instruction-llms-safety-eval/training/finetuning.py \
    --num_epochs 3 \
    --cutoff_len 1024 \
    --val_set_size 0 \
    --base_model PATH_TO_LLAMA1 \
    --data_path $meta_path/instruction-llms-safety-eval/data/training/saferpaca_Instructions_100_2_mix.json \
    --output_dir $meta_path/instruction-llms-safety-eval/checkpoints/Instructions_100_2_mix_llama


CUDA_VISIBLE_DEVICES='1' python instruction-llms-safety-eval/training/finetuning.py \
    --num_epochs 3 \
    --cutoff_len 1024 \
    --val_set_size 0 \
    --base_model PATH_TO_LLAMA2 \
    --prompt_template_name llama2 \
    --data_path $meta_path/instruction-llms-safety-eval/data/training/saferpaca_Instructions_100_2_mix.json \
    --output_dir $meta_path/instruction-llms-safety-eval/checkpoints/Instructions_100_2_mix_llama2 
