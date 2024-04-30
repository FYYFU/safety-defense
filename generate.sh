devices=(7)
numbers=(1000)
tasks=('translate' 'case' 'cloze' 'summarize')

for((i=0;i<1;i++))do
    device=${devices[i]}
    number=${numbers[0]}
    task=${tasks[i]}

    CUDA_VISIBLE_DEVICES=$device nohup python generation/generate_answers.py \
        --prompt_template_path ./configs/alpaca.json \
        --input_path ./data/evaluation/test_case.json \
        --output_path ./evaluation/${task}/ \
        --base_model PATH_TO_LLAMA \
        --lora_weights ./checkpoints/${task}/Instructions_100_${number}_${task}_llama/ \
        --load_8bit > ./logs/$task/eval_llama-${task}_${number}.log 2>&1 &

    CUDA_VISIBLE_DEVICES=$device nohup python generation/generate_answers.py \
        --prompt_template_path ./configs/llama2.json \
        --input_path ./data/evaluation/test_case.json \
        --output_path ./evaluation/${task}/ \
        --base_model PATH_TO_LLAMA2 \
        --lora_weights ./checkpoints/${task}/Instructions_100_${number}_${task}_llama2 \
        --load_8bit > ./logs/$task/eval_llama2-${task}_${number}.log 2>&1 &
done