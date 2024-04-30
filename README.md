# safety-defense
Code and data for paper: Cross-Task Defense: Instruction-Tuning LLMs for Content Safety (NNACL2024 TrustNLP Workshop)


# Data

The training data are placed in `instruction-llms-safety-eval/data/training`. Take sentiment task as example:
- `/data/training/sentiment/saferpaca_Instructions_100_10_sentiment.json`  -- 10 sentiment examples
- `/data/training/sentiment/saferpaca_Instructions_100_100_sentiment.json` -- 100 sentiment examples
    ...

The evaluation data are placed in `instruction-llms-safety-eval/data/evaluation`.

# Train
```python
CUDA_VISIBLE_DEVICES='0' python instruction-llms-safety-eval/training/finetuning.py \
    --num_epochs 3 \
    --cutoff_len 1024 \
    --val_set_size 0 \
    --base_model PATH_TO_LLAMA1 \
    --data_path $meta_path/instruction-llms-safety-eval/data/training/mix/saferpaca_Instructions_100_2_mix.json \
    --output_dir $meta_path/instruction-llms-safety-eval/checkpoints/Instructions_100_2_mix_llama
```
Please modified the `meta_path` to your current project directory and change `PATH_TO_LLAMA1` to your local directory. This code do the mix training and if you want to just use summarization dataset, you should change the `data_path` to `$meta_path/instruction-llms-safety-eval/data/training/summarize/saferpaca_Instructions_100_10_summrize.json`.

Please modify `meta_path` to your current project directory and change `PATH_TO_LLAMA1` to your local model directory. This code performs mixed training. If you only want to use the summarization dataset for training, you should change `data_path` to `$meta_path/instruction-llms-safety-eval/data/training/summarize/saferpaca_Instructions_100_10_summarize.json`.

# Generate
```python
devices=(7)
numbers=(1000)
tasks=('translate')

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
```

After training with various numbers of defense examples, we further evaluate the defensive capabilities of these models. The code above provides an example of how to evaluate NSP examples on the model trained with translation examples.


# Reference code
We build our project based on [Link](https://github.com/vinid/safety-tuned-llamas/tree/main). Please refer to the linked paper and code for more details.

