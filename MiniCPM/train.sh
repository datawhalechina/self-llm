num_gpus=1

deepspeed --num_gpus $num_gpus train.py \
    --deepspeed ./ds_config.json \
    --output_dir="./output/Qwen" \
    --per_device_train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --logging_steps=10 \
    --num_train_epochs=3 \
    --save_steps=500 \
    --learning_rate=1e-4 \
    --save_on_each_node=True \