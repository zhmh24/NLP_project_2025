python jsonl_to_npy.py \
    --input_jsonl_path data_s1/nlp_plains/generated_blocks.jsonl \
    --folder_name nlp_plains \
    -l 0

python write_to_world.py \
    --data_path data_s1/nlp_plains \
    --world_path worlds/pre \
    --cx 0 \
    --cz 0 \
    --y 160