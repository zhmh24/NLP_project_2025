python NLP_project_2025/data/jsonl_to_npy.py \
    --input-jsonl-path /home/zmh/2025Aut/pjt_nlp/generated_data_conditioned_10000.json \
    --folder-name nlp_plains \
    -l 0

python NLP_project_2025/data/write_to_world.py \
    --data_path data_s1/nlp_plains \
    --world_path worlds/pre \
    --cx 5 \
    --cz 5 \
    --y 160