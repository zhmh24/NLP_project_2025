python NLP_project_2025/data/jsonl_to_npy.py \
    --input-jsonl-path /home/zmh/2025Aut/pjt_nlp/generated_multiple_data_5000.jsonl \
    --folder-name nlp_all \
    -l 0

python NLP_project_2025/data/write_to_world.py \
    --data_path data_s1/nlp_all \
    --world_path worlds/pre \
    --cx 13 \
    --cz 0 \
    --y 160