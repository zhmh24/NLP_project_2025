python /home/zmh/2025Aut/pjt_nlp/NLP_project_2025/data_s2/chunk_process.py \
    --folder nlp_forest \
    --chunk-radius 5 \
    -s 6

python /home/zmh/2025Aut/pjt_nlp/NLP_project_2025/data_s2/chunk_postprocess.py \
    --folder-name nlp_all
# final jsonl in data_s1/nlp_all/s3/dataset.jsonl