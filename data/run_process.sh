python /home/zmh/2025Aut/pjt_nlp/NLP_project_2025/data/chunk_process.py \
    --folder nlp_all \
    --chunk-radius 512 \
    -s 60000

python /home/zmh/2025Aut/pjt_nlp/NLP_project_2025/data/chunk_postprocess.py \
    --folder-name nlp_all
# final jsonl in data_s1/nlp_all/s3/dataset.jsonl