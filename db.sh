# dreambooth script 
autotrain dreambooth \
--model runwayml/stable-diffusion-v1-5 \
--project-name kanji-diffusion \
--image-path data/kanji_image/ \
--prompt "Kanji character" \
--resolution 128 \
--batch-size 64 \
--num-steps 500 \
--gradient-accumulation 1 \
--lr 1e-5 \
--xformers \
--push-to-hub --token ${HF_TOKEN} --repo-id Ksgk-fy/kanji-diffusion-db-v0
# $( [[ "$USE_FP16" == "True" ]] && echo "--fp16" ) \
# $( [[ "$USE_XFORMERS" == "True" ]] && echo "--xformers" ) \
# $( [[ "$TRAIN_TEXT_ENCODER" == "True" ]] && echo "--train-text-encoder" ) \
# $( [[ "$PUSH_TO_HUB" == "True" ]] && echo "--push-to-hub --token ${HF_TOKEN} --repo-id ${REPO_ID}" )