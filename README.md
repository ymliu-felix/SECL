Code of paper "Evolution Rather Than Degradation: Structure-Guided Elastic Consensus Learning for Multimodal Knowledge Graph Completion".

## Dataset Download

Download raw data of FB15K-237 and WN18RR from [MMKB](https://github.com/mniepert/mmkb), [RSME](https://github.com/wangmengsd/RSME), or [MKGformer](https://github.com/zjunlp/MKGformer), MKG-W and MKG-Y from [MMRNS](https://github.com/quqxui/MMRNS).

## Dataset Preprocess

Preprocess triples by running ```src/process_datasets.py```. Extract visual and textual features using [CLIP](https://huggingface.co/openai/clip-vit-large-patch14). After extracting features from fixed encoders, we save the text and image features of entities in a pickle file and save the file in data/DATASET_NAME/.

## Training

You can train the model by running ```src/learner.py```. For example,

```
python learn.py --model ComplExMDR --ckpt_dir CKPT_DIR --dataset WN18RR --early_stopping 10 --fusion_dscp True --fusion_img True --modality_split True --img_info PATH_VISUAL  --dscp_info PATH_TEXTUAL
```

## Inference

After obtaining a trained model, you can get final predictions and performance by running
```
python boosting_inference.py --model_path YOUR_MODEL_PATH --dataset DATASET_NAME --boosting True
```

