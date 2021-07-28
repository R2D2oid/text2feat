# text2feat
Extracts feature embeddings for text input using a pretrained universal-setence-encoder or fasttext model. 

The input may be provided as json captions as in vide-text annotations or as a list of strings. See the code for more details.

## Environment Setup
```
virtualenv --system-site-packages -p python3 env_t2f
source env_t2f/bin/activate
pip install -r requirements.txt
```

### Extract Sentence Feats for a list of sentences
```
python text2feat.py --output-dir output --input-path data/sentences.txt --input-type list
```

## Extract Sentence Feats for MSRVTT captions
```
python text2feat.py --output-dir msrvtt_caption_feats_universal --input-path ../datasets/MSRVTT/train_val_annotation/train_val_videodatainfo.json --input-type json
```
