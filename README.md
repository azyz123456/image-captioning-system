# Image Captioning System (GroundingDINO + CLIP Augment, LLaVA-7B Fine-tune)

This project builds an end-to-end image captioning pipeline on MS COCO.

## 0) Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -c "import nltk; import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"
```

## 1) Data

Download MS COCO 2017 images and captions:
- Train images: `train2017/`
- Val images: `val2017/`
- Captions JSON: `annotations/captions_train2017.json`, `annotations/captions_val2017.json`

Set environment variables, or pass paths as CLI args:

```bash
export COCO_ROOT=/path/to/coco2017
# structure expected:
# $COCO_ROOT/train2017, $COCO_ROOT/val2017, $COCO_ROOT/annotations
```

## 2) Augmentation (GroundingDINO + CLIP)

We extract noun phrases from the human caption, pass them to GroundingDINO to get grounded boxes, crop regions,
and build region-level captions. We then filter region captions using CLIP image–text similarity.
The output is a JSONL training file mixing full-image captions and region-level examples.

```bash
python -m src.augment.augment_gdino_clip   --coco-root $COCO_ROOT   --split train   --out /tmp/coco_train_augmented.jsonl   --max-images 5000    --box-threshold 0.30 --text-threshold 0.25   --clip-threshold 0.26
```

## 3) Fine-tune LLaVA-7B with LoRA

```bash
python -m src.train.finetune_llava   --train-file /tmp/coco_train_augmented.jsonl   --val-ann $COCO_ROOT/annotations/captions_val2017.json   --val-images $COCO_ROOT/val2017   --output-dir /tmp/llava-lora-coco   --model-id llava-hf/llava-1.5-7b-hf   --epochs 1 --bsz 1 --grad-accum 16   --lr 1e-4 --image-size 336
```

> If VRAM is tight, add `--load-in-4bit` and reduce `--image-size`, `--bsz`, increase `--grad-accum`.

## 4) Evaluate (BLEU-4 + CIDEr)

After fine-tuning, run evaluation on COCO val2017:

```bash
python -m src.eval.evaluate_coco   --model /tmp/llava-lora-coco   --images $COCO_ROOT/val2017   --ann $COCO_ROOT/annotations/captions_val2017.json   --num-examples 5000   --batch-size 2   --model-id llava-hf/llava-1.5-7b-hf
```

This prints BLEU-4 and CIDEr.

## Notes

- GroundingDINO usage here relies on the Transformers implementation (`IDEA-Research/grounding-dino-tiny`).
- The LLaVA chat prompt format masks USER turns and trains on ASSISTANT tokens only.
