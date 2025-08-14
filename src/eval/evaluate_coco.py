import os, argparse, json, random
from typing import List, Dict, Any
from PIL import Image
from tqdm import tqdm

import torch
from transformers import AutoProcessor, AutoModelForCausalLM

from nltk.tokenize import word_tokenize

from src.data.coco import COCODataset
from src.eval.metrics import bleu4, cider

def generate_captions(model, processor, images: List[Image.Image], max_new_tokens=64):
    prompts = ["USER: <image>\nPlease provide a concise caption for this image.\nASSISTANT:" for _ in images]
    enc = processor(images=images, text=prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
    decoded = processor.batch_decode(out, skip_special_tokens=True)
    # Strip everything up to last 'ASSISTANT:'
    captions = []
    for txt in decoded:
        i = txt.rfind("ASSISTANT:")
        cap = txt[i+len("ASSISTANT:"):].strip() if i != -1 else txt.strip()
        captions.append(cap)
    return captions

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="Path to fine-tuned LoRA adapter dir or full model dir.")
    ap.add_argument("--images", type=str, required=True)
    ap.add_argument("--ann", type=str, required=True)
    ap.add_argument("--num-examples", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--model-id", type=str, default="llava-hf/llava-1.5-7b-hf")
    args = ap.parse_args()

    # Load base + adapter (the Auto* API will merge if the adapter is saved with PEFT)
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")

    # Data
    ds = COCODataset(os.path.dirname(args.images), split="val")  # expects images under $ROOT/val2017
    ids = ds.ids[: args.num_examples]

    preds: Dict[int, str] = {}
    refs: Dict[int, List[str]] = {}

    batch = []
    batch_ids = []
    for image_id in tqdm(ids):
        img_path = ds.get_image_path(image_id)
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue
        batch.append(img)
        batch_ids.append(image_id)
        if len(batch) == args.batch_size:
            caps = generate_captions(model, processor, batch)
            for iid, c in zip(batch_ids, caps):
                preds[iid] = c
            batch = []; batch_ids = []

        # refs
        caps_gt = [a["caption"].strip() for a in ds.ann_by_img.get(image_id, [])]
        refs[image_id] = caps_gt

    if batch:
        caps = generate_captions(model, processor, batch)
        for iid, c in zip(batch_ids, caps):
            preds[iid] = c

    # Prepare for BLEU-4
    hyp_tok = [[word_tokenize(preds[iid] if iid in preds else "")] for iid in refs.keys()]
    ref_tok = [[word_tokenize(c) for c in refs[iid]] for iid in refs.keys()]

    b4 = bleu4(hypotheses=[h[0] for h in hyp_tok], references=ref_tok)
    cd = cider(preds, refs)

    print(json.dumps({"BLEU-4": b4, "CIDEr": cd}, indent=2))

if __name__ == "__main__":
    main()
