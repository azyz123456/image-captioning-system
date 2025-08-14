import os, argparse, json, math, random, warnings
from typing import List, Dict, Any, Tuple
from PIL import Image
import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from transformers import CLIPProcessor, CLIPModel

# Simple noun phrase extractor (spaCy is heavier; fallback to NLTK if needed).
try:
    import spacy
    _NLP = spacy.load("en_core_web_sm")
    def extract_phrases(text: str) -> List[str]:
        doc = _NLP(text)
        nps = set([chunk.text.strip() for chunk in doc.noun_chunks if chunk.text.strip()])
        return list(nps)
except Exception:
    warnings.warn("spaCy model not found; falling back to a naive noun extractor.")
    import re
    def extract_phrases(text: str) -> List[str]:
        # very naive: split by commas/and; pick words longer than 2 chars as 'phrases'
        parts = re.split(r",| and | with | of ", text.lower())
        phrases = set([p.strip() for p in parts if len(p.strip()) > 2])
        return list(phrases)

from src.data.coco import COCODataset

def run_grounding_dino(
    gdino_model,
    gdino_processor,
    image: Image.Image,
    phrases: List[str],
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
) -> List[Tuple[Tuple[int,int,int,int], float, str]]:
    """Return list of (xyxy_box, score, phrase)."""
    if len(phrases) == 0:
        return []
    inputs = gdino_processor(images=image, text=phrases, return_tensors="pt")
    with torch.no_grad():
        outputs = gdino_model(**{k: v.to(gdino_model.device) for k,v in inputs.items()})
    # Post-process
    target_sizes = torch.tensor([image.size[::-1]]).to(gdino_model.device)  # h,w
    results = gdino_processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, box_threshold=box_threshold, text_threshold=text_threshold, target_sizes=target_sizes
    )
    res = results[0]
    boxes = res["boxes"].cpu().numpy().tolist()
    scores = res["scores"].cpu().numpy().tolist()
    labels = res["labels"]
    phrases_clean = [phrases[i] for i in labels]
    out = []
    for (xmin, ymin, xmax, ymax), s, p in zip(boxes, scores, phrases_clean):
        out.append(((int(xmin), int(ymin), int(xmax), int(ymax)), float(s), p))
    return out

def crop(image: Image.Image, box):
    x1, y1, x2, y2 = box
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.width, x2), min(image.height, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return image.crop((x1, y1, x2, y2))

def clip_score(clip_model, clip_proc, image: Image.Image, text: str) -> float:
    inputs = clip_proc(text=[text], images=image, return_tensors="pt", padding=True).to(clip_model.device)
    with torch.no_grad():
        out = clip_model(**inputs)
        image_embeds = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
        sim = (image_embeds @ text_embeds.T).squeeze().item()
    return float(sim)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco-root", type=str, required=True)
    ap.add_argument("--split", type=str, default="train", choices=["train","val"])  # source of captions
    ap.add_argument("--out", type=str, required=True, help="Output JSONL path of augmented data.")
    ap.add_argument("--max-images", type=int, default=10000)
    ap.add_argument("--box-threshold", type=float, default=0.30)
    ap.add_argument("--text-threshold", type=float, default=0.25)
    ap.add_argument("--clip-threshold", type=float, default=0.26)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--gdino-id", type=str, default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ds = COCODataset(args.coco_root, split=args.split)

    # GroundingDINO
    gdino_processor = AutoProcessor.from_pretrained(args.gdino_id)
    gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(args.gdino_id).to(args.device).eval()

    # CLIP
    clip_model_id = "openai/clip-vit-large-patch14"
    clip_model = CLIPModel.from_pretrained(clip_model_id).to(args.device).eval()
    clip_proc = CLIPProcessor.from_pretrained(clip_model_id)

    n_written = 0
    with open(args.out, "w") as fout:
        for i, (image_id, img_path, anns) in enumerate(tqdm(ds.iter_items(), total=len(ds))):
            if n_written >= args.max_images:
                break
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception:
                continue
            if not anns:
                continue
            # Pick one human caption (source for phrases)
            caption = random.choice(anns)["caption"].strip()

            # Always include the original full-image caption sample
            sample_full = {
                "image_path": img_path,
                "caption": caption,
                "image_id": image_id,
                "type": "full"
            }
            fout.write(json.dumps(sample_full) + "\n")
            n_written += 1

            # Phrase grounding augmentation
            phrases = extract_phrases(caption)
            if not phrases:
                continue
            dets = run_grounding_dino(gdino_model, gdino_processor, image, phrases,
                                      box_threshold=args.box_threshold, text_threshold=args.text_threshold)
            # Collect region samples
            for (box, score, phrase) in dets:
                crop_img = crop(image, box)
                if crop_img is None:
                    continue
                # Compose a simple region caption (you can swap with BLIP/LLaVA generation if desired)
                region_caption = phrase.strip()
                # CLIP score (filter)
                try:
                    s = clip_score(clip_model, clip_proc, crop_img, region_caption)
                except Exception:
                    continue
                if s < args.clip_threshold:
                    continue
                # Persist crop as ephemeral path? For simplicity, store the original image and box; training code will crop on-the-fly.
                sample_reg = {
                    "image_path": img_path,
                    "caption": region_caption,
                    "image_id": image_id,
                    "type": "region",
                    "box": box,
                    "score": float(score),
                    "clip": float(s),
                }
                fout.write(json.dumps(sample_reg) + "\n")
                n_written += 1

    print(f"Wrote ~{n_written} JSONL lines to {args.out}")

if __name__ == "__main__":
    main()
