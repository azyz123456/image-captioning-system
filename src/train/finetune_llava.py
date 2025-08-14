import os, argparse, json, math, random, warnings
from typing import List, Dict, Any, Optional
from PIL import Image, ImageOps
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import get_linear_schedule_with_warmup, AdamW
from peft import LoraConfig, get_peft_model, TaskType

from src.data.coco import COCODataset

IGNORE_INDEX = -100

def mask_labels_for_chat(input_ids: torch.LongTensor, tokenizer, assistant_token="ASSISTANT:"):
    """Mask labels up to and including the first occurrence of 'ASSISTANT:' marker.
    Assumes prompt format: 'USER: <image>\nDescribe...\nASSISTANT: <caption>'
    """
    labels = input_ids.clone()
    # Find index of first occurrence of assistant token span, if available
    # We'll search via raw string on decoded batchwise for simplicity
    for i in range(labels.size(0)):
        text = tokenizer.decode(input_ids[i], skip_special_tokens=False)
        idx = text.find(assistant_token)
        if idx == -1:
            labels[i, :] = IGNORE_INDEX
            continue
        # Tokenize split and mask tokens up to ASSISTANT:
        pre = text[: text.find(assistant_token)+len(assistant_token)]
        pre_ids = tokenizer(pre, add_special_tokens=False).input_ids
        keep_from = len(pre_ids)
        # Mask everything before 'keep_from'
        labels[i, :keep_from] = IGNORE_INDEX
    return labels

class JSONLAugmentedDataset(Dataset):
    """Reads JSONL lines with keys: image_path, caption, (optional) box."""
    def __init__(self, jsonl_path: str):
        self.items = []
        with open(jsonl_path, "r") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    self.items.append(obj)
                except Exception:
                    continue

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        img = Image.open(it["image_path"]).convert("RGB")
        if it.get("type") == "region" and it.get("box") is not None:
            x1,y1,x2,y2 = it["box"]
            img = img.crop((x1,y1,x2,y2))
        return {
            "image": img,
            "caption": it["caption"],
        }

def build_prompt(caption: str) -> str:
    # Minimal chat-style instruction prompt for LLaVA
    # LLaVA expects an <image> token in text input (processor will insert proper placeholders)
    user = "USER: <image>\nPlease provide a concise caption for this image.\n"
    assistant = f"ASSISTANT: {caption}"
    return user + assistant

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-file", type=str, required=True, help="Augmented JSONL from augment step.")
    ap.add_argument("--val-ann", type=str, required=True, help="COCO val captions JSON (for periodic sanity checks)." )
    ap.add_argument("--val-images", type=str, required=True, help="COCO val images dir." )
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--model-id", type=str, default="llava-hf/llava-1.5-7b-hf")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bsz", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--image-size", type=int, default=336)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--load-in-4bit", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    # Processor & Model (HF LLaVA)
    processor = AutoProcessor.from_pretrained(args.model_id)
    quant_config = None
    device_map = "auto"
    if args.load_in_4bit:
        quant_config = BitsAndBytesConfig(load_in_4bit=True,
                                          bnb_4bit_compute_dtype=torch.bfloat16,
                                          bnb_4bit_quant_type="nf4",
                                          bnb_4bit_use_double_quant=True)
        device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        quantization_config=quant_config,
        device_map=device_map
    )

    # LoRA
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # Data
    train_ds = JSONLAugmentedDataset(args.train_file)

    def collate(batch):
        images = [b["image"] for b in batch]
        texts = [build_prompt(b["caption"]) for b in batch]
        enc = processor(images=images, text=texts, return_tensors="pt", padding=True)
        input_ids = enc["input_ids"]
        labels = mask_labels_for_chat(input_ids, processor.tokenizer, assistant_token="ASSISTANT:")
        enc["labels"] = labels
        return {k: v for k,v in enc.items()}

    train_loader = DataLoader(train_ds, batch_size=args.bsz, shuffle=True, num_workers=4, collate_fn=collate)

    # Optimizer / Scheduler
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    total_steps = args.epochs * math.ceil(len(train_loader) / args.grad_accum)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.03*total_steps), num_training_steps=total_steps)

    model.train()
    step = 0
    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}")
        optimizer.zero_grad(set_to_none=True)
        for batch in pbar:
            batch = {k: (v.to(model.device) if isinstance(v, torch.Tensor) else v) for k,v in batch.items()}
            out = model(**batch)
            loss = out.loss
            (loss / args.grad_accum).backward()
            if (step + 1) % args.grad_accum == 0:
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            step += 1

        # Save LoRA-adapter checkpoint each epoch
        model.save_pretrained(os.path.join(args.output_dir, f"lora-epoch{epoch+1}"))
        processor.save_pretrained(args.output_dir)

    # Final save
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Saved LoRA adapter + processor to {args.output_dir}")

if __name__ == "__main__":
    main()
