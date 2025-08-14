import os, json, random
from typing import List, Dict, Optional
from PIL import Image

class COCODataset:
    """Minimal COCO captions loader.

    Expects:

    root/

      train2017/

      val2017/

      annotations/

        captions_train2017.json

        captions_val2017.json

    """

    def __init__(self, coco_root: str, split: str = "train"):
        self.coco_root = coco_root
        self.split = split
        ann_path = os.path.join(coco_root, "annotations", f"captions_{split}2017.json")
        with open(ann_path, "r") as f:
            ann = json.load(f)

        self.images = {img["id"]: img for img in ann["images"]}
        self.img_dir = os.path.join(coco_root, f"{split}2017")
        self.ann_by_img: Dict[int, List[Dict]] = {}
        for ca in ann["annotations"]:
            self.ann_by_img.setdefault(ca["image_id"], []).append(ca)

        self.ids = list(self.images.keys())

    def __len__(self):
        return len(self.ids)

    def get_image_path(self, image_id: int) -> str:
        file_name = self.images[image_id]["file_name"]
        return os.path.join(self.img_dir, file_name)

    def get_random_caption(self, image_id: int) -> str:
        caps = self.ann_by_img.get(image_id, [])
        if not caps:
            return ""
        return random.choice(caps)["caption"].strip()

    def iter_items(self):
        for image_id in self.ids:
            yield image_id, self.get_image_path(image_id), self.ann_by_img.get(image_id, [])
