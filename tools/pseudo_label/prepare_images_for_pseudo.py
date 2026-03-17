import os
import json
import argparse
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert images used for pseudo-label generation into an empty COCO-style JSON file."
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        required=True,
        help="Path to the image directory used for pseudo-label generation.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        required=True,
        help="Path to save the generated COCO-style JSON file.",
    )
    parser.add_argument(
        "--category_id",
        type=int,
        default=1,
        help="Category id used in the generated JSON. Default: 1",
    )
    parser.add_argument(
        "--category_name",
        type=str,
        default="car",
        help="Category name used in the generated JSON. Default: car",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # =========================
    # Supported image extensions
    # =========================
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    # =========================
    # Category definitions
    # NOTE:
    # The category setting should be consistent with the few-shot annotations.
    # =========================
    categories = [
        {"id": args.category_id, "name": args.category_name},
    ]

    # =========================
    # Collect image information
    # =========================
    image_files = []
    for fname in os.listdir(args.train_dir):
        fpath = os.path.join(args.train_dir, fname)
        if not os.path.isfile(fpath):
            continue
        ext = os.path.splitext(fname)[1].lower()
        if ext in valid_exts:
            image_files.append(fname)

    image_files.sort()

    images = []
    skipped = 0

    for idx, fname in enumerate(image_files, start=1):
        fpath = os.path.join(args.train_dir, fname)
        try:
            with Image.open(fpath) as img:
                width, height = img.size
        except Exception as e:
            print(f"[Warning] Skip unreadable image: {fpath}, error: {e}")
            skipped += 1
            continue

        images.append(
            {
                "id": idx,
                "width": width,
                "height": height,
                "file_name": fname,
                "license": 0,
            }
        )

    # =========================
    # Build COCO-style JSON
    # =========================
    coco_dict = {
        "images": images,
        "annotations": [],
        "categories": categories,
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(coco_dict, f, ensure_ascii=False, indent=4)

    print(f"[Done] Saved to: {args.output_json}")
    print(f"[Info] Number of valid images: {len(images)}")
    print(f"[Info] Number of skipped images: {skipped}")


if __name__ == "__main__":
    main()