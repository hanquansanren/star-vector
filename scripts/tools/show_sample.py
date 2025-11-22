import random
from pathlib import Path
from datasets import load_from_disk
from starvector.data.util import rasterize_svg

DATA_DIR = Path("/Data_PHD/phd23_weiguang_zhang/project/svg_data/svg-stack-hf")
SPLIT = "train"   # 也可以改成 "test" / "val"
INDEX = None      # 指定整数可固定某个样本，None 表示随机挑选

def main():
    dataset_dict = load_from_disk(str(DATA_DIR))
    dataset = dataset_dict[SPLIT]

    idx = INDEX if INDEX is not None else random.randrange(len(dataset))
    sample = dataset[idx]

    svg_str = sample["Svg"]
    filename = sample["Filename"]
    caption = sample.get("caption_blip2") or sample.get("caption_llava") or ""

    print(f"Split: {SPLIT}, Index: {idx}, Filename: {filename}")
    print(f"Caption: {caption}")

    # 将 SVG 树转成 PIL.Image
    image = rasterize_svg(svg_str, resolution=224)
    # image.show()  # X11/本地桌面可弹窗；若是远程无界面，可改成保存到文件

    # 如需保存文件，可用：
    out_path = Path("outputs/tmp/tmp_sample.png")
    image.save(out_path)
    print(f"Saved image to {out_path}")

if __name__ == "__main__":
    main()