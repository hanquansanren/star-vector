#!/usr/bin/env python
"""
从完整的 svg-stack 数据集中提取一个小规模 toy 子集，便于快速调试。
默认创建 200 train / 20 val / 20 test，可通过命令行参数自定义。
    python scripts/tools/make_toy_dataset.py \
      --source /Data_PHD/phd23_weiguang_zhang/project/svg_data/svg-stack-hf \
      --target /Data_PHD/phd23_weiguang_zhang/project/svg_data/svg-stack-toy \
      --train 200 --val 20 --test 20 --seed 42 --overwrite
"""

import argparse
import os
import shutil
from pathlib import Path

from datasets import DatasetDict, load_from_disk

DEFAULT_SOURCE = Path("/Data_PHD/phd23_weiguang_zhang/project/svg_data/svg-stack-hf")
DEFAULT_TARGET = Path("/Data_PHD/phd23_weiguang_zhang/project/svg_data/svg-stack-toy")


def take_subset(dataset, n_samples, seed):
    """返回 dataset 的随机子集；若 n_samples<=0 或超过长度则返回原数据。"""
    if n_samples <= 0 or n_samples >= len(dataset):
        return dataset
    return dataset.shuffle(seed=seed).select(range(n_samples))


def parse_args():
    parser = argparse.ArgumentParser(description="Make toy subset for svg-stack dataset.")
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"完整数据集所在目录 (默认: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=DEFAULT_TARGET,
        help=f"toy 数据集输出目录 (默认: {DEFAULT_TARGET})",
    )
    parser.add_argument("--train", type=int, default=200, help="toy train 样本数")
    parser.add_argument("--val", type=int, default=20, help="toy val 样本数")
    parser.add_argument("--test", type=int, default=20, help="toy test 样本数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="若 target 已存在则先删除后再写入",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.source.exists():
        raise FileNotFoundError(f"未找到源数据集目录：{args.source}")

    if args.target.exists():
        if args.overwrite:
            shutil.rmtree(args.target)
        else:
            raise FileExistsError(
                f"目标目录 {args.target} 已存在，若要覆盖请添加 --overwrite"
            )

    dataset_dict = load_from_disk(str(args.source))
    toy_dict = {}

    if "train" in dataset_dict:
        toy_dict["train"] = take_subset(dataset_dict["train"], args.train, args.seed)
    if "val" in dataset_dict:
        toy_dict["val"] = take_subset(dataset_dict["val"], args.val, args.seed + 1)
    if "test" in dataset_dict:
        toy_dict["test"] = take_subset(dataset_dict["test"], args.test, args.seed + 2)

    if not toy_dict:
        raise ValueError("源数据集中没有 train/val/test 任一 split，无法生成 toy 数据。")

    toy_dataset = DatasetDict(toy_dict)
    toy_dataset.save_to_disk(str(args.target))
    print(
        f"Toy 数据集已保存至 {args.target}："
        f"train={len(toy_dict.get('train', []))}, "
        f"val={len(toy_dict.get('val', []))}, "
        f"test={len(toy_dict.get('test', []))}"
    )


if __name__ == "__main__":
    main()

