#!/usr/bin/env python
"""
从完整的 svg-stack HF 数据集中抽取约 1% 的子集（含 train/val/test），
用于更快地验证和调试。

默认：
- 源数据集：/Data_PHD/phd23_weiguang_zhang/project/svg_data/svg-stack-hf
- 目标数据集：/Data_PHD/phd23_weiguang_zhang/project/svg_data/svg-stack-hf-1percent
- 子集比例：1%（ratio = 0.01）
- 测试集数量上限：50 条（避免验证过慢）

运行示例：
    cd /Data_PHD/phd23_weiguang_zhang/project/star-vector
    python scripts/tools/make_svg_stack_1percent.py \
        --ratio 0.01 \
        --max_test 50 \
        --overwrite
"""

import argparse
from pathlib import Path
import shutil

from datasets import DatasetDict, load_from_disk


DEFAULT_SOURCE = Path(
    "/Data_PHD/phd23_weiguang_zhang/project/svg_data/svg-stack-hf"
)
DEFAULT_TARGET = Path(
    "/Data_PHD/phd23_weiguang_zhang/project/svg_data/svg-stack-hf-1percent"
)


def take_ratio_subset(dataset, ratio: float, max_samples: int | None, seed: int):
    """按照比例随机采样子集，可选最大样本数限制。

    - ratio <= 0 或 ratio >= 1 时，直接返回原数据
    - 若 max_samples 不为 None，则最终样本数为 min(按比例数量, max_samples)
    """
    n_total = len(dataset)
    if ratio <= 0 or ratio >= 1:
        n_samples = n_total
    else:
        n_samples = max(1, int(n_total * ratio))

    if max_samples is not None:
        n_samples = min(n_samples, max_samples)

    if n_samples >= n_total:
        return dataset

    return dataset.shuffle(seed=seed).select(range(n_samples))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Make ~1% subset for svg-stack HF dataset."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"完整 HF 数据集所在目录 (默认: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=DEFAULT_TARGET,
        help=f"子集数据集输出目录 (默认: {DEFAULT_TARGET})",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.01,
        help="子集比例，例如 0.01 表示 1%%（默认: 0.01）",
    )
    parser.add_argument(
        "--max_test",
        type=int,
        default=50,
        help="测试集最大样本数（默认: 50，避免验证过慢）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（用于打乱后采样）",
    )
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
    small_dict: dict[str, object] = {}

    # train / val 使用统一比例
    if "train" in dataset_dict:
        train_src = dataset_dict["train"]
        small_dict["train"] = take_ratio_subset(
            train_src, ratio=args.ratio, max_samples=None, seed=args.seed
        )
        print(f"train: {len(train_src)} -> {len(small_dict['train'])}")

    if "val" in dataset_dict:
        val_src = dataset_dict["val"]
        small_dict["val"] = take_ratio_subset(
            val_src, ratio=args.ratio, max_samples=None, seed=args.seed + 1
        )
        print(f"val:   {len(val_src)} -> {len(small_dict['val'])}")

    # test 既按比例也受 max_test 限制（更小更快）
    if "test" in dataset_dict:
        test_src = dataset_dict["test"]
        small_dict["test"] = take_ratio_subset(
            test_src,
            ratio=args.ratio,
            max_samples=args.max_test if args.max_test > 0 else None,
            seed=args.seed + 2,
        )
        print(f"test:  {len(test_src)} -> {len(small_dict['test'])}")

    if not small_dict:
        raise ValueError("源数据集中没有 train/val/test 任一 split，无法生成子集数据。")

    small_dataset = DatasetDict(small_dict)
    small_dataset.save_to_disk(str(args.target))

    print(
        f"子集数据集已保存至 {args.target}："
        f"train={len(small_dict.get('train', []))}, "
        f"val={len(small_dict.get('val', []))}, "
        f"test={len(small_dict.get('test', []))}"
    )


if __name__ == "__main__":
    main()


