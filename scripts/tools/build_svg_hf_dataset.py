import argparse
import random
from pathlib import Path
from typing import Dict, List

from datasets import Dataset, DatasetDict


def _svg_records_from_paths(svg_files: List[Path], root_for_rel: Path) -> List[Dict[str, str]]:
    """
    从一组 svg 路径中构建 [{Filename, Svg}, ...] 记录。
    Filename 使用相对于 root_for_rel 的相对路径。
    """
    records: List[Dict[str, str]] = []
    for svg_path in svg_files:
        try:
            svg_str = svg_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            svg_str = svg_path.read_text(errors="ignore")

        rel_name = svg_path.relative_to(root_for_rel).as_posix()
        records.append({"Filename": rel_name, "Svg": svg_str})
    return records


def build_split(split_dir: Path) -> Dataset:
    """
    从某个 split 目录（例如 train/ 或 val/）中读取所有 .svg 文件，
    构建一个包含两列的 HuggingFace Dataset：
      - Filename: 相对于该 split 目录的相对路径（字符串）
      - Svg:      SVG 的字符串内容
    """
    svg_files = sorted(split_dir.rglob("*.svg"))
    if not svg_files:
        raise ValueError(f"在目录 {split_dir} 下没有找到任何 .svg 文件")

    records = _svg_records_from_paths(svg_files, split_dir)
    return Dataset.from_list(records)


def build_dataset_dict_pre_split(input_root: Path) -> DatasetDict:
    """
    从形如：
        input_root/train/**/*.svg
        input_root/val/**/*.svg
        input_root/test/**/*.svg
    的目录结构中构建 DatasetDict。

    至少需要存在一个 split 子目录（train/val/test 之一）。
    """
    splits: Dict[str, Dataset] = {}
    for split in ["train", "val", "test"]:
        split_dir = input_root / split
        if split_dir.is_dir():
            print(f"发现预先划分的 split: {split_dir}")
            splits[split] = build_split(split_dir)

    if not splits:
        raise ValueError(
            f"在 {input_root} 下没有找到 train/、val/ 或 test/ 目录，"
            "至少需要存在一个子目录来构建数据集（pre_split 模式）。"
        )

    if len(splits) == 1:
        # 只存在一个 split 时，也可以直接保存为单个 Dataset；
        # 但为了和当前 svg-stack-hf-1-8 格式一致，这里仍然返回 DatasetDict。
        print(f"仅发现单个 split: {list(splits.keys())[0]}，仍然包装为 DatasetDict")

    return DatasetDict(splits)


def build_dataset_dict_random_split(
    input_root: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> DatasetDict:
    """
    从一个总目录（例如 /.../zhuan_output）中收集所有 .svg 文件，
    然后按给定比例随机划分为 train/val/test。

    目录结构示例：
        input_root/font1/*.svg
        input_root/font2/*.svg
        ...

    png 文件会被自动忽略（只匹配 *.svg）。
    """
    all_svg_files = sorted(input_root.rglob("*.svg"))
    if not all_svg_files:
        raise ValueError(f"在 {input_root} 下没有找到任何 .svg 文件")

    total = len(all_svg_files)
    print(f"共发现 {total} 个 SVG 文件，将按比例 "
          f"{train_ratio*100:.1f}% / {val_ratio*100:.1f}% / {test_ratio*100:.1f}% "
          "划分为 train/val/test")

    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio 必须等于 1.0")

    rng = random.Random(seed)
    rng.shuffle(all_svg_files)

    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    n_test = total - n_train - n_val

    train_files = all_svg_files[:n_train]
    val_files = all_svg_files[n_train : n_train + n_val]
    test_files = all_svg_files[n_train + n_val :]

    print(f"实际样本数: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")

    train_records = _svg_records_from_paths(train_files, input_root)
    val_records = _svg_records_from_paths(val_files, input_root)
    test_records = _svg_records_from_paths(test_files, input_root)

    datasets: Dict[str, Dataset] = {}
    if train_records:
        datasets["train"] = Dataset.from_list(train_records)
    if val_records:
        datasets["val"] = Dataset.from_list(val_records)
    if test_records:
        datasets["test"] = Dataset.from_list(test_records)

    return DatasetDict(datasets)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "将离散的 SVG 文件打包为与 svg-stack-hf-1-8 相同 schema 的 "
            "HuggingFace Dataset 格式。支持：\n"
            "1) pre_split 模式：输入目录下已有 train/val/test 子目录；\n"
            "2) random_split 模式：输入目录下只有若干子目录（如 9 个字体），"
            "   从所有 *.svg 中按比例随机划分 train/val/test。"
        )
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["pre_split", "random_split"],
        default="pre_split",
        help=(
            "数据划分模式：\n"
            "  - pre_split: 期望 input_root 下已有 train/val/test 子目录；\n"
            "  - random_split: 从 input_root 下所有 *.svg 中按比例随机划分 train/val/test。"
        ),
    )
    parser.add_argument(
        "--input_root",
        type=str,
        required=True,
        help=(
            "根目录：\n"
            "  - pre_split 模式：包含 train/、val/、test 子目录，每个子目录中存放对应 split 的 .svg 文件；\n"
            "  - random_split 模式：例如 /.../zhuan_output，每个子目录是一个字体，内部是若干 .svg/.png 文件。"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help=(
            "HuggingFace Dataset 的输出路径，生成的目录结构会类似：\n"
            "/Data_PHD/phd23_weiguang_zhang/project/svg_data/svg-stack-hf-1-8"
        ),
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.98,
        help="random_split 模式下训练集比例，默认 0.98。",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.018,
        help="random_split 模式下验证集比例，默认 0.018。",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.002,
        help="random_split 模式下测试集比例，默认 0.002。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random_split 模式下用于随机划分的随机种子，默认 42。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not input_root.exists():
        raise FileNotFoundError(f"输入根目录不存在: {input_root}")

    if args.mode == "pre_split":
        dataset_dict = build_dataset_dict_pre_split(input_root)
    else:
        dataset_dict = build_dataset_dict_random_split(
            input_root=input_root,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
    print(dataset_dict)

    # 保存到磁盘，得到与 svg-stack-hf-1-8 一样的 .arrow / dataset_dict.json 结构
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_dir))
    print(f"数据集已保存到: {output_dir}")


if __name__ == "__main__":
    main()


