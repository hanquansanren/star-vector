#!/usr/bin/env python
"""
将 HF Hub 下载的快照目录（parquet 文件）转换为可离线加载的 DatasetDict。
"""

import os
import glob
from datasets import load_dataset, DatasetDict

SOURCE_PATH = "/Data_PHD/phd23_weiguang_zhang/project/svg_data/datasets--starvector--svg-stack"
TARGET_PATH = "/Data_PHD/phd23_weiguang_zhang/project/svg_data/svg-stack-hf"


def _resolve_snapshot_data_dir(source_root: str) -> str:
    """在 snapshots/ 下找到最新的快照目录并返回其中的 data 子目录。"""
    snapshot_root = os.path.join(source_root, "snapshots")
    if not os.path.isdir(snapshot_root):
        raise FileNotFoundError(f"未找到 snapshots 目录：{snapshot_root}")

    snapshot_candidates = sorted(
        [
            os.path.join(snapshot_root, d)
            for d in os.listdir(snapshot_root)
            if os.path.isdir(os.path.join(snapshot_root, d))
        ]
    )
    if not snapshot_candidates:
        raise FileNotFoundError(f"在 {snapshot_root} 下未找到任何快照目录")

    latest_snapshot = snapshot_candidates[-1]
    data_dir = os.path.join(latest_snapshot, "data")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"快照中缺少 data 目录：{data_dir}")
    return data_dir


def _collect_parquet_files(data_dir: str):
    """收集 train/test/val 分片 parquet 文件。"""
    splits = ["train", "test", "val"]
    data_files = {}
    for split in splits:
        pattern = os.path.join(data_dir, f"{split}-*.parquet")
        files = sorted(glob.glob(pattern))
        if files:
            data_files[split] = files
    if not data_files:
        raise FileNotFoundError(f"在 {data_dir} 未找到任何 parquet 分片")
    return data_files


def main():
    if not os.path.exists(SOURCE_PATH):
        raise FileNotFoundError(f"源目录不存在：{SOURCE_PATH}")

    data_dir = _resolve_snapshot_data_dir(SOURCE_PATH)
    data_files = _collect_parquet_files(data_dir)

    dataset = load_dataset("parquet", data_files=data_files, split=None)
    dataset = DatasetDict(dataset)

    os.makedirs(TARGET_PATH, exist_ok=True)
    dataset.save_to_disk(TARGET_PATH)
    print(f"保存完成：{TARGET_PATH}")


if __name__ == "__main__":
    main()