#!/usr/bin/env python3
"""
将 svg-stack-hf 数据集缩减到原来的 1/8 并保存到新目录
"""
import os
from datasets import load_from_disk
import argparse

def reduce_dataset(input_dir, output_dir, reduction_factor=8):
    """
    将数据集缩减到原来的 1/reduction_factor
    
    Args:
        input_dir: 输入数据集目录
        output_dir: 输出数据集目录
        reduction_factor: 缩减因子，默认8（即缩减到1/8）
    """
    print(f"正在加载数据集: {input_dir}")
    dataset = load_from_disk(input_dir)
    
    print(f"原始数据集大小:")
    for split in dataset.keys():
        print(f"  {split}: {len(dataset[split])}")
    
    # 创建缩减后的数据集
    reduced_dataset = {}
    
    for split in dataset.keys():
        original_size = len(dataset[split])
        new_size = original_size // reduction_factor
        
        print(f"\n正在缩减 {split} 分割:")
        print(f"  原始大小: {original_size}")
        print(f"  新大小: {new_size} (缩减到 1/{reduction_factor})")
        
        # 使用 select 方法选择前 new_size 个样本
        # 为了保持随机性，我们可以先打乱再选择，但为了可复现性，这里直接选择前 N 个
        reduced_dataset[split] = dataset[split].select(range(new_size))
        print(f"  完成！新数据集大小: {len(reduced_dataset[split])}")
    
    # 保存缩减后的数据集
    print(f"\n正在保存缩减后的数据集到: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    from datasets import DatasetDict
    reduced_dataset_dict = DatasetDict(reduced_dataset)
    reduced_dataset_dict.save_to_disk(output_dir)
    
    print(f"\n数据集已成功保存到: {output_dir}")
    print(f"\n缩减后的数据集大小:")
    for split in reduced_dataset.keys():
        print(f"  {split}: {len(reduced_dataset[split])}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="缩减数据集到原来的 1/8")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/Data_PHD/phd23_weiguang_zhang/project/svg_data/svg-stack-hf",
        help="输入数据集目录"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/Data_PHD/phd23_weiguang_zhang/project/svg_data/svg-stack-hf-1-8",
        help="输出数据集目录"
    )
    parser.add_argument(
        "--reduction_factor",
        type=int,
        default=8,
        help="缩减因子，默认8（即缩减到1/8）"
    )
    
    args = parser.parse_args()
    
    reduce_dataset(args.input_dir, args.output_dir, args.reduction_factor)

