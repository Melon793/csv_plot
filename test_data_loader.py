#!/usr/bin/env python3
"""
独立的数据加载器测试（不依赖PyQt6）
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import gc

def create_test_data(size_mb=50):
    """创建测试数据文件"""
    print(f"创建 {size_mb}MB 的测试数据...")
    
    # 计算行数（假设每行约100字节）
    rows = int(size_mb * 1024 * 1024 / 100)
    
    # 创建测试数据
    data = {
        'time': np.arange(rows),
        'signal1': np.random.randn(rows),
        'signal2': np.random.randn(rows) * 0.5,
        'signal3': np.sin(np.arange(rows) * 0.01) + np.random.randn(rows) * 0.1,
    }
    
    df = pd.DataFrame(data)
    
    # 保存为CSV
    filename = f"test_data_{size_mb}mb.csv"
    df.to_csv(filename, index=False)
    
    print(f"测试数据已保存为: {filename}")
    print(f"实际文件大小: {os.path.getsize(filename) / (1024*1024):.1f}MB")
    print(f"数据行数: {len(df)}")
    
    return filename

def test_memory_usage():
    """测试内存使用情况"""
    print("\n=== 内存使用测试 ===")
    
    try:
        import psutil
        process = psutil.Process()
        
        def get_memory_mb():
            memory_info = process.memory_info()
            return memory_info.rss / (1024 * 1024)
        
        print(f"初始内存使用: {get_memory_mb():.1f}MB")
        
        # 创建测试数据
        test_file = create_test_data(10)  # 10MB测试数据
        
        # 测试加载
        print("开始加载数据...")
        start_time = time.time()
        initial_memory = get_memory_mb()
        
        # 模拟数据加载过程
        df = pd.read_csv(test_file)
        
        load_time = time.time() - start_time
        after_load_memory = get_memory_mb()
        
        print(f"加载完成，耗时: {load_time:.2f}秒")
        print(f"数据行数: {len(df)}")
        print(f"列数: {len(df.columns)}")
        print(f"内存使用: {initial_memory:.1f}MB -> {after_load_memory:.1f}MB (+{after_load_memory-initial_memory:.1f}MB)")
        
        # 测试内存清理
        print("\n测试内存清理...")
        del df
        gc.collect()
        
        after_cleanup_memory = get_memory_mb()
        print(f"清理后内存: {after_cleanup_memory:.1f}MB (释放了 {after_load_memory-after_cleanup_memory:.1f}MB)")
        
        # 删除测试文件
        os.remove(test_file)
        print("测试完成")
        
    except ImportError:
        print("psutil不可用，跳过内存监控")
    except Exception as e:
        print(f"测试失败: {e}")

def test_data_sampling():
    """测试数据采样性能"""
    print("\n=== 数据采样性能测试 ===")
    
    # 创建大量测试数据
    print("创建测试数据...")
    n_points = 100000  # 10万个点
    x_data = np.linspace(0, 100, n_points)
    y_data = np.sin(x_data) + np.random.randn(n_points) * 0.1
    
    print(f"原始数据点数: {n_points}")
    
    # 测试不同缩放级别下的性能
    test_ranges = [
        (0, 100),      # 全范围
        (0, 10),       # 10%范围
        (0, 1),        # 1%范围
        (0, 0.1),      # 0.1%范围
    ]
    
    for x_min, x_max in test_ranges:
        start = time.time()
        
        # 计算可见点
        visible_mask = (x_data >= x_min) & (x_data <= x_max)
        visible_points = np.sum(visible_mask)
        
        # 模拟采样
        if visible_points > 10000:
            visible_indices = np.where(visible_mask)[0]
            step = len(visible_indices) // 10000
            sampled_indices = visible_indices[::step]
            sampled_points = len(sampled_indices)
            sampling_ratio = sampled_points / visible_points
        else:
            sampled_points = visible_points
            sampling_ratio = 1.0
        
        end = time.time()
        
        print(f"缩放范围 {x_min}-{x_max}: {visible_points} -> {sampled_points} 个点 (采样率: {sampling_ratio:.2%}), 耗时 {end-start:.4f}秒")
    
    print("数据采样性能测试完成")

def test_chunk_loading():
    """测试分块加载性能"""
    print("\n=== 分块加载性能测试 ===")
    
    # 创建测试数据
    test_file = create_test_data(5)  # 5MB测试数据
    
    try:
        # 测试不同chunk size的性能
        chunk_sizes = [100, 500, 1000, 2000]
        
        for chunk_size in chunk_sizes:
            print(f"\n测试chunk size: {chunk_size}")
            start_time = time.time()
            
            chunks = []
            for chunk in pd.read_csv(test_file, chunksize=chunk_size):
                chunks.append(chunk)
            
            df = pd.concat(chunks, ignore_index=True)
            load_time = time.time() - start_time
            
            print(f"加载完成，耗时: {load_time:.2f}秒")
            print(f"数据行数: {len(df)}")
            
            # 清理
            del df, chunks
            gc.collect()
    
    except Exception as e:
        print(f"分块加载测试失败: {e}")
    finally:
        # 删除测试文件
        if os.path.exists(test_file):
            os.remove(test_file)
    
    print("分块加载性能测试完成")

if __name__ == "__main__":
    print("开始数据加载器性能测试...")
    
    # 测试内存使用
    test_memory_usage()
    
    # 测试数据采样
    test_data_sampling()
    
    # 测试分块加载
    test_chunk_loading()
    
    print("\n所有测试完成！")