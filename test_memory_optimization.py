#!/usr/bin/env python3
"""
内存优化测试脚本（无GUI）
"""

import sys
import os
import time
import numpy as np
import pandas as pd

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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

def test_memory_optimization():
    """测试内存优化效果"""
    print("\n=== 内存优化测试 ===")
    
    try:
        from test_pyqt6_v5 import FastDataLoader
        
        # 创建测试数据
        test_file = create_test_data(20)  # 20MB测试数据
        
        print("\n测试1: 首次加载")
        start_time = time.time()
        loader1 = FastDataLoader(test_file)
        load_time1 = time.time() - start_time
        print(f"首次加载完成，耗时: {load_time1:.2f}秒")
        print(f"数据行数: {loader1.datalength}")
        print(f"列数: {loader1.column_count}")
        
        # 模拟内存清理
        print("\n测试2: 清理旧数据后重新加载")
        del loader1
        import gc
        gc.collect()
        
        start_time = time.time()
        loader2 = FastDataLoader(test_file)
        load_time2 = time.time() - start_time
        print(f"重新加载完成，耗时: {load_time2:.2f}秒")
        print(f"数据行数: {loader2.datalength}")
        
        # 测试数据采样功能
        print("\n测试3: 数据采样性能")
        x_data = np.linspace(0, 100, 100000)  # 10万个点
        y_data = np.sin(x_data) + np.random.randn(100000) * 0.1
        
        # 模拟不同缩放级别
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
            else:
                sampled_points = visible_points
            
            end = time.time()
            print(f"缩放范围 {x_min}-{x_max}: {visible_points} -> {sampled_points} 个点, 耗时 {end-start:.4f}秒")
        
        # 清理
        del loader2
        gc.collect()
        
        # 删除测试文件
        os.remove(test_file)
        print("\n内存优化测试完成")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_error_handling():
    """测试异常处理"""
    print("\n=== 异常处理测试 ===")
    
    try:
        from test_pyqt6_v5 import FastDataLoader
        
        # 测试不存在的文件
        print("测试1: 不存在的文件")
        try:
            loader = FastDataLoader("nonexistent_file.csv")
        except Exception as e:
            print(f"正确处理了文件不存在错误: {type(e).__name__}")
        
        # 测试空文件
        print("\n测试2: 空文件")
        empty_file = "empty_test.csv"
        with open(empty_file, 'w') as f:
            f.write("")
        
        try:
            loader = FastDataLoader(empty_file)
        except Exception as e:
            print(f"正确处理了空文件错误: {type(e).__name__}")
        finally:
            if os.path.exists(empty_file):
                os.remove(empty_file)
        
        # 测试损坏的文件
        print("\n测试3: 损坏的CSV文件")
        corrupt_file = "corrupt_test.csv"
        with open(corrupt_file, 'w') as f:
            f.write("col1,col2,col3\n")
            f.write("1,2,3\n")
            f.write("invalid,data,here\n")
            f.write("4,5,6\n")
        
        try:
            loader = FastDataLoader(corrupt_file)
            print("成功处理了损坏的CSV文件")
            del loader
        except Exception as e:
            print(f"正确处理了损坏文件错误: {type(e).__name__}")
        finally:
            if os.path.exists(corrupt_file):
                os.remove(corrupt_file)
        
        print("异常处理测试完成")
        
    except Exception as e:
        print(f"异常处理测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("开始内存优化和异常处理测试...")
    
    # 测试内存优化
    test_memory_optimization()
    
    # 测试异常处理
    test_error_handling()
    
    print("\n所有测试完成！")