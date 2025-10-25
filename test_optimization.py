#!/usr/bin/env python3
"""
性能优化测试脚本
用于验证内存和性能优化的效果
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

def test_memory_usage():
    """测试内存使用情况"""
    print("\n=== 内存使用测试 ===")
    
    try:
        from test_pyqt6_v5 import FastDataLoader
        
        # 创建测试数据
        test_file = create_test_data(10)  # 10MB测试数据
        
        # 测试加载
        print("开始加载数据...")
        start_time = time.time()
        
        loader = FastDataLoader(test_file)
        
        load_time = time.time() - start_time
        print(f"加载完成，耗时: {load_time:.2f}秒")
        print(f"数据行数: {loader.datalength}")
        print(f"列数: {loader.column_count}")
        
        # 清理
        del loader
        import gc
        gc.collect()
        
        # 删除测试文件
        os.remove(test_file)
        print("测试完成")
        
    except Exception as e:
        print(f"测试失败: {e}")

def test_plot_performance():
    """测试绘图性能"""
    print("\n=== 绘图性能测试 ===")
    
    try:
        from test_pyqt6_v5 import DraggableGraphicsLayoutWidget
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import Qt
        
        # 创建Qt应用
        app = QApplication([])
        
        # 创建绘图组件
        plot_widget = DraggableGraphicsLayoutWidget()
        plot_widget.setup_plot_area()
        
        # 创建大量测试数据
        print("创建测试数据...")
        n_points = 100000  # 10万个点
        x_data = np.linspace(0, 100, n_points)
        y_data = np.sin(x_data) + np.random.randn(n_points) * 0.1
        
        # 测试绘图性能
        print("测试绘图性能...")
        start_time = time.time()
        
        # 模拟绘图
        plot_widget.original_index_x = np.arange(len(x_data))
        plot_widget.original_y = y_data
        plot_widget.offset = 0
        plot_widget.factor = 1
        
        # 测试不同缩放级别下的性能
        test_ranges = [
            (0, 100),      # 全范围
            (0, 10),       # 10%范围
            (0, 1),        # 1%范围
            (0, 0.1),      # 0.1%范围
        ]
        
        for x_min, x_max in test_ranges:
            start = time.time()
            # 模拟缩放操作
            plot_widget.view_box.setXRange(x_min, x_max)
            # 触发样式更新
            plot_widget.update_plot_style(plot_widget.view_box, [(x_min, x_max), (0, 1)])
            end = time.time()
            
            visible_points = np.sum((x_data >= x_min) & (x_data <= x_max))
            print(f"缩放范围 {x_min}-{x_max}: {visible_points} 个可见点, 耗时 {end-start:.4f}秒")
        
        print("绘图性能测试完成")
        
    except Exception as e:
        print(f"绘图性能测试失败: {e}")

if __name__ == "__main__":
    print("开始性能优化测试...")
    
    # 测试内存使用
    test_memory_usage()
    
    # 测试绘图性能
    test_plot_performance()
    
    print("\n所有测试完成！")