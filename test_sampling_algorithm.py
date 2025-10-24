#!/usr/bin/env python3
"""
测试改进的智能采样算法
"""

import sys
import os
import time
import numpy as np
import pandas as pd

def create_test_data_with_peaks():
    """创建包含突变的测试数据"""
    print("创建包含突变的测试数据...")
    
    n_points = 100000
    x = np.linspace(0, 100, n_points)
    
    # 基础信号
    y = np.sin(x * 0.1) + np.random.randn(n_points) * 0.1
    
    # 添加一些突变点
    peak_positions = [20000, 40000, 60000, 80000]
    peak_amplitudes = [5, -3, 4, -2]
    
    for pos, amp in zip(peak_positions, peak_amplitudes):
        if pos < n_points:
            # 创建尖锐的突变
            y[pos-10:pos+10] += amp * np.exp(-((np.arange(-10, 10))**2) / 2)
    
    return x, y

def intelligent_sampling(visible_indices, x_data, y_data, max_points):
    """智能采样算法，保留突变细节"""
    if len(visible_indices) <= max_points:
        return visible_indices
        
    # 1. 均匀采样基础点
    step = len(visible_indices) // max_points
    uniform_indices = visible_indices[::step]
    
    # 2. 检测突变点（一阶导数变化大的点）
    if len(visible_indices) > 2:
        # 计算一阶导数
        y_values = y_data[visible_indices]
        dy = np.diff(y_values)
        
        # 计算二阶导数来检测突变
        d2y = np.diff(dy)
        
        # 找到突变点（二阶导数绝对值大的点）
        threshold = np.std(d2y) * 2  # 使用2倍标准差作为阈值
        peak_indices = np.where(np.abs(d2y) > threshold)[0] + 1  # +1因为diff减少了1个元素
        
        # 将突变点添加到采样中
        peak_indices = visible_indices[peak_indices]
        
        # 合并均匀采样点和突变点
        all_indices = np.unique(np.concatenate([uniform_indices, peak_indices]))
        
        # 如果合并后还是太多，进一步均匀采样
        if len(all_indices) > max_points:
            step = len(all_indices) // max_points
            all_indices = all_indices[::step]
            
        return all_indices
    else:
        return uniform_indices

def simple_sampling(visible_indices, max_points):
    """简单均匀采样（用于对比）"""
    if len(visible_indices) <= max_points:
        return visible_indices
    step = len(visible_indices) // max_points
    return visible_indices[::step]

def test_sampling_quality():
    """测试采样质量"""
    print("\n=== 采样质量测试 ===")
    
    # 创建测试数据
    x, y = create_test_data_with_peaks()
    
    # 测试不同缩放级别
    test_ranges = [
        (0, 100),      # 全范围
        (0, 10),       # 10%范围
        (0, 1),        # 1%范围
        (0, 0.1),      # 0.1%范围
    ]
    
    max_points = 1000
    
    for x_min, x_max in test_ranges:
        print(f"\n测试范围: {x_min}-{x_max}")
        
        # 计算可见点
        visible_mask = (x >= x_min) & (x <= x_max)
        visible_indices = np.where(visible_mask)[0]
        visible_points = len(visible_indices)
        
        if visible_points == 0:
            print("  无可见点")
            continue
            
        print(f"  可见点数: {visible_points}")
        
        # 测试简单采样
        start = time.time()
        simple_indices = simple_sampling(visible_indices, max_points)
        simple_time = time.time() - start
        
        # 测试智能采样
        start = time.time()
        intelligent_indices = intelligent_sampling(visible_indices, x, y, max_points)
        intelligent_time = time.time() - start
        
        print(f"  简单采样: {len(simple_indices)} 个点, 耗时 {simple_time:.4f}秒")
        print(f"  智能采样: {len(intelligent_indices)} 个点, 耗时 {intelligent_time:.4f}秒")
        
        # 分析突变点保留情况
        if len(visible_indices) > 0:
            # 找到原始数据中的突变点
            y_visible = y[visible_indices]
            dy = np.diff(y_visible)
            d2y = np.diff(dy)
            threshold = np.std(d2y) * 2
            original_peaks = np.where(np.abs(d2y) > threshold)[0] + 1
            
            # 检查简单采样保留的突变点
            simple_peaks_retained = 0
            for peak_idx in original_peaks:
                if peak_idx in simple_indices:
                    simple_peaks_retained += 1
            
            # 检查智能采样保留的突变点
            intelligent_peaks_retained = 0
            for peak_idx in original_peaks:
                if peak_idx in intelligent_indices:
                    intelligent_peaks_retained += 1
            
            print(f"  原始突变点数: {len(original_peaks)}")
            print(f"  简单采样保留: {simple_peaks_retained}/{len(original_peaks)} ({simple_peaks_retained/len(original_peaks)*100:.1f}%)")
            print(f"  智能采样保留: {intelligent_peaks_retained}/{len(original_peaks)} ({intelligent_peaks_retained/len(original_peaks)*100:.1f}%)")

def test_pyqtgraph_downsample():
    """测试pyqtgraph的downsample效果"""
    print("\n=== PyQtGraph Downsample测试 ===")
    
    # 创建测试数据
    x, y = create_test_data_with_peaks()
    
    print(f"原始数据点数: {len(x)}")
    
    # 模拟pyqtgraph的downsample设置
    # 通常pyqtgraph的downsample能处理到50000-100000个点
    pyqtgraph_threshold = 50000
    
    if len(x) > pyqtgraph_threshold:
        print(f"数据量超过PyQtGraph阈值 ({pyqtgraph_threshold})，需要额外采样")
        # 计算需要的采样率
        downsample_ratio = pyqtgraph_threshold / len(x)
        print(f"PyQtGraph downsample比例: {downsample_ratio:.2%}")
        
        # 模拟downsample后的数据
        step = int(1 / downsample_ratio)
        downsampled_x = x[::step]
        downsampled_y = y[::step]
        
        print(f"Downsample后点数: {len(downsampled_x)}")
        
        # 检查突变点保留情况
        original_peaks = [20000, 40000, 60000, 80000]
        retained_peaks = 0
        for peak in original_peaks:
            if peak < len(x) and peak // step < len(downsampled_x):
                retained_peaks += 1
        
        print(f"突变点保留情况: {retained_peaks}/{len(original_peaks)}")
    else:
        print("数据量在PyQtGraph处理范围内，无需额外采样")

if __name__ == "__main__":
    print("开始采样算法测试...")
    
    # 测试采样质量
    test_sampling_quality()
    
    # 测试pyqtgraph downsample
    test_pyqtgraph_downsample()
    
    print("\n所有测试完成！")