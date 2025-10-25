#!/usr/bin/env python3
"""
测试重载功能的简单脚本
"""
import sys
import os
import numpy as np
import pandas as pd

# 创建测试数据
def create_test_data(filename, rows=1000):
    """创建测试CSV文件"""
    data = {
        'time': np.arange(rows),
        'signal1': np.sin(np.linspace(0, 4*np.pi, rows)) + np.random.normal(0, 0.1, rows),
        'signal2': np.cos(np.linspace(0, 4*np.pi, rows)) + np.random.normal(0, 0.1, rows),
        'signal3': np.random.normal(0, 1, rows)
    }
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"创建测试文件: {filename}, 行数: {rows}")

def main():
    # 创建两个测试文件
    create_test_data("test_data1.csv", 1000)
    create_test_data("test_data2.csv", 1500)
    
    print("测试文件已创建:")
    print("- test_data1.csv (1000行)")
    print("- test_data2.csv (1500行)")
    print("\n请在PyQt6应用中测试重载功能:")
    print("1. 加载 test_data1.csv")
    print("2. 绘制一些变量")
    print("3. 重载数据")
    print("4. 检查plot是否正确重新绘制")

if __name__ == "__main__":
    main()