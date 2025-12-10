#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
重训练示例脚本

此脚本演示如何：
1. 从本地磁盘读取 CSV 文件
2. 上传到 Flask 后端的 /api/upload-and-retrain 端点
3. 触发模型重训练
4. 获取新的模型性能指标和统计信息

使用方法:
    python examples/retrain_with_new_data.py <csv_file_path> [--host HOST] [--port PORT]

示例:
    python examples/retrain_with_new_data.py ./data/Wednesday-workingHours.pcap_ISCX.csv
    python examples/retrain_with_new_data.py ./data/custom_attack_data.csv --host 127.0.0.1 --port 5000
"""

import requests
import argparse
import sys
from pathlib import Path


def retrain_model(csv_file, host='127.0.0.1', port=5000):
    """
    向后端发送 CSV 文件并触发重训练
    
    参数:
        csv_file (str): CSV 文件的本地路径
        host (str): Flask 服务器主机
        port (int): Flask 服务器端口
    
    返回:
        dict: 响应数据
    """
    csv_path = Path(csv_file)
    
    # 验证文件是否存在
    if not csv_path.exists():
        print(f"❌ 错误：文件不存在 {csv_file}")
        return None
    
    if not csv_file.endswith('.csv'):
        print(f"❌ 错误：文件必须是 CSV 格式 {csv_file}")
        return None
    
    # 构建 API URL
    url = f'http://{host}:{port}/api/upload-and-retrain'
    
    print(f"📤 正在上传文件: {csv_file}")
    print(f"🔗 API 端点: {url}")
    
    try:
        # 打开文件并发送 POST 请求
        with open(csv_file, 'rb') as f:
            files = {'files': f}
            response = requests.post(url, files=files, timeout=300)  # 300 秒超时
        
        # 处理响应
        if response.status_code == 200:
            data = response.json()
            print(f"\n✅ 重训练成功！")
            print(f"状态: {data.get('status')}")
            print(f"消息: {data.get('message')}")
            
            # 显示统计信息
            if 'stats' in data:
                stats = data['stats']
                print(f"\n📊 数据统计:")
                print(f"  • 总样本数: {stats.get('total_samples', 'N/A')}")
                print(f"  • 唯一标签数: {len(stats.get('unique_labels', []))}")
                print(f"  • 新增标签数: {stats.get('new_labels_count', 0)}")
                if stats.get('new_labels'):
                    print(f"  • 新增标签: {', '.join(stats['new_labels'])}")
                
                print(f"\n📋 标签分布:")
                for label, count in stats.get('label_distribution', {}).items():
                    print(f"  • {label}: {count}")
            
            # 显示性能指标
            if 'performance' in data:
                perf = data['performance']
                print(f"\n📈 模型性能指标:")
                print(f"  • 准确率 (Accuracy): {perf.get('accuracy', 'N/A'):.4f}")
                print(f"  • 精确度 (Precision): {perf.get('precision', 'N/A'):.4f}")
                print(f"  • 召回率 (Recall): {perf.get('recall', 'N/A'):.4f}")
                print(f"  • F1-Score: {perf.get('f1_score', 'N/A'):.4f}")
            
            return data
        
        else:
            print(f"\n❌ 请求失败，状态码: {response.status_code}")
            try:
                error_data = response.json()
                print(f"错误信息: {error_data.get('message', 'Unknown error')}")
                if 'details' in error_data:
                    print(f"详情: {error_data['details']}")
            except:
                print(f"响应: {response.text}")
            return None
    
    except requests.exceptions.ConnectionError:
        print(f"\n❌ 连接错误：无法连接到 {url}")
        print(f"   请确保 Flask 服务器正在运行 (python app.py)")
        return None
    
    except requests.exceptions.Timeout:
        print(f"\n❌ 超时：请求耗时过长（300秒）")
        print(f"   CSV 文件可能过大，请尝试使用更小的文件")
        return None
    
    except Exception as e:
        print(f"\n❌ 发生错误: {str(e)}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='上传 CSV 文件并重训练 DDoS 检测模型',
        epilog='示例: python retrain_with_new_data.py ./data/custom_data.csv --host 127.0.0.1 --port 5000'
    )
    
    parser.add_argument(
        'csv_file',
        help='CSV 文件路径（必须包含 Label 列）'
    )
    
    parser.add_argument(
        '--host',
        default='127.0.0.1',
        help='Flask 服务器主机 (默认: 127.0.0.1)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Flask 服务器端口 (默认: 5000)'
    )
    
    args = parser.parse_args()
    
    # 调用重训练函数
    result = retrain_model(args.csv_file, args.host, args.port)
    
    # 返回退出码
    sys.exit(0 if result else 1)


if __name__ == '__main__':
    main()
