#!/usr/bin/env python3
"""
启动新的 6G MIMO 节点式仿真器
只依赖 Tkinter (Python 内置)，无需额外安装！
"""
import sys
import os

# 添加当前路径到模块搜索路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modular_ui.node_simulator import main

if __name__ == "__main__":
    main()
