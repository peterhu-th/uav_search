import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, 'src'))

from environment import Environment

try:
    env = Environment()
    print("\n[成功] DLL 加载并初始化成功！网格大小：", env.prob_map.shape)
except Exception as e:
    print("\n[失败] ", e)