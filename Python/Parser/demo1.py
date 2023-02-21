# [Python中Parser的超详细用法实例](https://zhuanlan.zhihu.com/p/539331146)
# [argparse --- 命令行选项、参数和子命令解析器](https://docs.python.org/zh-cn/3.7/library/argparse.html?highlight=parse_known_args#module-argparse)


'''
基础用法
'''
# 导入库
import argparse
 
# 1. 定义命令行解析器对象
parser = argparse.ArgumentParser(description='Demo of argparse')
 
# 2. 添加命令行参数
parser.add_argument('--epochs', type=int, default=30, help="set epochs")
parser.add_argument('--batch', type=int, default=4, help="set batch")
 
# 3. 从命令行中结构化解析参数
args = parser.parse_args()
epochs = args.epochs
batch = args.batch
print('show {}  {}'.format(epochs, batch))