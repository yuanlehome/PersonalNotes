[显卡，显卡驱动,nvcc, cuda driver,cudatoolkit,cudnn区别](https://blog.csdn.net/ahelloyou/article/details/114092789)
[理清GPU、CUDA、CUDA Toolkit、cuDNN关系以及下载安装](https://blog.csdn.net/qq_42406643/article/details/109545766)

[【亲测有效】Linux下安装cuda（Ubuntu、CentOS等）](https://blog.csdn.net/weixin_56119039/article/details/125828575)

[cuda下载官网: cuda-toolkit-archive](https://developer.nvidia.com/cuda-toolkit-archive)

[NVIDIA CUDA Toolkit Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#abstract)

安装官网流程操作即可;

[cudnn下载官网: cudnn-archive](https://developer.nvidia.com/rdp/cudnn-archive)

# cuda多版本切换
查看环境变量 PATH， PATH 中 cuda 的环境变量一般为 /usr/local/cuda

/usr/local/cuda 是一个软链接，修改这个软链接就可以切换使用的 cuda 版本。

```bash
# 删除之前的软连接
rm -rf /usr/local/cuda
# 建立新的软连接
ln -s /usr/local/cuda-11.2 /usr/local/cuda
# 查看 nvcc 版本
nvcc --version
```


