# onnx proto定义
[官方参考文档](https://github.com/onnx/onnx/blob/main/docs/IR.md)

# demo
一个加载 onnx 模型，然后打印模型信息的 demo;

onnx 模型可以从官方提供的获取渠道下载:[onnx/models](https://github.com/onnx/models)

这里以 `VGG16` 模型为例：
```shell
wget https://github.com/onnx/models/raw/main/vision/classification/vgg/model/vgg16-7.onnx
```

[完整代码](./code_1)
## 使用 protoc 编译 onnx.proto
```shell
protoc -I=$SRC_DIR  --cpp_out=$DST_DIR  onnx.proto
```
编译成功之后，在`DST_DIR`目录下生成`onnx.pb.h`,`onnx.pn.cc`文件。
```