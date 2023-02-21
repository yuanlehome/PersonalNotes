# Probobuf简介
`protobuf` (protocol buffer) 是谷歌内部的混合语言数据标准。通过将结构化的数据进行序列化(串行化)，用于通讯协议、数据存储等领域和语言无关、平台无关、可扩展的序列化结构数据格式。

[官网文档](https://developers.google.com/protocol-buffers/docs/overview)

我们说的 protobuf 通常包括下面三点:

1. `一种二进制数据交换格式`。可以将 C++ 中定义的存储类的内容与二进制序列串相互转换，主要用于数据传输或保存

2. `定义了一种源文件`，扩展名为 .proto(类比.cpp文件)，使用这种源文件，可以定义存储类的内容

3. `protobuf有自己的编译器protoc`，可以将 .proto 编译成.cc文件，使之成为一个可以在 C++ 工程中直接使用的类

`序列化`：将数据结构或对象转换成二进制串的过程。

`反序列化`：将在序列化过程中所产生的二进制串转换成数据结构或对象的过程。

# 从源码编译/安装 Protobuf 的编译器 protoc
源码编译安装，参考下面文档：

[源码编译参考文档](https://github.com/protocolbuffers/protobuf/blob/master/src/README.md) 

从官方仓库下载源码:
```shell
git clone https://github.com/protocolbuffers/protobuf.git
git checkout v3.21.5
```

编译大致过程如下：
```shell
./autogen.sh
./configure  #--prefix=
mkdir build && cd build
cmake .. -Dprotobuf_BUILD_TESTS=OFF #-DBUILD_SHARED_LIBS=ON
make -j12
make install #DESTDIR=xxx
```

# 定义 proto 文件
[参考教程](https://developers.google.com/protocol-buffers/docs/cpptutorial)

## message 介绍
`message`：`protobuf`中定义一个消息类型是通过关键字`message`字段指定的，这个关键字类似于C++/Java中的`class`关键字。

使用 `protobuf` 编译器将`proto`编译成 C++ 代码之后，每个`message`都会生成一个名字与之对应的 C++ 类，该类公开继承自`google::protobuf::Message`。


## 定义一个 message
创建tutorial.person.proto文件，文件内容如下：

```proto
// FileName: tutorial.person.proto 
// 通常文件名建议命名格式为 包名.消息名.proto 

// 表示正在使用proto2命令
syntax = "proto2"; 

//包声明，tutorial 也可以声明为二级类型。例如a.b，表示a类别下b子类别
package tutorial; 

//编译器将生成一个名为person的类
//类的字段信息包括姓名name,编号id,邮箱email，以及电话号码phones
message Person { 

  required string name = 1;
  required int32 id = 2;  
  optional string email = 3;

  enum PhoneType {  //电话类型枚举值 
    MOBILE = 0;  //手机号  
    HOME = 1;    //家庭联系电话
    WORK = 2;    //工作联系电话
  } 
  
  //电话号码phone消息体
  //组成包括号码number、电话类型 type
  message PhoneNumber {
    required string number = 1;    
    optional PhoneType type =  2 [default = HOME];
  }  
  
  repeated PhoneNumber phones = 4;
} 

// 通讯录消息体，包括一个Person类的people
message AddressBook { 
  repeated Person people = 1; 

}
```

proto 文件以`package`声明开头，这有助于防止不同项目之间命名冲突。

在C++中，以`package`声明的文件内容生成的类将放在与包名匹配的`namespace`中，上面的.proto文件中所有的声明都属于`tutorial`。

其他细节建议阅读官方文档。

# 编译 proto 文件

```shell
protoc -I=$SRC_DIR  --cpp_out=$DST_DIR  addressbook.proto
# $SRC_DIR 所在的源目录
# --cpp_out 生成C++代码
# $DST_DIR 生成代码的目标目录
# xxx.proto:要针对 $SRC_DIR 目录下哪个 proto 文件生成接口，在这里对应 tutorial.person.proto
```

# demo：构建一个通信录
下面是一个 demo， 构建一个通信录，往通信录中增加一个人员，然后将构建好的通信录信息“序列化”到文件，然后再从文件中“反序列化”，读取通信录中的信息。

完整代码查看 [cpp_demo](./cpp_demo)

## Writing A Message
先构建一个通信录，往里面录入人员信息：

主要代码如下；
```cpp
  // 构建一个 address_book
  tutorial::AddressBook my_address_book;

  // 往通信录中加入人员信息
  tutorial::Person* people = my_address_book.add_people();

  people->set_name("xiaoming");
  people->set_id(9827);
  people->set_email("123@qq.com");
  tutorial::Person_PhoneNumber* phone_number = people->add_phones();

  phone_number->set_number("1234567890");
  phone_number->set_type(tutorial::Person_PhoneType::Person_PhoneType_MOBILE);
```

## Parsing and Serialization
将构建好的通信录序列化到磁盘中（文件中）：

[序列化 API](https://developers.google.com/protocol-buffers/docs/cpptutorial#parsing-and-serialization)

对于 `proto` 中定义的任何 `message`，都可以进行序列化和反序列化操作。

主要代码如下：
```cpp
  // 将构建好的通信录序列化保存在磁盘中
  std::ofstream outFile("ADDRESS_BOOK", std::ios::out | std::ios::binary);
  my_address_book.SerializeToOstream(&outFile);
  outFile.close();
```

## Reading A Message
将磁盘中的文件反序列化到 address_book 中并读取通信录中的人员信息

主要代码如下；
```cpp
  //将磁盘中的文件反序列化到 address_book 中
  tutorial::AddressBook address_book;
  std::ifstream inFile("ADDRESS_BOOK", std::ios::in | std::ios::binary);
  address_book.ParseFromIstream(&inFile);
  inFile.close();

  // 读取通信录中的人员信息
  auto &person = address_book.people(0);
  std::cout << person.name() << std::endl;
  std::cout << person.id() << std::endl;
  std::cout << person.email() << std::endl;
  
  auto &number = person.phones(0);
  std::cout << number.type() << std::endl;
  std::cout << number.number() << std::endl;
```

编译指令如下：
```shell
g++ main.cc addressbook.pb.cc -static -lprotobuf #使用静态库编译可执行文件，提高代码可移植性
```
执行成功后，目录下生成一个文件：`ADDRESS_BOOK`。

# Python 使用 protobuf
[python 安装文档](https://github.com/protocolbuffers/protobuf/tree/main/python)


编译过程需要`libprotobuf.so`, cmake 的过程中需要额外指定 `-DBUILD_SHARED_LIBS=ON`。

安装成功后，默认安装到路径：`/usr/local/lib/python3.7/dist-packages/protobuf-4.21.5-py3.7-linux-x86_64.egg`

完整代码查看 [python_demo](./python_demo), 参考[官方 python demo](https://developers.google.com/protocol-buffers/docs/pythontutorial)

编译 `addressbook..proto`的命令为：
```shell
protoc -I=$SRC_DIR --python_out=$DST_DIR $SRC_DIR/addressbook.proto
```

编译成功后在 `$DST_DIR`目录下生成 `addressbook_pb2.py`文件。

# 总结
protobuf提供一个自定义储存格式的工具。通过.proto文件定义数据存储方式，然后利用protoc工具生成读写定义数据的接口！