# GCC

[参考:GCC编译器30分钟入门教程](http://c.biancheng.net/gcc/)
[参考:动态链接库和静态链接库](http://c.biancheng.net/dll/)
[gcc编译工具常用命令以及汇编语言](https://blog.csdn.net/qq_44644740/article/details/109086520)

C 或者 C++ 程序从源代码生成可执行程序的过程，需经历 4 个过程，分别是`预处理`、`编译`、`汇编`和`链接`。

# 1. 预处理：

主要是处理那些源文件和头文件中以 # 开头的命令。

```shell
gcc -E demo.c -o demo.i
gcc -E -C demo.c -o demo.i  # -C 选项，阻止 GCC 删除源文件和头文件中的注释：
```

# 2. 编译：（编译为汇编代码， Compilation）

经过一系列的词法分析、语法分析、语义分析以及优化，加工为当前机器支持的`汇编代码`。

gcc 指令添加 -S（注意是大写）选项，即可令 GCC 编译器仅将指定文件加工至编译阶段，并生成对应的汇编代码文件。(-I 可以指定的目录)

```shell
gcc -S demo.c -o demo.s
```

# 3. 汇编（Assembly）

对已得到的 dmeo.s 执行汇编操作，并得到相应的目标文件。所谓目标文件，其本质为二进制文件，但由于尚未经过链接操作，所以无法直接运行。

汇编其实就是将汇编代码转换成可以执行的机器指令。大部分汇编语句对应一条机器指令，有的汇编语句对应多条机器指令。相对于编译操作，汇编过程会简单很多，它并没有复杂的语法，也没有语义，也不需要做指令优化，只需要根据汇编语句和机器指令的对照表一一翻译即可。

gcc 指令添加 -c 选项（注意是小写字母 c），即可让 GCC 编译器将指定文件加工至汇编阶段，并生成相应的目标文件

```shell
gcc -c demo.c -o democ.o #加工c语言源码至 .o 文件
gcc -c demo.s -o democ.o #加工汇编代码至 .o 文件
```

# 4. 链接: gcc -l

将多个目标文件链接在一起，生成可执行文件。（库也是一种目标文件）

```shell
gcc democ.o -o democ.exe #GCC 默认会链接 libc.a 或者 libc.so
```

`编译时找不到想链接的库`

通常，GCC 会自动在标准库目录中搜索文件，例如 /usr/lib，如果想链接其它目录中的库，就得特别指明。有三种方式可以链接在 GCC 搜索路径以外的链接库。

1) 把链接库作为一般的目标文件，为 GCC 指定该链接库的完整路径与文件名

```shell
 gcc main.c -o main.out /usr/lib/libm.a
```

2) 使用-L选项，为 GCC 增加另一个搜索链接库的目录：

```shell
gcc main.c -o main.out -L/usr/lib -lm
```

可以使用多个`-L`选项，或者在一个 `-L` 选项内使用冒号分割的路径列表。

3) 把包括所需链接库的目录加到环境变量 `LIBRARYPATH` 中。

```shell
export LIBRARY_PATH = /usr/locl/lib
```

[GCC分步编译C++源程序（汇总版）](http://c.biancheng.net/view/vip_8524.html)

# 5. gcc指令一次处理多个文件

假设一个项目中仅包含 2 个源文件，其中 myfun.c 文件用于存储一些功能函数，以方便直接在 main.c 文件中调用。

```shell
[root@bogon demo]# ls
main.c  myfun.c
[root@bogon demo]# gcc -c myfun.c main.c
[root@bogon demo]# ls
main.c  main.o  myfun.c  myfun.o
[root@bogon demo]# gcc myfun.o main.o -o main.exe
[root@bogon demo]# ls
main.c  main.exe  main.o  myfun.c  myfun.o
```

gcc 指令还可以直接编译并链接它们：

```shell
[root@bogon demo]# gcc myfun.c main.c -o main.exe
[root@bogon demo]# ls
main.c  main.exe  myfun.c
```

用 *.c 表示所有的源文件

```shell
gcc *.c -o main.exe
```

# 6. GCC使用静态链接库和动态链接库
链接过程，总的来说链接阶段要完成的工作，就是将同一项目中各源文件生成的目标文件以及程序中用到的库文件整合为一个可执行文件。

所谓库文件，可以将其等价为压缩包文件，该文件内部通常包含不止一个目标文件（也就是二进制文件）。

事实上，库文件只是一个统称，代指的是一类压缩包，它们都包含有功能实用的目标文件。要知道，虽然库文件用于程序的链接阶段，但编译器提供有 2 种实现链接的方式，分别称为静态链接方式和动态链接方式，其中采用静态链接方式实现链接操作的库文件，称为静态链接库；采用动态链接方式实现链接操作的库文件，称为动态链接库。

在程序运行之前确定符号地址的过程叫做静态链接（Static Linking）；如果需要等到程序运行期间再确定符号地址，就叫做动态链接（Dynamic Linking）。

`静态链接库`

静态链接库实现链接操作的方式很简单，即程序文件中哪里用到了库文件中的功能模块，GCC 编译器就会将该模板代码直接复制到程序文件的适当位置，最终生成可执行文件。

使用静态库文件实现程序的链接操作，既有优势也有劣势：

  优势是，生成的可执行文件不再需要任何静态库文件的支持就可以独立运行（可移植性强）；

  劣势是，如果程序文件中多次调用库中的同一功能模块，则该模块代码势必就会被复制多次，生成的可执行文件中会包含多段完全相同的代码，造成代码的冗余。

和使用动态链接库生成的可执行文件相比，静态链接库生成的可执行文件的体积更大。

`动态链接库`

动态链接库，又称为共享链接库。和静态链接库不同，采用动态链接库实现链接操作时，程序文件中哪里需要库文件的功能模块，GCC 编译器不会直接将该功能模块的代码拷贝到文件中，而是将功能模块的位置信息记录到文件中，直接生成可执行文件。

显然，这样生成的可执行文件是无法独立运行的。采用动态链接库生成的可执行文件运行时，GCC 编译器会将对应的动态链接库一同加载在内存中，由于可执行文件中事先记录了所需功能模块的位置信息，所以在现有动态链接库的支持下，也可以成功运行。

采用动态链接库实现程序的连接操作，其优势和劣势恰好和静态链接库相反：

  优势是，由于可执行文件中记录的是功能模块的地址，真正的实现代码会在程序运行时被载入内存，这意味着，即便功能模块被调用多次，使用的都是同一份实现代码（这也是将动态链接库称为共享链接库的原因）。

  劣势是，此方式生成的可执行文件无法独立运行，必须借助相应的库文件（可移植性差）。

和使用静态链接库生成的可执行文件相比，动态链接库生成的可执行文件的体积更小，因为其内部不会被复制一堆冗余的代码。

值得一提的是，GCC 编译器生成可执行文件时，默认情况下会优先使用动态链接库实现链接操作，除非当前系统环境中没有程序文件所需要的动态链接库，GCC 编译器才会选择相应的静态链接库。如果两种都没有（或者 GCC 编译器未找到），则链接失败。

# 7. 用GCC制作静态链接库

```shell
[root@bogon demo]# ls                       <- demo 目录结构
add.c  div.c  main.c  sub.c  test.h

# 1) 将所有指定的源文件，都编译成相应的目标文件：
[root@bogon demo]# gcc -c sub.c add.c div.c
[root@bogon demo]# ls
add.c  add.o  div.c  div.o  main.c  sub.c  sub.o  test.h

# 2) 然后使用 ar 压缩指令，将生成的目标文件打包成静态链接库，其基本格式如下：
# ar rcs 静态链接库名称 目标文件1 目标文件2 ...

[root@bogon demo]# ar rcs libmymath.a add.o sub.o div.o
[root@bogon demo]# ls
add.c  add.o  div.c  div.o  libmymath.a  main.c  sub.c  sub.o  test.h
```

`静态库的使用`

```shell
root@bogon demo]# gcc -c main.c
[root@bogon demo]# ls
add.c  div.c  libmymath.a  main.o  sub.c
test.h  add.o  div.o  main.c  sub.o

[root@bogon demo]# gcc main.o -static -L /root/demo/ -lmymath
[root@bogon demo]# ls
add.c  a.out  div.o        main.c  sub.c  test.h
add.o  div.c  libmymath.a  main.o  sub.o
```

其中，-static 选项强制 GCC 编译器使用静态链接库。

其中，-L（大写的 L）选项用于向 GCC 编译器指明静态链接库的存储位置（可以借助 pwd 指令查看具体的存储位置）； -l（小写的 L）选项用于指明所需静态链接库的名称，注意这里的名称指的是 xxx 部分，且建议将 -l 和 xxx 直接连用（即 -lxxx），中间不需有空格。

# 8. 用GCC制作动态库

1. 方式一：

`gcc -fpic -shared 源文件名... -o 动态链接库名`

```shell
[root@bogon demo]# ls
add.c  div.c  main.c  sub.c  test.h
[root@bogon demo]# gcc -fpic -shared add.c sub.c div.c -o libmymath.so
[root@bogon demo]# ls
add.c  div.c  libmymath.so  main.c  sub.c  test.h
```

2. 方式二：

`先使用 gcc -c 指令将指定源文件编译为目标文件。仍以 demo 项目中的 add.c、sub.c 和 div.c 为例，先执行如下命令：`

```shell
[root@bogon demo]# ls
add.c  div.c  main.c  sub.c  test.h
[root@bogon demo]# gcc -c -fpic add.c sub.c div.c
[root@bogon demo]# ls
add.c  add.o  div.c  div.o  main.c  sub.c  sub.o  test.h
```

`注意，为了后续生成动态链接库并能正常使用，将源文件编译为目标文件时，也需要使用 -fpic 选项。`

`在此基础上，接下来利用上一步生成的目标文件，生成动态链接库：`

```shell
[root@bogon demo]# gcc -shared add.o sub.o div.o -o libmymath.so
[root@bogon demo]# ls
add.c  add.o  div.c  div.o  libmymath.so  main.c  sub.c  sub.o  test.h
```
以上 2 种操作，生成的动态链接库是完全一样的，读者任选一种即可。


`动态库的使用`

提示找不到库时：执行 `ldd ./mian`命令查看当前文件在执行时需要用到的所有动态链接库，以及各个库文件的存储位置。

`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:xxx`

# 9. 显示调用C/C++动态链接库

总的来讲，动态链接库的调用方式有 2 种，分别是：

  `隐式调用（静态调用）`：将动态链接库和其它源程序文件（或者目标文件）一起参与链接；

  `显式调用（动态调）`：手动调用动态链接库中包含的资源，同时用完后要手动将资源释放。

和隐式调用动态链接库不同，在 C/C++ 程序中显示调用动态链接库时，无需引入和动态链接库相关的头文件。但与此同时，程序中需要引入另一个头文件，即 `<dlfcn.h>` 头文件，
因为要显式调用动态链接库，需要使用该头文件提供的一些函数。

`四个函数` [参考链接：](http://c.biancheng.net/view/vip_8527.html)

`void *dlopen (const char *filename, int flag);`

`void *dlsym(void *handle, char *symbol);`

`int dlclose (void *handle);`

`const char *dlerror(void);`

# 10. GCC找不到头/库文件怎么办？

[参考](http://c.biancheng.net/view/vip_8528.html)

1. gcc 指令中用 -L 选项明确指明其存储路径

2. export LIBRARY_PATH=...  (找到静态链接库的路径)

3. export LD_LIBRARY_PATH=..   (找到动态链接库的路径)

4. export CPLUS_INCLUDE_PATH=... 
   export C_INCLUDE_PATH=...







# 附件
[GCC 使用手册](https://gcc.gnu.org/onlinedocs/gcc-12.2.0/gcc/)

`nonnull`
`nonnull (arg-index, …)`

The nonnull attribute may be applied to a function that takes at least one argument of a pointer type. It indicates that the referenced arguments must be non-null pointers. 

[demo](./code/code_1)