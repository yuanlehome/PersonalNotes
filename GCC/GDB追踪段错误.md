用以下命令来阻止系统生成core文件:

ulimit -c 0

用以下命令来让系统生成core文件:
ulimit -c unlimited

下面的命令可以检查生成core文件的选项是否打开:

ulimit -a

该命令将显示所有的用户定制，其中选项-a代表“all”。


生成 core 文件后：

gdb [exec_file] [core_file]
