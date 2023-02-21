[python中的lambda函数](https://blog.csdn.net/weixin_48077303/article/details/115432398)

定义：

    <函数名> = Lambda <参数> : <表达式>

冒号后面的表达式的计算结果即为该lambda函数的返回值

等价于：
```python
    def <函数名>(<参数>):
        <函数体>
        return <返回值>
```

```python
g=lambda x,y,z:x+y+z*2
print(g(1,2,3))
#结果为9

#也可直接传递参数
(lambda x:x**2)(3)
#结果为9
```

lambda表达式会返回一个函数对象，如果没有变量接受这个返回值的话，它很快就会被丢弃。也正是由于lambda只是一个表达式，所以它可以直接作为list和dict的成员

```python
list_a = [lambda a: a**3, lambda b: b**3]
print(type(list_a[0]))

# 结果为： <class 'function'>

# 与map函数进行使用
a = map(lambda x:x**2,range(5))
print(list(a))

# [0, 1, 4, 9, 16]
```
