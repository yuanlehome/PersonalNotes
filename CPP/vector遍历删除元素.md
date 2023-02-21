关于这个话题，网上有太多的版本了。

以后，以这一篇的版本为准：

```cpp
void del_vec_foreach(vector<int>& vec,int target) {
	for (vector<int>::iterator it = vec.begin(); it != vec.end();) {
		if (*it == target) {
			it = vec.erase(it);
		}
		else {
			++it;
		}
	}
}
————————————————
版权声明：本文为CSDN博主「看，未来」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/qq_43762191/article/details/119332596
```