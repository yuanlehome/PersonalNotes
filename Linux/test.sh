# activate debugging from here
set -ex
# 中间脚本逻辑
export HELLO=hello
echo $HELLO
unset HELLO
echo $HELLO
# stop debugging from here
set +x
# set -u
echo $HELLO
echo "last"