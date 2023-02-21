nvcc  -c ../vector_add_op.cu 
g++ -c  ../memory_pool.cc -I /usr/local/cuda-11.2/include -L /usr/local/cuda-11.2/lib64 -lcudart
g++ -c  ../tensor_util.cc -I /usr/local/cuda-11.2/include -L /usr/local/cuda-11.2/lib64 -lcudart

if [[ $1 == "LOAD_WEIGHT_ON_RUNTIME" ]]
then
  echo "compile with maro LOAD_WEIGHT_ON_RUNTIME"
  g++ test_predictor.cc vector_add_op.o memory_pool.o tensor_util.o  -D $1 -I /usr/local/cuda-11.2/include -L /usr/local/cuda-11.2/lib64 -lcudart -lpthread
else
  g++ test_predictor.cc vector_add_op.o memory_pool.o tensor_util.o  -I /usr/local/cuda-11.2/include -L /usr/local/cuda-11.2/lib64 -lcudart
fi

rm vector_add_op.o memory_pool.o tensor_util.o -rf


