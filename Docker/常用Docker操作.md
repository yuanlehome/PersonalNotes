`进入docker` 
```shell
sudo docker attach $docker_id
sudo docker exec -it $docker_id /bin/bash  
```

```shell
sudo docker ps -a | grep
sudo nvidia-docker run  -it --name=weishengying_gpu_docker_cuda_11_6 --net=host -v $PWD/weishengying:/weishengying --privileged=true $image_id  /bin/bash
```

 