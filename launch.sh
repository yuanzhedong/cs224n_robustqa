
   
nvidia-docker run --gpus all -it --rm --ipc=host \
  --gpus device=all \
  --network=host \
  --shm-size=1g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --privileged \
  -v $PWD:/workspace \
  c5b25e35dbe8
  