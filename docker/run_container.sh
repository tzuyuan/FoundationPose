docker rm -f foundationpose
DIR=$(pwd)/../
# xhost +  && docker run --gpus all --env NVIDIA_DISABLE_REQUIRE=1 -it --network=host --name foundationpose  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v $DIR:$DIR -v /home:/home -v /mnt:/mnt -v /tmp/.X11-unix:/tmp/.X11-unix -v /tmp:/tmp  --ipc=host -e DISPLAY=${DISPLAY} -e GIT_INDEX_FILE shingarey/foundationpose_custom_cuda121:latest bash -c "cd $DIR && bash"
echo "here"
xhost + && \
docker run \
  -it \
  --gpus all \
  --env NVIDIA_DISABLE_REQUIRE=1 \
  --network=host \
  --name foundationpose \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v $DIR:$DIR \
  -v /home:/home \
  -v /mnt:/mnt \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /tmp:/tmp \
  --ipc=host \
  -e DISPLAY=${DISPLAY} \
  foundationpose_image:latest \
  bash -c "cd $DIR && bash"