## Build Image

```
docker build -t book dockerfiles
```
## Run

```
docker run --device /dev/video0 --env DISPLAY=$DISPLAY  -v="/tmp/.X11-unix:/tmp/.X11-unix:rw"  -v `pwd`:/book -it book
```

From there cd chapterX

python chapterX.py


### Troubleshooting

#### Could not connect to any X display.

The X Server should allow connections from a docker container.

Run `xhost +local:docker`, also check [this](https://forums.docker.com/t/start-a-gui-application-as-root-in-a-ubuntu-container/17069)
