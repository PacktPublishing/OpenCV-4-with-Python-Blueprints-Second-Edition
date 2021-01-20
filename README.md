# OpenCV-4-with-Python-Blueprints-Second-Edition

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.154060.svg)](https://doi.org/10.5281/zenodo.154060)
[![Google group](https://img.shields.io/badge/Google-Discussion%20group-lightgrey.svg)](https://groups.google.com/d/forum/opencv-python-blueprints)
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](http://www.gnu.org/licenses/gpl-3.0)

This repository contains all up-to-date source code for the following book:

<img src="https://www.packtpub.com/media/catalog/product/cache/ecd051e9670bd57df35c8f0b122d8aea/9/7/9781789801811-original.png" align="left" width="200" style="margin-right: 5px"/>
Dr. Menua Gevorgyan, Arsen Mamikonyan, Michael Beyeler <br/>
<a href="https://www.packtpub.com/data/opencv-4-with-python-blueprints-second-edition"><b>OpenCV 4 with Python Blueprints - Second Edition</b></a> Build creative computer vision projects with the latest version of OpenCV 4 and Python 3


Packt Publishing Ltd. <br/>
Paperback: 366 pages <br/>
ISBN 978-178980-181-1
<br clear="both"/>

This book demonstrates how to develop a series of intermediate to advanced projects using OpenCV and Python,
rather than teaching the core concepts of OpenCV in theoretical lessons. Instead, the working projects
developed in this book teach the reader how to apply their theoretical knowledge to topics such as
image manipulation, augmented reality, object tracking, 3D scene reconstruction, statistical learning,
and object categorization.

By the end of this book, readers will be OpenCV experts whose newly gained experience allows them to develop their own advanced computer vision applications.

If you use either book or code in a scholarly publication, please cite as:
> Menua Gevorgyan, Arsen Mamikonyan, Michael Beyeler, (2020). OpenCV with Python Blueprints - Second Edition: Build creative computer vision projects with the latest version of OpenCV 4 and Python 3. Packt Publishing Ltd., London, England, 230 pages, ISBN 978-178980-181-1.

Or use the following bibtex:
```
@book{OpenCVWithPythonBlueprints,
	title = {{OpenCV with Python Blueprints}},
	subtitle = {Build creative computer vision projects with the latest version of {OpenCV 4} and {Python 3}},
	author = {Menua Gevorgyan, Arsen Mamikonyan, Michael Beyeler},
	year = {2020},
	pages = {366},
	publisher = {Packt Publishing Ltd.},
	isbn = {978-178980-181-1}
}
```

Scholarly work referencing first edition of the book:
- B Zhang et al. (2018). Automatic matching of construction onsite resources under camera views. *Automation in Construction*.
- A Jakubović & J Velagić (2018). Image Feature Matching and Object Detection Using Brute-Force Matchers. *International Symposium ELMAR*.
- B Zhang et al. (2018). Multi-View Matching for Onsite Construction Resources with Combinatorial Optimization. *International Symposium on Automation and Robotics in Construction (ISARC)* 35:1-7.
- LA Marcomini (2018). Identificação automática do comportamento do tráfego a partir de imagens de vídeo. *Escola de Engenharia de São Carlos*, Master's Thesis.
- G Laica et al. (2018). Diseño y construcción de un andador inteligente para el desplazamiento autónomo de los adultos mayores con visión reducida y problemas de movilidad del hogar de vida "Luis Maldonado Tamayo" mediante la investigación de técnicas de visión artificial. *Departamento de Ciencias de la Energía y Mecánica, Universidad de las Fuerzas Armadas ESPE*, Master's Thesis.
- I Huitzil-Velasco et al. (2017). Test of a Myo Armband. *Revista de Ciencias Ambientales y Recursos Naturales* 3(10): 48-56.
- Y Güçlütürk et al. (2016). Convolutional sketch inversion. *European Conference on Computer Vision (ECCV)* 810-824.


All code was tested with OpenCV 4.2.0 and Python 3.8 on Ubuntu 18.04, and is available from:
https://github.com/PacktPublishing/OpenCV-4-with-Python-Blueprints-Second-Edition/

We have also created a Docker file in https://github.com/PacktPublishing/OpenCV-4-with-Python-Blueprints-Second-Edition/tree/master/dockerfiles/Dockerfile which has README.md that will help you compile and run the code using the docker file.


## Critical Reception (First Edition)

<img src="https://3.bp.blogspot.com/-m8yl8xCrM3Q/V9yFYMAj3YI/AAAAAAAAAq8/5IzGqAeUp9cCwq13j1EL7aunfUvvre5bQCLcB/s640/opencv-python-blueprints-amazon-new.png" style="width: 70%; margin-left: 15%"/>

What readers on Amazon have to say:

> The author does a great job explaining the concepts needed to understand what's happening in the application without
> the need of going into too many details. <br/>
&ndash; [Sebastian Montabone](http://www.samontab.com)

> Excellent book to build practical OpenCV projects! I'm still relatively new to OpenCV, but all examples are well
> laid out and easy to follow. The author does a good job explaining the concepts in detail and shows how they apply
> in real life. As a professional programmer, I especially love that you can just fork the code from GitHub and follow
> along. Strongly recommend to readers with basic knowledge of computer vision, machine learning, and Python!
&ndash; Amazon Customer

> Usually I'm not a big fan of technical books because they are too dull, but this one is written in an engaging
> manner with a few dry jokes here and there. Can only recommend! <br/>
&ndash; lakesouth

## Who This Book Is for
As part of Packt's Blueprints series, this book is for intermediate users of OpenCV who aim to master their skills
by developing advanced practical applications. You should already have some
experience of building simple applications, and you are expected to be familiar with
OpenCV's concepts and Python libraries. Basic knowledge of Python programming
is expected and assumed.

By the end of this book, you will be an OpenCV expert, and your newly gained
experience will allow you to develop your own advanced computer vision
applications.


##  Getting Started
All projects can run on Windows, Mac, or Linux. The required packages can be installed with pip or you can use the docker images available in the repository to run scripts of the chapters.

## Installation With Pip


```
pip install -r requirements.txt
```

## Runninning With Docker



### Build the Image

The repository contains two docker images:

1. Without GPU acceleration
```
docker build -t book dockerfiles
```
2. With GPU (CUDA) acceleration
```
docker build -t book dockerfiles -f dockerfiles/gpu.Dockerfile
```

### Start a Container

```
docker run --device /dev/video0 --env DISPLAY=$DISPLAY  -v="/tmp/.X11-unix:/tmp/.X11-unix:rw"  -v `pwd`:/book -it book
```

Here, we have allowed docker to connect to the default camera and to use the X-11 server of the host machine to run graphical applications. In case if you use the GPU version of the images, you also have to pass `--runtime nvidia`.

### Run an App
In the container, locate a desired chapter:
```
cd /book/chapterX
```
and run a desired script of the chapter:
```
python chapterX.py
```



### Troubleshooting

#### Could not connect to any X display.

The X Server should allow connections from a docker container.

Run `xhost +local:docker`, also check [this](https://forums.docker.com/t/start-a-gui-application-as-root-in-a-ubuntu-container/17069)


## The Following Packages Were Used in the Chapters of the Book
* OpenCV 4.2 or later: Recent 32-bit and 64-bit versions as well as installation instructions are available at
http://opencv.org/downloads.html. Platform-specific installation instructions can be found at
http://docs.opencv.org/doc/tutorials/introduction/table_of_content_introduction/table_of_content_introduction.html.
* Python 3.8 or later: Recent 32-bit and 64-bit installers are available at https://www.python.org/downloads. The
installation instructions can be found at https://wiki.python.org/moin/BeginnersGuide/Download.
* NumPy 1.18.1 or later: This package for scientific computing officially comes in 32-bit format only, and can be
obtained from http://www.scipy.org/scipylib/download.html. The installation instructions can be found at
http://www.scipy.org/scipylib/building/index.html#building.

In addition, some chapters require the following free Python modules:
* wxPython 4.0 or later (Chapters 1 to 4, 8): This GUI programming toolkit can be obtained from
  http://www.wxpython.org/download.php.
* matplotlib 3.1 or later (Chapters 4, 5, 6, and 7): This 2D plotting library can be obtained from
  http://matplotlib.org/downloads.html. Its installation instructions can be found by going to
  http://matplotlib.org/faq/installing_faq.html#how-to-install.
* SciPy 1.4 or later (Chapter 1 and 10): This scientific Python library officially comes in 32-bit only, and can be
  obtained from http://www.scipy.org/scipylib/download.html. The installation instructions can be found at
  http://www.scipy.org/scipylib/building/index.html#building.
* rawpy 0.14 and ExifRead==2.1.2 (Chapter 5)
* requests==2.22.0 to download data in chapter 7

Furthermore, the use of iPython (http://ipython.org/install.html) is highly recommended as it provides a flexible,
interactive console interface.


## License
The software is released under the GNU General Public License (GPL), which is the most commonly used free software
license according to Wikipedia. GPL allows for commercial use, distribution, modification, patent use, and private use.

The GPL is a copyleft license, which means that derived works can only be distributed under the same license terms.
For more information, please see the license file.
