#!/bin/bash
# This gist is a step by step instructions to build and install OpenCV from source on ubuntu 18.04 LTS
# note:  The easy and quick way to install is
# sudo pip3 install opencv-python
# sudo pip3 install opencv-contrib-python
# But this easy pypi installation can’t open video files on GNU/Linux distribution or on mac OS X system.
# And on some system opencv binaries provided packages are not compiled.
# Therefor we have no way rather than build it from source. 


### first update and upgrade pre-install apt-get packages.
sudo apt-get update
sudo apt-get upgrade


### install developer tools
sudo apt -y install build-essential checkinstall cmake pkg-config
sudo apt -y install git gfortran


### install image I/O packages for loading various image file formats from disk
sudo apt -y install libjpeg8-dev libpng-dev

sudo add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main"
sudo apt install libjasper1 libjasper-dev

###  GTK development library to build Graphical User Interfaces
sudo apt -y install libgtk-3-dev libtbb-dev qt5-default

### Other dependcies
sudo apt -y install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libdc1394-22-dev
sudo apt -y install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
sudo apt -y install libatlas-base-dev
sudo apt -y install libfaac-dev libmp3lame-dev libtheora-dev
sudo apt -y install libxvidcore-dev libx264-dev
sudo apt -y install libopencore-amrnb-dev libopencore-amrwb-dev
sudo apt -y install libgphoto2-dev libeigen3-dev libhdf5-dev doxygen x264 v4l-utils

### downloading opencv and opencv_contrib packages from their GitHub repositories
#git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git

### checkout the same version opencv and opencv_contrib
cd opencv 
git checkout 3.4.1 
cd ..

cd opencv_contrib
git checkout 3.4.1
cd ..

#### now compile and install OpenCV with contrib modules
# create a build directory
cd opencv
mkdir build
cd build

#  configure our build using cmake
cmake -D CMAKE_BUILD_TYPE=RELEASE \
	-D CMAKE_INSTALL_PREFIX=/usr/local \
	-D INSTALL_C_EXAMPLES=ON \
	-D INSTALL_PYTHON_EXAMPLES=ON \
	-D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
        -D PYTHON_EXECUTABLE=/usr/bin/python3 \
	-D BUILD_EXAMPLES=ON ..

# OPENCV_EXTRA_MODULES_PATH path can be differenet depending upon opencv_contrib/modules location
# python executable path can be found by entering following code in python terminal 
# for python3 execuatable path open the terminal and type python3. and then enter
# import sys; print(sys.executable)

# compile opencV in the same the build folder
make -j8

# here 4 is the availabe core of my processor 
# to find  number of CPU cores in your machine enter "nproc" command on your terminal
# to speed up compiling process enter highest number of available cores on your processor.

# now  install it on your ubuntu system
sudo make install
sudo sh -c 'echo "/usr/local/lib" >> /etc/ld.so.conf.d/opencv.conf'
sudo ldconfig


# opencv’s Python binary (cv2.so) can be installed either in directory site-packages or dist-packages
# for finding corrent location enter
find /usr/local/lib/ -type f -name "cv2*.so"

# my binary is installed in dist-packages. above command shows following output
#/usr/local/lib/python3.6/dist-packages/cv2.cpython-36m-x86_64-linux-gnu.so
# now we need to rename it to cv2.so

cd /usr/local/lib/python3.6/dist-packages/
sudo mv cv2.cpython-36m-x86_64-linux-gnu.so cv2.so

## well that's it! to confirm your installation go to the python terminal and enter
## import cv2; print(cv2.__version__)
## if that outputs "3.4.1" and we are done !!!
