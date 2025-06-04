# Software Architecture

## Folder structure

```
+---doc
+---scripts
+---src
¦   +---contrib
¦   ¦   +---ecal_whl
¦   ¦   ¦   +---ecal5-5.13.3-cp312-cp312-linux_aarch64.whl
¦   ¦   +---QHY183c_sdk
¦   ¦   ¦   +---libqhyccd.so
¦   ¦   +---Bobtracker
¦   +---background_sub
¦   +---camera_calibration
¦   +---ecal
¦   ¦   +---proto_files
¦   ¦   ¦   +---ADSB.proto
¦   ¦   +---ecal
¦   +---img_processing
¦   +---ros
¦   +---utils
¦   +---Camera controller
¦   ¦   +---QHY_controller
¦   ¦   +---Opecv_controller
¦   ¦   +---PTF_controller
¦   +---ADSB
+---platform_opi5
¦   +---nodes
¦   +---CameraNode
¦   +---ADSBNode
+---platform_pc
¦   +---nodes
+---platform_web
+---tests
+---tools
    +---camera_tools
    +---Multicast_Debug_Tools
```
# Getting Started
If you still need to setup the environment, pleas go through the Development setup.

Otherwise, make sure to update the proto binding by running the script:
```
. scripts/create_proto_bindings.sh
```

And activate the environment:
```
conda activate sky360
```


# Development setup
- Important tools

```
sudo apt install -y g++ cmake make git gedit
```

##  _~/dir_ and _~/opt_
- Copy code to _~/dir_ and install dependency libs in _~/opt_

```
cd ~ && mkdir dev && mkdir opt
```

## github
````
ssh-keygen -o -t rsa -C "your_email@email.com"
cd ~/.ssh/
gedit id_rsa.pub
````

- Install a mergetool:
```
sudo apt install -y meld
```

- Config git:
```
git config --global user.email your_email@email.com
git config --global user.name your_name
git config --global core.editor gedit
git config --global merge.tool meld
git config --global diff.tool meld
git config --global mergetool.keepBackup false
git config --global mergetool.trustExitCode false
```

### Git quick guide
Please follow git documentation:
- [Git quick guide.md](Git_quick_guide.md)

## OpenCV

- Dependencies (there might be some repeated, I lost track of it...):
```
sudo apt install -y build-essential cmake git
sudo apt install -y libgtk2.0-dev libgtk-3-dev pkg-config
sudo apt install -y libcanberra-gtk-module libcanberra-gtk3-module
sudo apt install -y libavcodec-dev libavformat-dev libswscale-dev
sudo apt install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
sudo apt install -y libjpeg-dev libpng-dev libtiff-dev
sudo apt install -y libopenblas-dev liblapack-dev libatlas-base-dev
sudo apt install -y libxvidcore-dev libx264-dev
sudo apt install -y libv4l-dev
sudo add-apt-repository universe
sudo apt update
sudo apt install -y libdc1394-dev
sudo apt install -y mesa-common-dev freeglut3-dev
```

- Clone the repo onto _~/opt_, build it and install:
```
cd ~/opt
git clone https://github.com/opencv/opencv.git
cd opencv
mkdir -p build && cd build
cmake -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D WITH_GTK=ON \
      -D WITH_QT=OFF \
      -D WITH_OPENGL=ON \
      -D WITH_V4L=ON \
      -D BUILD_EXAMPLES=OFF \
      -D LAPACK_LIBRARIES=/usr/lib/aarch64-linux-gnu/libopenblas.so \
      -D LAPACK_INCLUDE_DIR=/usr/include/aarch64-linux-gnu/openblas-pthread \
      ..
make -j4
sudo make install
```

- Some linking issues
```
ldd /usr/local/lib/libopencv_highgui.so | grep gtk
echo "/usr/local/lib" | sudo tee -a /etc/ld.so.conf.d/opencv.conf
sudo ldconfig
```

## Conda environment
```
cd ~/opt
mkdir -p miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O ~/opt/miniconda3/miniconda.sh
bash miniconda3/miniconda.sh -b -u -p miniconda3
source miniconda3/bin/activate
conda init
```

## Setting eCAL
Please follow eCAL documentation:
- [eCAL.md](eCAL.md)
