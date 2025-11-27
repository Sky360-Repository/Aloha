# Getting Started

Our curated list of recommended hardware products can be found in [Sky360 Hardware](https://www.sky360.org/documentation#CuratedListofHardware)

Minimum setup to contribute to this project:
* Orange Pi5+ 16GB
* SanDisk MAX ENDURANCE 64 GB microSDXC Memory Card - Recommended to have a second cards as a backup
* QHY183c - please contact us to get the correct [QHY SDK](https://drive.google.com/drive/u/0/folders/1H8pIfihB8eMw67oz7WuLpJ6XvD_zrS6P?ths=true)
* UBUNTU-24.04 desktop for Orange P5 plus - please contact us to get the correct [UBUNTU-24.04 Image ISO](https://drive.google.com/file/d/1muHc1hqhuZ0o6j6jQEbMYEPHzJo2qxyl/view?usp=drive_link)

We understand that an astrophotography camera like QHY183c is expensive, specially if you just want to contribute to software
development. For this reason we also recommend these budget cameras: USB Fisheye Camera Module ELP_USB500W05G_BL180 or Dual Lens version ELP-USB3D1080P02-L180

## Flashing [UBUNTU-24.04 desktop for Orange P5 plus](https://drive.google.com/file/d/1muHc1hqhuZ0o6j6jQEbMYEPHzJo2qxyl/view?usp=drive_link)

1. Download the [UBUNTU-24.04 Image ISO](https://drive.google.com/file/d/1muHc1hqhuZ0o6j6jQEbMYEPHzJo2qxyl/view?usp=drive_link)
   and flash the Image to SD Card using BalenaEtcher
2. Boot from SD Card: Start by booting your Orange Pi 5 using the SD card and go through the installation process.
3. CPU speed and governor - configure these settings using tools like `cpufrequtils`
```
sudo apt install cpufrequtils
```

Current configurations
```
cpufreq-info
```

Set best performance
```
sudo cpufreq-set -g schedutil && sudo cpufreq-set -u 2.21GHz -d 408MHz
```

4. Configure **ZRAM Swap**

Ubuntu typically enables swap via a disk-backed swapfile (on SD or NVMe).

This will cause wear on the SD or NVMe drives.

Please follow this instructions to configure **ZRAM Swap**: [swapfile.md](swapfile.md)

5. Install QHY SDK
```
extract sdk_Arm64_24.12.26.tgz
sudo chmod +x *.sh
sudo ./install.sh
setup UDEV rules
sudo cp 85-qhyccd.rules /etc/rules.d
add the following text at the end of this file
SUBSYSTEM==”usb”, ATTR{idVendor}==”1618”, MODE=”0666”, GROUP=”plugdev”
sudo udevadm control –reload-rules
sudo udevadm trigger
```

6. Setup a github account

[https://github.com](https://github.com)

7. Add the SSH key to the github account
````
ssh-keygen -o -t rsa -C "your_email@email.com"
cd ~/.ssh/
gedit id_rsa.pub
````
Copy the SSH key to your github account [https://github.com/settings/keys](https://github.com/settings/keys)

8. Configure git
```
git config --global user.email your_email@email.com
git config --global user.name your_name
```

9. You can now set your OrangePi5+ development environment using this setup script

```bash
curl -O https://raw.githubusercontent.com/Sky360-Repository/Aloha/scripts/setup.sh

source setup.sh
```

10. Check if development environment is ready:
```bash
source scripts/diagnose_setup.sh
```

11. All done

Go to Aloha
```
cd ~/dev/Aloha
conda activate sky360
```

Make sure to update the proto binding by running the script:
```
. scripts/create_proto_bindings.sh
```

And activate the environment follow our [Git quick guide](Git_quick_guide.md)


## Manual setup

If you still need to setup the environment, please go through this Development setup.

- Important tools
```
sudo apt install -y g++ cmake make git gedit git-gui
```

###  _~/dir_ and _~/opt_
- Copy code to _~/dir_ and install dependency libs in _~/opt_

```
cd ~ && mkdir dev && mkdir opt
```

### github
````
ssh-keygen -o -t rsa -C "your_email@email.com"
cd ~/.ssh/
gedit id_rsa.pub
````

- Install a mergetool:
```
sudo apt install -y meld
```

- Configure git:
```
git config --global user.email your_email@email.com
git config --global user.name your_name
git config --global core.editor gedit
git config --global merge.tool meld
git config --global diff.tool meld
git config --global mergetool.keepBackup false
git config --global mergetool.trustExitCode false
```

### OpenCV

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

- Linking issues
```
ldd /usr/local/lib/libopencv_highgui.so | grep gtk
echo "/usr/local/lib" | sudo tee -a /etc/ld.so.conf.d/opencv.conf
sudo ldconfig
```

### Conda environment
```
cd ~/opt
mkdir -p miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O ~/opt/miniconda3/miniconda.sh
bash miniconda3/miniconda.sh -b -u -p miniconda3
source miniconda3/bin/activate
conda init
```

### Setting eCAL

Please follow eCAL documentation: [eCAL.md](eCAL.md)

## All done

Check if development environment is ready:
```bash
source scripts/diagnose_setup.sh
```

