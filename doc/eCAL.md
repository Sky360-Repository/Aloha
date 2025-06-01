# eCAL Apps

The .whl is a python binding to eCAL installation, so we need to install eCAL:

```
sudo add-apt-repository ppa:ecal/ecal-latest
sudo apt-get update
sudo apt-get install ecal
sudo apt install -y protobuf-compiler
```

This installs the app but also the environment for c++

You should have a set of new apps installed: eCAL Monitor, eCAL Player, eCAL Recorder, eCAL Launcher and eCAL Sys.

The next step is to install the .whl that was compiled for the latest ecal version 5.13.3 and python 3.12

Use miniconda to make sure I have the correct python version

```
conda create -n sky360 python=3.12
conda activate sky360
pip install src/contrib/ecal_whl/ecal5-5.13.3-cp312-cp312-linux_aarch64.whl
```

# ecal5-5.13.3-cp312-cp312-linux_aarch64.whl

If the .whl is missing or you have a different setup, you need to compile the code and build the .whl binding.

## Ecal dependencies - may be optional
This is a list of dependencies that were needed during the process.

Ecal dependencies:
```
sudo apt-get install -y git cmake doxygen graphviz build-essential zlib1g-dev qtbase5-dev libhdf5-dev libprotobuf-dev libprotoc-dev protobuf-compiler libcurl4-openssl-dev libqwt-qt5-dev libyaml-cpp-dev
sudo apt install -y cmake build-essential python3-dev python3-pip
sudo apt install -y qtbase5-dev
sudo apt install -y libhdf5-dev
sudo apt install -y libyaml-cpp-dev
sudo apt install -y libqwt-qt5-dev
sudo apt install -y libprotobuf-dev protobuf-compiler
sudo apt install -y doxygen
sudo apt install -y patchelf

sudo apt-get -y install python3.10-dev python3-pip
sudo apt install python3-dev python3-pip python3-setuptools
pip install --upgrade setuptools wheel build

python3 -m pip install setuptools

sudo apt update
```

Clone eCAL and checkout the version matching the applications installed. Go to eCAL Launcher and confirm the version, the version for this ecample was v5.13.3. The version in the code is a TAG v5.13.3.

```
cd ~/opt
git clone https://github.com/eclipse-ecal/ecal.git
cd ecal
git checkout v5.13.3
```

Don't forget the submodules!!
```
git submodule sync --recursive && git submodule update --init --recursive
```

make:
```
mkdir build && cd build

cmake -DPYTHON_EXECUTABLE=$(which python3) \
      -DBUILD_PY_BINDING=ON \
      -DBUILD_STANDALONE_PY_WHEEL=ON \
      -DCMAKE_BUILD_TYPE=Release \
      -DECAL_THIRDPARTY_BUILD_PROTOBUF=OFF \
      -DECAL_THIRDPARTY_BUILD_CURL=OFF \
      -DECAL_THIRDPARTY_BUILD_HDF5=OFF \
      -DECAL_THIRDPARTY_BUILD_QWT=OFF ..
make -j4
```

Following eCAL documentation:
```
cmake --build . --target create_python_wheel --config Release
```

This should build ***~/opt/ecal/build/_deploy/ecal5-5.13.3-cp312-cp312-linux_aarch64.whl***

Copy this file and pip install in the conda environment.

# py hello eCAL
From here you should be able to run the python examples in

There's an hello world in
```
Aloha/src/ecal
```

Basic test:

- Open a terminal and run
```
conda activate sky360
python test_rec.py
```
- Open a second terminal and run
```
conda activate sky360
python test_snd.py
```

# Working with cameras in eCAL

```
python webcam2ecal.py --camera_id 0 --channel_name camera_1 &
python view_ecal_video.py --channel_name camera_1 &
```

- camera_id 0 is the camera number used in opencv and this function needs a webcamera.
- channel_name camera_1 is the topic name. eCAL calls it channels and ROS calls it topics, but is the same concept.

Use eCAL Monitor to see the communication in the channel.
