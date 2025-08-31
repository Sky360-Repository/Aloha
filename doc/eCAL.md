# eCAL Apps

Due to recent updates in eCAL, we have to build and install eCAL v5.13.3 as this is the last stable version.

We recommend building and installing from source rather than using APT, to prevent automatic updates.

⚠️ Do not install eCAL 6.x

Clone eCAL:
```
cd ~/opt
git clone --recurse-submodules https://github.com/eclipse-ecal/ecal.git
```

Get the last stable version:
```
cd ecal
git fetch
git checkout v5.13.3
git submodule sync --recursive && git submodule update --init --recursive

```

Known dependencies
```
sudo apt install -y git cmake doxygen graphviz build-essential zlib1g-dev qtbase5-dev \
    libhdf5-dev libprotobuf-dev libprotoc-dev protobuf-compiler libcurl4-openssl-dev \
    libqwt-qt5-dev libyaml-cpp-dev python3-dev python3-pip patchelf
```

This installs the core applications and sets up the C++ development environment.
```
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
sudo make install

cmake -DPYTHON_EXECUTABLE=$(which python3) \
  -DBUILD_PY_BINDING=ON \
  -DBUILD_STANDALONE_PY_WHEEL=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DECAL_THIRDPARTY_BUILD_PROTOBUF=OFF \
  -DECAL_THIRDPARTY_BUILD_CURL=OFF \
  -DECAL_THIRDPARTY_BUILD_HDF5=OFF \
  -DECAL_THIRDPARTY_BUILD_QWT=OFF ..

make -j4
sudo make install
```

After building, you may need to update and reboot to ensure all system paths and libraries are properly registered:
```
sudo apt update && sudo reboot
```

Once rebooted, verify that the correct version of eCAL is installed:
```
ecal_config --version  # Should return v5.13.3
```

You should have a set of new apps installed: eCAL Monitor, eCAL Player, eCAL Recorder, eCAL Launcher and eCAL Sys.

The next step is to install the .whl that was compiled for the latest ecal version 5.13.3 and python 3.12

```
# Build Python bindings and wheel (after installing C++ core)
cmake --build . --target create_python_wheel --config Release

# Create and activate the Conda environment for Aloha
conda create -n sky360 python=3.12
conda activate sky360

## Build and Install the Python Wheel
cd python
pip install --upgrade setuptools wheel build
pip wheel . -w ../dist
```

Copy the wheel to the Aloha repository
```
cp ~/opt/ecal/build/dist/*.whl ~/dev/Aloha/src/contrib/ecal_whl
```

Use Miniconda to ensure the correct Python version is installed
```
conda create -n sky360 python=3.12
conda activate sky360
pip install -r requirements.txt
pip install src/contrib/ecal_whl/ecal5-5.13.3-cp312-cp312-linux_aarch64.whl

conda install -c conda-forge libstdcxx-ng
```

Confirm the development environment is ready:
```bash
cd ~/dev/Aloha
source scripts/diagnose_setup.sh
```

# py hello eCAL
From here you should be able to run the python examples in

There's an hello world in
```
Aloha/src/ecal_lib
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

# eCAL 6.0.0

Our project doesn't support eCAL 6.0.0.
Recent updates have change the API and this has a impact on python.
We decided to continue with eCAL 5.13.3 and this is the last stable version.

## Downgrading eCAL from 6.x to 5.x on Ubuntu

This guide walks through the steps to safely remove eCAL 6.x, configure your system to prefer eCAL 5.x, and ensure future upgrades do not override the pinned version.

## Step 1: Remove eCAL 6.x
```bash
sudo apt-get purge ecal libecal* python3-ecal
```

Removes core eCAL packages. If installed from source, manually delete any residual files in `/usr/local/lib` and `/usr/local/include`.

##  Step 2: Remove Conflicting PPAs

```bash
sudo add-apt-repository --remove ppa:ecal/ecal
sudo add-apt-repository --remove ppa:ecal/ecal-6.x
```

Ensures apt won't pull newer 6.x versions.

## Step 3: Add the Correct 5.x PPA

```bash
sudo add-apt-repository ppa:ecal/ecal-5
sudo apt-get update
```

Adds the stable 5.13.3 release stream.

## Step 4: Pin eCAL to 5.13.3

Create a preferences file:

```bash
sudo gedit /etc/apt/preferences.d/ecal
```

Paste the following content:

```plaintext
Package: ecal
Pin: version 5.13.3
Pin-Priority: 1001

Package: libecal*
Pin: version 5.13.3
Pin-Priority: 1001

Package: python3-ecal
Pin: version 5.13.3
Pin-Priority: 1001
```

This forces apt to prefer 5.13.3 versions even if newer ones are available.

## Step 5: Install and Hold eCAL 5.13.3

```bash
sudo apt-get install ecal
sudo apt-mark hold ecal libecal-core libecal-*.so python3-ecal
```

Prevents future upgrades from pulling in 6.x packages.

## Step 6: Verify Installed Version

```bash
ecal_core --version
```

Should return a version like `v5.13.3`.


## Optional Cleanup (if needed)

### Remove eCAL and Related Packages

```bash
sudo apt-get purge 'libecal*' 'ecal*'
```

### Clean Up Residuals

```bash
sudo apt-get autoremove --purge
sudo apt-get clean
```

### Double-Check Removal

```bash
dpkg -l | grep ecal
which ecal_monitor
```

## Unpin eCAL (if reverting)

### Remove the Pinning File

```bash
sudo rm /etc/apt/preferences.d/ecal
```

### Refresh Package Lists

```bash
sudo apt-get update
```

### Confirm Pinning Is Gone

```bash
apt-cache policy ecal
```

### Clear Cached `.deb` Files

```bash
sudo apt-get clean
```


