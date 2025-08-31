#!/bin/bash
set -e

log() { echo -e "\033[1;34m[CHECK]\033[0m $1"; }
warn() { echo -e "\033[1;33m[WARN]\033[0m $1"; }
fail() { echo -e "\033[1;31m[FAIL]\033[0m $1"; }

# In case of errors, try
# ``` bash
# sed -i 's/\r$//' setup.sh
# ```

# Platform detection
detect_platform() {
  log "Checking platform..."
  if grep -q "Orange Pi 5 Plus" /proc/device-tree/model; then
    log "✅ OrangePi5+ detected"
  else
    warn "⚠️ Platform not OrangePi5+. Detected: $(cat /proc/device-tree/model)"
  fi
}

# Create dev and opt folders
setup_folders() {
  log "Creating ~/dev and ~/opt folders"
  mkdir -p ~/dev ~/opt
}

# Install dev tools
install_dev_tools() {
  log "Installing development tools"
  sudo apt update
  sudo apt install -y g++ cmake make git gedit git-gui meld
}

# Clone Aloha repo
clone_aloha() {
  log "Cloning Aloha repository"
  if ! ssh -T git@github.com &> /dev/null; then
    fail "❌ SSH access to GitHub failed. Please set up your SSH key before running this script."
    return 1
  fi

  cd ~/dev || { fail "❌ Failed to access ~/dev"; return 1; }

  if ! git clone git@github.com:Sky360-Repository/Aloha.git; then
    fail "❌ Failed to clone Aloha repository. Check SSH access or repo URL."
    return 1
  fi
}

# Install dependencies
install_dependencies() {
  log "Installing system dependencies"
  sudo apt install -y build-essential libgtk2.0-dev libgtk-3-dev pkg-config \
    libcanberra-gtk-module libcanberra-gtk3-module \
    libavcodec-dev libavformat-dev libswscale-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    libjpeg-dev libpng-dev libtiff-dev \
    libopenblas-dev liblapack-dev libatlas-base-dev \
    libxvidcore-dev libx264-dev libv4l-dev \
    libdc1394-dev mesa-common-dev freeglut3-dev
  sudo apt update
}

# Build OpenCV
build_opencv() {
  log "Building OpenCV"
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

  ldd /usr/local/lib/libopencv_highgui.so | grep gtk
  echo "/usr/local/lib" | sudo tee -a /etc/ld.so.conf.d/opencv.conf
  sudo ldconfig
}

# Conda environment
install_conda() {
  log "Installing Miniconda"
  cd ~/opt
  mkdir -p miniconda3
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O miniconda3/miniconda.sh
  bash miniconda3/miniconda.sh -b -u -p miniconda3
  source miniconda3/bin/activate
  conda init
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
}

purge_ecal() {
  log "🔧 Purging system-installed eCAL packages"
  sudo apt-get purge -y ecal libecal* python3-ecal || true
  sudo apt-get purge -y 'libecal*' 'ecal*' || true

  log "🔧 Removing eCAL PPAs"
  sudo add-apt-repository --remove -y ppa:ecal/ecal || true
  sudo add-apt-repository --remove -y ppa:ecal/ecal-5.x || true
  sudo add-apt-repository --remove -y ppa:ecal/ecal-6.x || true

  log "🧹 Cleaning up orphaned packages"
  sudo apt-get autoremove --purge -y
  sudo apt-get clean

  log "📦 Re-adding eCAL 5.x PPA for compatibility"
  sudo add-apt-repository -y ppa:ecal/ecal-5
  sudo apt-get update
}

purge_ecal_source() {
  log "🧹 Removing source-installed eCAL files"

  # Remove system-level eCAL binaries and headers
  sudo rm -rf /usr/local/bin/ecal*
  sudo rm -rf /usr/local/lib/libecal*
  sudo rm -rf /usr/local/include/ecal

  # Uninstall eCAL from Conda environment
  if conda info --envs | grep -q "^sky360"; then
    log "🔧 Uninstalling eCAL from Conda environment 'sky360'"
    conda run -n sky360 pip uninstall ecal -y || true
  else
    warn "⚠️ Conda environment 'sky360' not found — skipping pip uninstall"
  fi

  # Clean build artifacts and wheel copies
  rm -rf ~/opt/ecal/build
}

# Build and install eCAL
build_ecal() {
  log "Building eCAL v5.13.3"
  cd ~/opt
  git clone --recurse-submodules https://github.com/eclipse-ecal/ecal.git
  cd ecal
  git fetch
  git checkout v5.13.3
  git submodule sync --recursive && git submodule update --init --recursive

  sudo apt install -y git cmake doxygen graphviz build-essential zlib1g-dev qtbase5-dev \
    libhdf5-dev libprotobuf-dev libprotoc-dev protobuf-compiler libcurl4-openssl-dev \
    libqwt-qt5-dev libyaml-cpp-dev python3-dev python3-pip patchelf

  pip install --upgrade setuptools wheel build

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
  cmake --build . --target create_python_wheel --config Release

  cd python
  pip wheel . -w ../dist
  cp ~/opt/ecal/build/dist/*.whl ~/dev/Aloha/src/contrib/ecal_whl
}

# Final setup
finalize_aloha_env() {
  log "Setting up Aloha Conda environment"
  cd ~/dev/Aloha
  conda create -n sky360 python=3.12 -y
  conda activate sky360
  pip install -r requirements.txt
  pip install src/contrib/ecal_whl/ecal5-5.13.3-cp312-cp312-linux_aarch64.whl
  conda install -c conda-forge libstdcxx-ng -y
  conda run -n sky360 python -c "import ecal; print('eCAL loaded successfully')"
}

# Main execution
main() {
  detect_platform
  setup_folders
  install_dev_tools
  clone_aloha
  install_dependencies

  if ldconfig -p | grep -q libopencv_core; then
    log "✅ OpenCV libraries detected"
  else
    build_opencv
  fi

  if command -v conda &> /dev/null; then
    log "Conda already installed"
  else
    install_conda
  fi

  log "🔍 Checking eCAL installation..."
  # Check if eCAL is installed via APT
  if dpkg -s libecal-core &> /dev/null; then
    installed_version=$(dpkg -s libecal-core | grep Version | awk '{print $2}')
    if [[ "$installed_version" != "5.13.3" ]]; then
      warn "⚠️ eCAL installed via APT: version $installed_version (expected 5.13.3)"
      purge_ecal
      build_ecal
    else
      log "✅ eCAL APT version is correct: $installed_version"
    fi

  # Check if eCAL is installed from source
  elif ls /usr/local/lib | grep -q libecal-core; then
    version_hint=$(strings /usr/local/lib/libecal-core.so | grep -Eo '5\.[0-9]+\.[0-9]+' | head -n 1)
    if [[ "$version_hint" != "5.13.3" ]]; then
      warn "⚠️ eCAL source version appears to be $version_hint (expected 5.13.3)"
      purge_ecal_source
      build_ecal
    else
      log "✅ eCAL source version appears to be correct: $version_hint"
    fi

  # No eCAL detected
  else
    warn "⚠️ No eCAL installation detected — proceeding with build"
    build_ecal
  fi

  finalize_aloha_env
  log "✅ Setup complete. Ready to develop!"
}

main
