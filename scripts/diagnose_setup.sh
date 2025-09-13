#!/bin/bash
# set -e

log() { echo -e "\033[1;34m[CHECK]\033[0m $1"; }
warn() { echo -e "\033[1;33m[WARN]\033[0m $1"; }
fail() { echo -e "\033[1;31m[FAIL]\033[0m $1"; }

# In case of errors, try
# ``` bash
# sed -i 's/\r$//' diagnostic_setup.sh
# ```

check_orangepi() {
  log "Checking platform..."
  if grep -q "Orange Pi 5 Plus" /proc/device-tree/model; then
    log "✅ OrangePi5+ detected"
  else
    warn "⚠️ Platform not OrangePi5+. Detected: $(cat /proc/device-tree/model)"
  fi
}

check_os_version() {
  log "Checking OS version..."
  if grep -q "Ubuntu 24.04" /etc/os-release; then
    log "✅ Ubuntu 24.04 detected"
  else
    warn "⚠️ Unexpected OS version: $(grep PRETTY_NAME /etc/os-release)"
  fi
}

check_git_config() {
  log "Checking Git configuration..."

  name=$(git config --global user.name || echo "unset")
  email=$(git config --global user.email || echo "unset")

  if [[ "$name" == "unset" || "$email" == "unset" ]]; then
    warn "⚠️ Git user.name or user.email not set"
  else
    log "✅ Git configured as $name <$email>"

    # Check SSH key comment
    if [ -f "$HOME/.ssh/id_rsa.pub" ]; then
      ssh_comment=$(awk '{print $3}' "$HOME/.ssh/id_rsa.pub")
      if [[ "$ssh_comment" == "$email" ]]; then
        log "✅ SSH key email matches Git config"
      else
        warn "⚠️ SSH key email ($ssh_comment) does not match Git config ($email)"
      fi
    else
      warn "⚠️ SSH public key not found at ~/.ssh/id_rsa.pub"
    fi
  fi
}

check_conda() {
  log "Checking Conda installation..."
  if command -v conda &> /dev/null; then
    log "✅ Conda found: $(conda --version)"
  else
    fail "❌ Conda not found"
  fi
}

check_opencv() {
  log "Checking OpenCV installation..."
  if ldconfig -p | grep -q libopencv_core; then
    log "✅ OpenCV libraries detected"
  else
    fail "❌ OpenCV not found in system libraries"
  fi
}

check_ecal_libs() {
  log "Checking eCAL installation..."

  # Check for eCAL 5.x via libecal-core
  if dpkg -s libecal-core &> /dev/null; then
    installed_version=$(dpkg -s libecal-core | grep Version | awk '{print $2}')
    if [[ "$installed_version" == "5.13.3" ]]; then
      log "✅ eCAL 5.x installed via APT: version $installed_version"
    else
      warn "⚠️ eCAL 5.x installed via APT: version $installed_version (expected 5.13.3)"
    fi
  # Check for eCAL 6.x via ecal package
  elif dpkg -s ecal &> /dev/null; then
    installed_version=$(dpkg -s ecal | grep Version | awk '{print $2}')
    warn "⚠️ eCAL 6.x installed via APT: version $installed_version (expected 5.13.3)"

  # Check for source install
  elif ls /usr/local/lib | grep -q libecal_core; then
    log "✅ eCAL core libraries found in /usr/local/lib (likely source install)"
    version_hint=$(ecal_config --version | grep -Eo 'v[0-9]+\.[0-9]+\.[0-9]+')
    if [[ "$version_hint" == "v5.13.3" ]]; then
      log "✅ eCAL source version appears to be $version_hint"
    else
      warn "⚠️ eCAL source version appears to be $version_hint (expected v5.13.3)"
    fi

  # No eCAL detected
  else
    fail "❌ eCAL not found via APT or in /usr/local/lib"
  fi
}

check_ecal_python() {
  log "Checking eCAL Python module in Conda environment 'sky360'..."

  # Check if the environment exists
  if conda info --envs | grep -q "^sky360"; then
    log "✅ Conda environment 'sky360' exists"
    
    # Check if eCAL is importable inside the environment
    if conda run -n sky360 python -c "import ecal" &> /dev/null; then
      log "✅ eCAL module is importable inside 'sky360'"
    else
      fail "❌ eCAL module not found in 'sky360' environment"
    fi
  else
    fail "❌ Conda environment 'sky360' does not exist"
  fi
}

check_aloha_repo() {
  log "Checking Aloha repo presence..."
  if [ -d "$HOME/dev/Aloha" ]; then
    log "✅ Aloha repo found at ~/dev/Aloha"
  else
    fail "❌ Aloha repo not found"
  fi
}

main() {
  echo -e "\n\033[1;36m🔍 Running system diagnostics...\033[0m\n"
  check_orangepi
  check_os_version
  check_git_config
  check_conda
  check_opencv
  check_ecal_libs
  check_ecal_python
  check_aloha_repo
  echo -e "\n\033[1;36m✅ Diagnostics complete.\033[0m\n"
}

main