if [ ! -d "./blender" ]; then
  echo "Installing Blender (requires root access).."
  # Download Blender
  wget -O 'blender.tar.bz2' -nc "https://ftp.halifax.rwth-aachen.de/blender/release/Blender2.81/blender-2.81-linux-glibc217-x86_64.tar.bz2"

  # Unpacking tarball
  mkdir blender
  tar -xf 'blender.tar.bz2' -C ./blender --strip-components=1

  sudo apt-get install -y libboost-all-dev libgl1-mesa-dev libglu1-mesa libsm-dev
fi