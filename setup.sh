#!/bin/sh

macOS=false
current_dir="$(pwd -P)"

check_requirements() {
  case $(uname -s) in
    Darwin)
      printf "Installing on macOS\n"
      export CFLAGS='-stdlib=libc++'
      macOS=true
      ;;
    Linux)
      printf "Installing on Linux\n"
      ;;
    *)
      printf "Only Linux and macOS are currently supported.\n"
      exit 1
      ;;
  esac
}

install_python_packages() {
  printf "\nInstall conda packages...\n"
  if [ $macOS = "true" ]; then
    conda install pytorch torchvision torchaudio -c pytorch -y
  else
    conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch -y
  fi
  printf "\nInstall pip packages..."
  pip install -r requirements.txt
}

set_python_path() {
  printf "\nSet conda environment variables...\n"
  conda env config vars set PYTHONPATH=$PYTHONPATH:$current_dir
}

check_requirements
install_python_packages
set_python_path

printf '\nSetup completed. Please restart your conda environment\n'