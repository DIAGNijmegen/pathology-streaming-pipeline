#!/bin/sh

# echo '=========== Installing wandb'
# echo 'user' | sudo -S apt-get install pv > /dev/null
# echo 'user' | sudo -S pip3.7 install recordclass==0.12.1 > /dev/null
echo 'user' | sudo -S pip3.7 install wandb > /dev/null
# echo 'user' | sudo -S pip3.7 install bcolz > /dev/null

echo '=========== Installing vips'
cd /home/user/
echo 'Downloading'
wget -q https://github.com/libvips/libvips/releases/download/v8.9.0/vips-8.9.0.tar.gz
echo 'Install deps'
echo 'user' | sudo -S apt-get update > /dev/null 2>&1
echo 'user' | sudo -S apt-get install -y libjpeg-turbo8-dev > /dev/null 2>&1
echo 'user' | sudo -S apt-get install -y libgtk2.0-dev > /dev/null 2>&1
tar xf vips-8.9.0.tar.gz > /dev/null 2>&1
cd vips-8.9.0
echo 'Configure'
./configure > /dev/null 2>&1
echo 'Make'
make > /dev/null 2>&1
echo 'Install'
echo 'user' | sudo -S make install > /dev/null 2>&1
cd ..
echo 'user' | sudo -S ldconfig > /dev/null 2>&1

echo '=========== Installing pyvips'
echo 'user' | sudo -S pip3.7 install pyvips > /dev/null 2>&1

echo '=========== Installing pytorch nightly'
echo 'user' | sudo -S pip3.7 install -U --pre torch torchvision -f https://download.pytorch.org/whl/nightly/cu101/torch_nightly.html > /dev/null 2>&1

export PYTHONPATH=$PYTHONPATH:/mnt/netcache/pathology/projects/pathology-streaming-pipeline
export OMP_NUM_THREADS=3
export WANDB_DISABLE_CODE=true

if [ -z "$1" ]; then
    echo "No argument supplied"
else
    echo "=========== Execute command: ${@}"
    cd /mnt/netcache/pathology/projects/pathology-streaming-pipeline
    python3.7 streaming/train_remote.py ${@}
fi

