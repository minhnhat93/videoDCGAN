#!/usr/bin/env bash
mkdir out
mkdir checkpoints
cd data
chmod a+x ./download_moving_mnist.sh
./download_moving_mnist.sh
cd ..