#!/usr/bin/env bash
DATA_DIR=/data/tinyvideo
mkdir ${DATA_DIR}
cd ${DATA_DIR}
mkdir beach
mkdir golf
cd beach
wget http://data.csail.mit.edu/videogan/beach.tar.bz2
tar xvjf beach.tar.bz2
cd ../golf
wget http://data.csail.mit.edu/videogan/golf.tar.bz2
tar xvjf golf.tar.bz2
