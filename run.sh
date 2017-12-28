#! /bin/sh
echo "\n==== Running Q1.tif ===="
time python fft.py --filename Q1.tif

echo "\n==== Running Q2.tif ===="
time python fft.py --filename Q2.tif

echo "\n==== Running Q3.tif ===="
time python fft.py --filename Q3.tif

echo "\n==== Running Q4.tif ===="
time python fft.py --filename Q4.tif
