#!/bin/bash
echo '****** COORDINATE PLOTS ******'
python plot_IND.py coordinate

echo '****** GAUSSIAN PLOTS ******'
python plot_IND.py gaussian

echo '****** HASHING PLOTS ******'
python plot_IND.py hashing
