#!/bin/bash
echo '****** COORDINATE PLOTS ******'
python BCGN_IND.py plot coordinate

echo '****** COORDINATE METRICS ******'
python BCGN_IND.py metrics coordinate

echo '****** GAUSSIAN PLOTS ******'
python BCGN_IND.py plot gaussian

echo '****** GAUSSIAN METRICS ******'
python BCGN_IND.py metrics gaussian

echo '****** HASHING PLOTS ******'
python BCGN_IND.py plot hashing

echo '****** HASHING METRICS ******'
python BCGN_IND.py metrics hashing
