#!/bin/bash
for year in {2014..2024}
do
    echo "downloading data for $year"
    python src/download_data.py $year
done 

echo "starting data processing" 
python src/processdat.py


