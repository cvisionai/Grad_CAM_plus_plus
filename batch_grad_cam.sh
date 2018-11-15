#!/bin/bash


#SPECIES_DIR=/mnt/md0/Projects/FathomNet/Data_Files/vars_images_metadata/vars_images/
SPECIES_DIR=/mnt/md0/Projects/FathomNet/Data_Files/
#SPECIES_LIST=(Aegina Apolemia Atolla Bathochordaeus Bathocyroe Beroe) 
#SPECIES_LIST=(Bathyraja Chionoecetes Coryphaenoides Keratoisis Merluccius Pannychia Paragorgia Peniagone Rathbunaster Sebastes Sebastolobus Strongylocentrotus)
#SPECIES_LIST=(Coryphaenoides)
SPECIES_LIST=(media_lab_demo/Midwater_Demo)
#SPECIES_LIST=(Aegina Atolla Bathocyroe Beroe Erenna Eusergestes Lampocteis Llyria Nanomia Poeobius Poralia Solmissus Tomopteris) 
#SPECIES_LIST=(Erenna Eusergestes Lampocteis Llyria Nanomia Poeobius Poralia Solmissus Tomopteris)
#SPECIES_LIST=(Bathyraja Chionoecetes Coryphaenoides)
OUTDIR=/

for i in ${SPECIES_LIST[*]}
do
  echo $i
  python -W ignore classify_batch.py -gpu 1 -l 2 -d ${SPECIES_DIR}${i}/ -o ${OUTDIR}/${i}/ 
done

