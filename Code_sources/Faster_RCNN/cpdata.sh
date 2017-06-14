#!/bin/bash

dhome=$HOME/2017/Workspace/Fevrier/CodeSource/AnnotationTool/AnnotationTool/DATASET/permis
i=0
mkdir Annotations
mkdir Images
mkdir ImageSets
while [ $i -lt 4 ]
do
    echo "processing folder $dhome/$i"
    cp $dhome/$i/*.png Images
    cp $dhome/$i/*.xml Annotations
    let "i=i+1"
done
ls Annotations/ -m | sed s/\\s/\\n/g | sed s/.xml//g | sed s/,//g > ImageSets/train.txt