#!/bin/bash
cnt=1
for file in `ls -v  *.txt`
do
  mv "$file" $cnt.gt.txt
  let cnt=cnt+1
done
