#!/bin/sh
if [ $# -eq 1 ]
then
  #cd $1/
  echo "Started running verification for $1 models."
  python3 $1
  cd ..
fi