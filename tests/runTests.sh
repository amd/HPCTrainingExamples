#!/bin/sh

git clone https://github.com/amd/HPCTrainingExamples.git

mkdir TestExamples
cp /Examples/TestExamples/* TestExamples
cd TestExamples

mkdir build && cd build
cmake ..
make test

                                                                      
