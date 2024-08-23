#!/bin/bash

# This test checks that hpcviewer 
# returns the version

module purge

module load hpctoolkit
hpcviewer --version
