#!/bin/bash

# This test checks that hpcviewer
# returns the version

module load rocm

module load hpctoolkit
hpcviewer --version
