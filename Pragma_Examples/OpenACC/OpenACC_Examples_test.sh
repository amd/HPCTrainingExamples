#!/bin/bash
git clone https://github.com/OpenACCUserGroup/OpenACCV-V.git
cd OpenACCV-V

module load og
sed -e "s|^!CC:gcc|CC:/opt/rocmplus-5.7.2/og13-23-12-08/bin/gcc|" \
    -e "s|^!CPP:g++|CPP:/opt/rocmplus-5.7.2/og13-23-12-08/bin/g++|" \
    -e "s|^!FC:gfortran|FC:/opt/rocmplus-5.7.2/og13-23-12-08/bin/gfortran|" \
    -e "/^!CCFlags:-fopenacc -cpp -lm -foffload=-march=native'/s/^!//" \
    -e "/^!CPPFlags:-fopenacc -cpp -lm -foffload=-march=native'/s/^!//" \
    -e "/^!FCFlags:-fopenacc -cpp -lm -foffload=-march=native/s/^!//" \
    -e "s/^!ResultsFormat:json/ResultsFormat:html/" \
    init_config.txt > gcc_config.txt

mkdir gcc_openacc_results
cp -r results_template/* gcc_openacc_results
python3 infrastructure.py -c=gcc_config.txt -o=gcc_openacc_results/results.json
tar -czvf gcc_openacc_results.tgz gcc_openacc_results

module load clacc
sed -e "s|^!CC:gcc|CC:/opt/rocmplus-5.7.2/clacc_clang/bin/clang|" \
    -e "s|^!CCFlags:-fopenacc -cpp -lm -foffload='-lm'|CCFlags:-fopenacc --offload-arch=native|" \
    -e "s|^!ResultsFormat:json|ResultsFormat:html|" \
    init_config.txt > clacc_config.txt

mkdir clacc_openacc_results
cp -r results_template/* clacc_openacc_results
python3 infrastructure.py -c=clacc_config.txt -o=clacc_openacc_results/results.json
tar -czvf clacc_openacc_results.tgz clacc_openacc_results
