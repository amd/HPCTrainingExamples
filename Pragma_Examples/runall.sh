#!/bin/bash

for compiler in aomp gcc sourcery
do
   source ${compiler}_env
   ./runtest.sh

   echo "========================"       >  ${compiler}_result.txt
   echo "For ${compiler} compiler"       >> ${compiler}_result.txt
   echo "========================"       >> ${compiler}_result.txt
   grep succeded */*/*/*/result.txt      >> ${compiler}_result.txt
   grep Ran */*/*/*/result.txt           >> ${compiler}_result.txt
   grep Runtime */*/*/*/*_run.out        >> ${compiler}_result.txt
   echo ""                               >> ${compiler}_result.txt
   grep failed */*/*/*/result.txt        >> ${compiler}_result.txt
   grep "Did not run" */*/*/*/result.txt >> ${compiler}_result.txt
   echo ""                               >> ${compiler}_result.txt

   source ${compiler}_unset
done
cat *_result.txt
rm -f */*/*/*/result.txt *_result.txt
rm -f */*/*/*/*_run.out */*/*/*/*_stderr.out
