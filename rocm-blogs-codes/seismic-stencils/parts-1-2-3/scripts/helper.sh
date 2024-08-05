analyze_and_copy() {
    mkdir -p $1

    # Generate .csv and .toml output files using XR. 
    # -k Only capture the relevant kernels
    # -s 1 : Skip first occurrence
    xr -k _z -s 1 -T $1/results.toml -xls > $1/summary.csv
    xr -k _z -s 1 > $1/summary.txt
    # Copy rocprof results and log file
    cp results* $1/
    cp log.txt $1/
}

run() {
  begin "Task: sliding_window_gpu_R_${radius}.x $nx $ny $nz $nt $nw $align $use_offset "
  scripts/profile.sh sliding_window_gpu_R_${radius}_vec_${vec}.x $nx $ny $nz $nt $nw $align $use_offset
  analyze_and_copy logs/$study/${nx}x${ny}x${nz}/${name}_${radius}_align_${align}_nw_${nw}_use_offset_${use_offset}/
  end
}

vary_nw() {
  for (( nw=$1; nw < $2; nw+= $3 ));
  do 
      echo "nw=$nw"
      time run
  done

}

vary_grid_size() {
  for (( n=$1; n < $2; n+= $3 ));
  do 
      nx=$n
      ny=$n
      nz=$n
      nw=$n
      echo "n=$nw"
      time run
  done

}

begin () {
    msg=$1
    echo "========================================================================================="
    echo $msg
    st=$SECONDS
}

end() {
    echo "Done" $msg
    echo "Task took:" $(( SECONDS - st)) "s"
    echo "========================================================================================="
}


