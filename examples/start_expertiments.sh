!/bin/bash

  # Define script variables

  scriptDir=$(pwd)/$(dirname $0)
  execFile=${scriptDir}/test_dds.py
  appClasspath=${scriptDir}/
  appPythonpath=${scriptDir}/

for size in 1;
do

  size_dir=$scriptDir/experiments/$size
  mkdir -p $size_dir
  cd $size_dir

  file_name=/gpfs/projects/bsc19/COMPSs_DATASETS/csvm/mnist/partitions/1024/3
  # 104857600 67108864 268435456 
  for chunk_size in 1;
  do
    chunk_dir=$size_dir/$chunk_size
    mkdir -p $chunk_dir
    cd $chunk_dir

    for i in 2;
    do
      node_dir="$chunk_dir/$i"
      mkdir -p $node_dir
      cd $node_dir

      # Retrieve arguments
      numNodes=$i
      executionTime="$((15-$i))"
      tracing=true
      graph=false

      comma="enqueue_compss \
                -d \
                --qos=debug \
                --job_dependency=None \
                --num_nodes=$numNodes \
                --exec_time=$executionTime \
                --tracing=$tracing \
                --graph=$graph \
                --classpath=$appClasspath \
                --pythonpath=$appPythonpath \
                --lang=python \
                --worker_in_master_cpus=0 \
                $execFile $file_name"
      echo "$comma" > test.txt
      $comma
      sleep 2
    done
  done
done

