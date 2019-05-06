#!/bin/bash
GAIA_DIR="$( dirname "$( dirname "${BASH_SOURCE[0]}" )" )"

enqueue_compss --lang=python --scheduler=es.bsc.compss.scheduler.fifoDataScheduler.FIFODataScheduler --worker_in_master_cpus=0 --worker_working_dir=gpfs --exec_time=45 --num_nodes=4 $GAIA_DIR/clustering.py -r 1 -p 15000 -o $GAIA_DIR/out $GAIA_DIR/df_tgas_real.csv
