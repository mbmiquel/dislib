#!/bin/bash
GAIA_DIR="$( dirname "$( dirname "${BASH_SOURCE[0]}" )" )"

runcompss --python_interpreter=python3 --pythonpath=$GAIA_DIR $GAIA_DIR/clustering.py -r 1 -p 15000 -o $GAIA_DIR/out $GAIA_DIR/df_tgas_real.csv
