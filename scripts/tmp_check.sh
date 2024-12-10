#!/bin/bash

grad1=0.1
grad2=0.5
grad3=0.1

pjsub -g gb20 -x grad1=${grad1},grad2=${grad2},grad3=${grad3} scripts/exec_grid_search_3params.sh
