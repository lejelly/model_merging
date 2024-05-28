#!/bin/bash

START_JOB_NID=$1 
END_JOB_NID=$2

for ((id=START_JOB_NID; id<=END_JOB_NID; id++)); do
    qdel $id
done