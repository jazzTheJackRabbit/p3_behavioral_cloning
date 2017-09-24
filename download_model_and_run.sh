#!/bin/bash
scp carnd@carnd:~/sdc_project3/model/model.h5 ./model/model.h5
bin/python3 drive.py ./model/model.m5
