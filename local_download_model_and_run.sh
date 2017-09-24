#!/bin/bash
scp carnd@carnd:~/sdc_project3/model/model.h5 ./model/model.h5
./bin/python drive.py model/model.h5
