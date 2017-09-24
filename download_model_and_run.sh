#!/bin/bash
scp carnd@carnd:/sdc_project3/models/model.m5 ./models/model.m5
bin/python3 drive.py models/model.m5
