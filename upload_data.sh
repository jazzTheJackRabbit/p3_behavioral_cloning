#!/bin/bash
zip -r ./data.zip data
scp data.zip carnd@carnd:/sdc_project3/
