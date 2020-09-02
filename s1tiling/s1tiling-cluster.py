#!/usr/bin/env python
#-*- coding: utf-8 -*-

import ConfigParser
import shutil
import sys
import fileinput
import os

if len(sys.argv) != 2:
    print "Usage: "+sys.argv[0]+" config.cfg"
    sys.exit(1)

CFG = sys.argv[1]
config = ConfigParser.SafeConfigParser()
config.read(CFG)


raw_directory = config.get('Paths', 'S1Images')

output_preprocess = config.get('Paths', 'Output')

tiles_list = [s.strip() for s in config.get('Processing','Tiles').split(",")]
try:
    os.remove(os.path.join("./jobs","*.cfg"))
except:
    pass

for itile,tile in enumerate(tiles_list):
    cfgFilename=os.path.join("./jobs","job-"+str(itile+1)+".cfg")
    if not os.path.exists("./jobs"):
        os.mkdir("./jobs")
    config.set("Processing","Tiles",tile)
    config.set("PEPS","ROI_by_tiles","ALL")
    with open(cfgFilename, 'wb') as configfile:
        config.write(configfile)
    print itile," ",tile,"->" ,cfgFilename
with open("s1tiling.jobarray.template") as f:
    newtext=f.read().replace("#PBS -J","#PBS -J 1:"+str(len(tiles_list))+":1")
with open("s1tiling.jobarray","w") as f:
    f.write(newtext)
