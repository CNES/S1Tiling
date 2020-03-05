## Documentation step by step

# First load cmake and otb
# module load otb/6.6-python2

## cmake configuration
SET(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE )
SET(OTB_BUILD_MODULE_AS_STANDALONE ON CACHE BOOL "" FORCE)
SET(OTB_DIR "$ENV{OTB_HOME}/lib/cmake/OTB-6.6/" CACHE PATH "" FORCE )
SET(CMAKE_CXX_FLAGS "-std=c++14" CACHE STRING "" FORCE )
SET(CMAKE_INSTALL_PREFIX "/work/OT/biomass/logiciels/module-otb" CACHE STRING "" FORCE )
## Launch cmake configuration
# 
# cd build
# CC=$GCCHOME/bin/gcc CXX=$GCCHOME/bin/g++ cmake -C ../build.cmake ..





