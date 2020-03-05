

cmake_minimum_required(VERSION 2.8.9)
project(SARMultiTempFiltering)

#set(DiapOTBModule_LIBRARIES DiapOTBModule)
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

if(NOT OTB_SOURCE_DIR)
    find_package(OTB REQUIRED)
    list(APPEND CMAKE_MODULE_PATH ${OTB_CMAKE_DIR})
    include(OTBModuleExternal)
else()
    otb_module_impl()
endif()

