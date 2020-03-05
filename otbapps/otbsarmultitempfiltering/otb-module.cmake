set(DOCUMENTATION "Multitemporal Speckle Filtering for SAR")

# define the dependencies of the include module and the tests
otb_module(SARMultiTempFiltering
  DEPENDS
    OTBCommon
	OTBApplicationEngine
  TEST_DEPENDS

  DESCRIPTION
    "${DOCUMENTATION}"
)
