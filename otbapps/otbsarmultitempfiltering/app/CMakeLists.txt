otb_create_application(
  NAME           MultitempFiltering
  SOURCES        otbMultitempFiltering.cxx
  LINK_LIBRARIES ${${otb-module}_LIBRARIES})

otb_create_application(
  NAME           MultitempFilteringOutcore
  SOURCES        otbMultitempFilteringOutcore.cxx
  LINK_LIBRARIES ${${otb-module}_LIBRARIES})

otb_create_application(
  NAME           MultitempFilteringFilter
  SOURCES        otbMultitempFilteringFilter.cxx
  LINK_LIBRARIES ${${otb-module}_LIBRARIES})
