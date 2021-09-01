message("-- building core utils for modules,source dir:${CMAKE_CURRENT_SOURCE_DIR}")
message("-- building core with opencv libraries:${OPENCV_LIBRARIES}")
set(DL_CORE_LIBRARY dl_core CACHE INTERNAL "")
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
file(GLOB SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)

add_library(${DL_CORE_LIBRARY} SHARED ${SOURCE_FILES})
target_link_libraries(${DL_CORE_LIBRARY} ${TORCH_LIBRARIES} ${OPENCV_LIBRARIES} ${JSON_LIBRARIES})
set_property(TARGET ${DL_CORE_LIBRARY} PROPERTY CXX_STANDARD 17)

set(DL_CORE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(DL_CORE_INCLUDE_DIR "${DL_CORE_SOURCE_DIR}/include" CACHE INTERNAL "")

install(FILES ${HEADER_FILES}
        DESTINATION ${CMAKE_INSTALL_PREFIX}/include COMPONENT dev)
install(TARGETS ${DL_CORE_LIBRARY}
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib COMPONENT libs)
# If the PARENT_SCOPE option is given the variable will be set in the scope
# above the current scope. Each new directory or function creates a new scope.
set(DL_CORE_INCLUDE_DIRS ${DL_CORE_INCLUDE_DIR} PARENT_SCOPE)
set(DL_CORE_LIBRARIES ${DL_CORE_LIBRARY} PARENT_SCOPE)

message("-- core utils config with library name:${DL_CORE_LIBRARY}")
