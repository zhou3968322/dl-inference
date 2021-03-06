message("-- building text recognition module,source dir:${CMAKE_CURRENT_SOURCE_DIR}")
message("-- building text recognition module,core include dir:${DL_CORE_INCLUDE_DIRS}")
message("-- building with json libraries:${JSON_LIBRARIES}")
message("-- building with opencv libraries:${OPENCV_LIBRARIES}")
message("-- building with dl core libraries:${DL_CORE_LIBRARY}")
set(TEXT_RECOGNITION_LIBRARY text_recognition CACHE INTERNAL "")
include_directories(include ${DL_CORE_INCLUDE_DIRS})

file(GLOB SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)
add_library(${TEXT_RECOGNITION_LIBRARY} SHARED ${SOURCE_FILES})
target_link_libraries(${TEXT_RECOGNITION_LIBRARY} ${TORCH_LIBRARIES} ${DL_CORE_LIBRARY} ${OPENCV_LIBRARIES} ${JSON_LIBRARIES})
#target_compile_definitions(${TEXT_RECOGNITION_LIBRARY} PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
set_property(TARGET ${TEXT_RECOGNITION_LIBRARY} PROPERTY CXX_STANDARD 17)

set(TEXT_RECOGNITION_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(TEXT_RECOGNITION_INCLUDE_DIR "${TEXT_RECOGNITION_SOURCE_DIR}/include" CACHE INTERNAL "")

install(FILES ${HEADER_FILES}
        DESTINATION ${CMAKE_INSTALL_PREFIX}/include COMPONENT dev)
install(TARGETS ${TEXT_RECOGNITION_LIBRARY}
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib COMPONENT libs)

set(TEXT_RECOGNITION_INCLUDE_DIRS ${TEXT_RECOGNITION_INCLUDE_DIR} PARENT_SCOPE)
set(TEXT_RECOGNITION_LIBRARIES ${TEXT_RECOGNITION_LIBRARY} PARENT_SCOPE)

message("-- text recognition config with library name:${TEXT_RECOGNITION_LIBRARY}")