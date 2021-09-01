message("-- building ocr application,source dir:${CMAKE_CURRENT_SOURCE_DIR}")
message("-- building with opencv libraries:${OPENCV_LIBRARIES}")
message("-- building with dl core libraries:${DL_CORE_LIBRARY}")
message("-- building with text recognition libraries:${TEXT_RECOGNITION_LIBRARY}")
message("-- building with text detection libraries:${TEXT_DETECTION_LIBRARY}")
set(OCR_LIBRARY ocr CACHE INTERNAL "")
include_directories(include ${DL_CORE_INCLUDE_DIRS} ${TEXT_DETECTION_INCLUDE_DIRS}  ${TEXT_RECOGNITION_INCLUDE_DIRS})
file(GLOB SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB HEADER_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h)
add_library(${OCR_LIBRARY} SHARED ${SOURCE_FILES})
target_link_libraries(${OCR_LIBRARY} ${TORCH_LIBRARIES} ${DL_CORE_LIBRARY} ${OPENCV_LIBRARIES}
        ${JSON_LIBRARIES} ${TEXT_RECOGNITION_LIBRARY} ${TEXT_DETECTION_LIBRARY})
#target_compile_definitions(${TEXT_RECOGNITION_LIBRARY} PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
set_property(TARGET ${OCR_LIBRARY} PROPERTY CXX_STANDARD 17)

set(OCR_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(OCR_INCLUDE_DIR "${OCR_SOURCE_DIR}/include" CACHE INTERNAL "")
set(OCR_INCLUDE_DIRS ${OCR_INCLUDE_DIR} PARENT_SCOPE)
set(OCR_LIBRARIES ${OCR_LIBRARY} PARENT_SCOPE)

install(FILES ${HEADER_FILES}
        DESTINATION ${CMAKE_INSTALL_PREFIX}/include COMPONENT dev)
install(TARGETS ${OCR_LIBRARY}
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib COMPONENT libs)

message("-- ocr config with library name:${OCR_LIBRARY}")