message("-- build tests get text detection include dirs:${TEXT_DETECTION_INCLUDE_DIRS},libraries:${TEXT_DETECTION_LIBRARIES}")
message("-- build tests get core include dirs:${DL_CORE_INCLUDE_DIRS},libraries:${DL_CORE_LIBRARIES}")
message("-- build tests get text recognition include dirs:${TEXT_RECOGNITION_INCLUDE_DIRS},libraries:${TEXT_RECOGNITION_LIBRARIES}")


include_directories(${DL_CORE_INCLUDE_DIRS} ${TEXT_DETECTION_INCLUDE_DIRS} ${TEXT_RECOGNITION_INCLUDE_DIRS})

add_executable(test_core ${CMAKE_CURRENT_SOURCE_DIR}/test_core.cpp)
target_link_libraries(test_core ${TORCH_LIBRARIES} ${DL_CORE_LIBRARIES} ${JSON_LIBRARIES} ${OPENCV_LIBRARIES})
set_property(TARGET test_core PROPERTY CXX_STANDARD 17)

add_executable(test_config ${CMAKE_CURRENT_SOURCE_DIR}/test_config.cpp)
target_link_libraries(test_config ${TORCH_LIBRARIES} ${TEXT_DETECTION_LIBRARIES} ${DL_CORE_LIBRARIES}
        ${TEXT_RECOGNITION_LIBRARIES} ${JSON_LIBRARIES} ${OPENCV_LIBRARIES})
set_property(TARGET test_config PROPERTY CXX_STANDARD 17)

add_executable(test_load_model ${CMAKE_CURRENT_SOURCE_DIR}/test_load_model.cpp)
target_link_libraries(test_load_model ${TORCH_LIBRARIES} ${TEXT_DETECTION_LIBRARIES} ${DL_CORE_LIBRARIES}
        ${TEXT_RECOGNITION_LIBRARIES} ${JSON_LIBRARIES} ${OPENCV_LIBRARIES})
set_property(TARGET test_load_model PROPERTY CXX_STANDARD 17)

add_executable(test_text_detection ${CMAKE_CURRENT_SOURCE_DIR}/test_text_detection.cpp)
target_link_libraries(test_text_detection ${TORCH_LIBRARIES} ${TEXT_DETECTION_LIBRARIES} ${DL_CORE_LIBRARIES}
        ${JSON_LIBRARIES} ${OPENCV_LIBRARIES})
set_property(TARGET test_text_detection PROPERTY CXX_STANDARD 17)

add_executable(test_text_recognition ${CMAKE_CURRENT_SOURCE_DIR}/test_text_recognition.cpp)
target_link_libraries(test_text_recognition ${TORCH_LIBRARIES} ${TEXT_RECOGNITION_LIBRARIES} ${DL_CORE_LIBRARIES}
        ${JSON_LIBRARIES} ${OPENCV_LIBRARIES})
set_property(TARGET test_text_recognition PROPERTY CXX_STANDARD 17)

include_directories(${OCR_INCLUDE_DIRS})
add_executable(test_ocr ${CMAKE_CURRENT_SOURCE_DIR}/test_ocr.cpp)
target_link_libraries(test_ocr ${TORCH_LIBRARIES} ${TEXT_RECOGNITION_LIBRARIES} ${TEXT_DETECTION_LIBRARIES}
        ${DL_CORE_LIBRARIES} ${JSON_LIBRARIES} ${OPENCV_LIBRARIES} ${OCR_LIBRARIES})
set_property(TARGET test_ocr PROPERTY CXX_STANDARD 17)

install(TARGETS test_core test_config test_load_model test_text_detection test_text_recognition test_ocr
        DESTINATION ${CMAKE_INSTALL_PREFIX}/tests COMPONENT tests)

message("-- test config cmake set finished")


