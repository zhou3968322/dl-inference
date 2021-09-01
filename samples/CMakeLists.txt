message("-- get core include dirs:${DL_CORE_INCLUDE_DIRS},libraries:${DL_CORE_LIBRARIES},${TEXT_DETECTION_LIBRARIES}")

include_directories(${DL_CORE_INCLUDE_DIRS} ${TEXT_DETECTION_INCLUDE_DIRS})

add_executable(throw_exception ${CMAKE_CURRENT_SOURCE_DIR}/throw_exception.cpp)
target_link_libraries(throw_exception)
set_property(TARGET throw_exception PROPERTY CXX_STANDARD 17)

add_executable(shuffle_get_file ${CMAKE_CURRENT_SOURCE_DIR}/shuffle_get_file.cpp)
target_link_libraries(shuffle_get_file ${DL_CORE_LIBRARIES})
set_property(TARGET shuffle_get_file PROPERTY CXX_STANDARD 17)

add_executable(tensor_interpolate ${CMAKE_CURRENT_SOURCE_DIR}/tensor_interpolate.cpp)
target_link_libraries(tensor_interpolate ${DL_CORE_LIBRARIES} ${TORCH_LIBRARIES})
set_property(TARGET tensor_interpolate PROPERTY CXX_STANDARD 17)

add_executable(test_perspective ${CMAKE_CURRENT_SOURCE_DIR}/test_perspective.cpp)
target_link_libraries(test_perspective ${TORCH_LIBRARIES} ${TEXT_DETECTION_LIBRARIES} ${DL_CORE_LIBRARIES}
        ${JSON_LIBRARIES} ${OPENCV_LIBRARIES} opencv_calib3d)
set_property(TARGET test_perspective PROPERTY CXX_STANDARD 17)


#add_executable(tensor_solve_linear ${CMAKE_CURRENT_SOURCE_DIR}/tensor_solve_linear.cpp)
#target_link_libraries(tensor_solve_linear ${DL_CORE_LIBRARIES} ${TORCH_LIBRARIES})
#set_property(TARGET tensor_solve_linear PROPERTY CXX_STANDARD 17)

message("-- samples set finished")


