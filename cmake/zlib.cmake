# ----------------------------------------------------------------------------
#  Detect 3rd-party image IO libraries
# ----------------------------------------------------------------------------

if(BUILD_ZLIB)
    ocv_clear_vars(ZLIB_FOUND)
else()
    ocv_clear_internal_cache_vars(ZLIB_LIBRARY ZLIB_INCLUDE_DIR)
    find_package(ZLIB "${MIN_VER_ZLIB}")
    if(ZLIB_FOUND AND ANDROID)
        if(ZLIB_LIBRARIES MATCHES "/usr/(lib|lib32|lib64)/libz.so$")
            set(ZLIB_LIBRARIES z)
        endif()
    endif()
endif()

if(NOT ZLIB_FOUND)
    ocv_clear_vars(ZLIB_LIBRARY ZLIB_LIBRARIES ZLIB_INCLUDE_DIR)

    set(ZLIB_LIBRARY zlib CACHE INTERNAL "")
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/zlib")
    set(ZLIB_INCLUDE_DIR "${${ZLIB_LIBRARY}_SOURCE_DIR}" "${${ZLIB_LIBRARY}_BINARY_DIR}" CACHE INTERNAL "")
    set(ZLIB_INCLUDE_DIRS ${ZLIB_INCLUDE_DIR})
    set(ZLIB_LIBRARIES ${ZLIB_LIBRARY})

    ocv_parse_header2(ZLIB "${${ZLIB_LIBRARY}_SOURCE_DIR}/zlib.h" ZLIB_VERSION)
    message(ZLIB_VERSION:${ZLIB_VERSION})
endif()

