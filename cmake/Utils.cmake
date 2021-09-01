# all below are from opencv cmake files

macro(ocv_update VAR)
    if(NOT DEFINED ${VAR})
        if("x${ARGN}" STREQUAL "x")
            set(${VAR} "")
        else()
            set(${VAR} ${ARGN})
        endif()
    else()
        #ocv_debug_message("Preserve old value for ${VAR}: ${${VAR}}")
    endif()
endmacro()


macro(ocv_clear_vars)
    foreach(_var ${ARGN})
        unset(${_var})
        unset(${_var} CACHE)
    endforeach()
endmacro()

# Clears passed variables with INTERNAL type from CMake cache
macro(ocv_clear_internal_cache_vars)
    foreach(_var ${ARGN})
        get_property(_propertySet CACHE ${_var} PROPERTY TYPE SET)
        if(_propertySet)
            get_property(_type CACHE ${_var} PROPERTY TYPE)
            if(_type STREQUAL "INTERNAL")
                message("Cleaning INTERNAL cached variable: ${_var}")
                unset(${_var} CACHE)
            endif()
        endif()
    endforeach()
    unset(_propertySet)
    unset(_type)
endmacro()

# read single version define from the header file
macro(ocv_parse_header2 LIBNAME HDR_PATH VARNAME)
    ocv_clear_vars(${LIBNAME}_VERSION_MAJOR
            ${LIBNAME}_VERSION_MAJOR
            ${LIBNAME}_VERSION_MINOR
            ${LIBNAME}_VERSION_PATCH
            ${LIBNAME}_VERSION_TWEAK
            ${LIBNAME}_VERSION_STRING)
    set(${LIBNAME}_H "")
    if(EXISTS "${HDR_PATH}")
        file(STRINGS "${HDR_PATH}" ${LIBNAME}_H REGEX "^#define[ \t]+${VARNAME}[ \t]+\"[^\"]*\".*$" LIMIT_COUNT 1)
    endif()

    if(${LIBNAME}_H)
        string(REGEX REPLACE "^.*[ \t]${VARNAME}[ \t]+\"([0-9]+).*$" "\\1" ${LIBNAME}_VERSION_MAJOR "${${LIBNAME}_H}")
        string(REGEX REPLACE "^.*[ \t]${VARNAME}[ \t]+\"[0-9]+\\.([0-9]+).*$" "\\1" ${LIBNAME}_VERSION_MINOR  "${${LIBNAME}_H}")
        string(REGEX REPLACE "^.*[ \t]${VARNAME}[ \t]+\"[0-9]+\\.[0-9]+\\.([0-9]+).*$" "\\1" ${LIBNAME}_VERSION_PATCH "${${LIBNAME}_H}")
        set(${LIBNAME}_VERSION_MAJOR ${${LIBNAME}_VERSION_MAJOR} ${ARGN})
        set(${LIBNAME}_VERSION_MINOR ${${LIBNAME}_VERSION_MINOR} ${ARGN})
        set(${LIBNAME}_VERSION_PATCH ${${LIBNAME}_VERSION_PATCH} ${ARGN})
        set(${LIBNAME}_VERSION_STRING "${${LIBNAME}_VERSION_MAJOR}.${${LIBNAME}_VERSION_MINOR}.${${LIBNAME}_VERSION_PATCH}")

        # append a TWEAK version if it exists:
        set(${LIBNAME}_VERSION_TWEAK "")
        if("${${LIBNAME}_H}" MATCHES "^.*[ \t]${VARNAME}[ \t]+\"[0-9]+\\.[0-9]+\\.[0-9]+\\.([0-9]+).*$")
            set(${LIBNAME}_VERSION_TWEAK "${CMAKE_MATCH_1}" ${ARGN})
        endif()
        if(${LIBNAME}_VERSION_TWEAK)
            set(${LIBNAME}_VERSION_STRING "${${LIBNAME}_VERSION_STRING}.${${LIBNAME}_VERSION_TWEAK}" ${ARGN})
        else()
            set(${LIBNAME}_VERSION_STRING "${${LIBNAME}_VERSION_STRING}" ${ARGN})
        endif()
    endif()
endmacro()


# get_property command to retrieve the value of the directory property

#get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
#foreach(dir ${dirs})
#    message(STATUS "dir='${dir}'")
#endforeach()