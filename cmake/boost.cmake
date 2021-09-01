# ----------------------------------------------------------------------------
#  Detect BOOST libraries
# ----------------------------------------------------------------------------
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
set(BOOST_ROOT /usr/local/cpp_libs/boost)
ocv_clear_internal_cache_vars(BOOST_LIBRARY BOOST_INCLUDE_DIR)
find_package(Boost 1.65 REQUIRED COMPONENTS system date_time regex serialization filesystem)
message("-- Boost_FOUND:${Boost_FOUND},Boost_INCLUDE_DIRS:${Boost_INCLUDE_DIRS},Boost_LIBRARIES:${Boost_LIBRARIES}--")

