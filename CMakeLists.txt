cmake_minimum_required(VERSION 3.15)

include(InitGitSubmodule.cmake)
include(CMakePrintHelpers)
if(WIN32)
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()
#Ensure the build uses the install RPATH
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin")
  # Set the RPATH to include a relative path to the shared library directory
  set(CMAKE_INSTALL_RPATH "@loader_path/lib")
elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
  set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)
  set(BUILD_FOR_LINUX TRUE)
endif()
cmake_print_variables(CMAKE_HOST_SYSTEM_NAME)
# Optionally, if you want to append to an existing RPATH
#set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)


project(${SKBUILD_PROJECT_NAME} LANGUAGES C)



find_package(
  Python
  COMPONENTS Interpreter Development.Module
  NumPy
  REQUIRED)


cmake_print_variables(Python_NumPy_INCLUDE_DIRS)
# Use the found Python interpreter
message(STATUS "Python executable: ${Python_EXECUTABLE}")
message(STATUS "Python include dir: ${Python_INCLUDE_DIRS}")
message(STATUS "Python library: ${Python_LIBRARIES}")

python_add_library(incstatspy MODULE "" WITH_SOABI)
target_include_directories(incstatspy PRIVATE ${CMAKE_CURRENT_LIST_DIR}/src/incstats/src/incstats/include ${Python_NumPy_INCLUDE_DIRS})
if(BUILD_FOR_LINUX)
  set_target_properties(incstatspy PROPERTIES INSTALL_RPATH "\$ORIGIN/lib64;\$ORIGIN/lib")
endif()
target_link_libraries(incstatspy PRIVATE incstats Python::NumPy)
install(TARGETS incstatspy DESTINATION .)

if(NOT DEFINED ENV{incstatsPY_EXCLUDE_TESTS})
  find_package(Python COMPONENTS Development.Embed REQUIRED)
  add_executable(test_incstatspy "test/test_incstatspy.c")
  target_include_directories(test_incstatspy PRIVATE ${Python_INCLUDE_DIRS})
  target_link_libraries(test_incstatspy PRIVATE ${Python_LIBRARIES})
else()
  message(STATUS "EXCLUDE_TESTS is defined. Skipping test_incstatspy.")
endif()

init_git_submodule(${CMAKE_CURRENT_LIST_DIR}/src/incstats)
add_subdirectory(src)