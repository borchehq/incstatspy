cmake_minimum_required(VERSION 3.15)
project(${SKBUILD_PROJECT_NAME} LANGUAGES C)

include(InitGitSubmodule.cmake)
include(CMakePrintHelpers)

find_package(
  Python
  COMPONENTS Interpreter Development.Module
  NumPy
  REQUIRED)

cmake_print_variables(Python_NumPy_INCLUDE_DIRS)

python_add_library(rstatspy MODULE "" WITH_SOABI)
target_include_directories(rstatspy PRIVATE ${CMAKE_CURRENT_LIST_DIR}/src/rstats/src/rstats/include ${Python_NumPy_INCLUDE_DIRS})
target_link_libraries(rstatspy PRIVATE rstats Python::NumPy)
install(TARGETS rstatspy DESTINATION .)

init_git_submodule(${CMAKE_CURRENT_LIST_DIR}/src/rstats)
add_subdirectory(src)