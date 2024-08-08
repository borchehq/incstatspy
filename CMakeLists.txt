cmake_minimum_required(VERSION 3.15)
project(${SKBUILD_PROJECT_NAME} LANGUAGES C)

include(AddGitSubmodule.cmake)

find_package(
  Python
  COMPONENTS Interpreter Development.Module
  REQUIRED)

python_add_library(rstatspy MODULE "" WITH_SOABI)
target_include_directories(rstatspy PRIVATE ${CMAKE_CURRENT_LIST_DIR}/src/rstats/src/rstats/include)
target_link_libraries(rstatspy PRIVATE rstats)
install(TARGETS rstatspy DESTINATION .)

add_git_submodule(${CMAKE_CURRENT_LIST_DIR}/src/rstats)
add_subdirectory(src)