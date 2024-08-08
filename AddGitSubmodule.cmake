find_package(Git REQUIRED)

function(add_git_submodule dir)

    if(NOT EXISTS ${dir}/.git)
    execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive -- ${dir}
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        COMMAND_ERROR_IS_FATAL ANY)
    endif()

    add_subdirectory(${dir})

endfunction(add_git_submodule)