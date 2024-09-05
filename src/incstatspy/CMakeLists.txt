target_sources(incstatspy
    PRIVATE
    incstatspy.c
    incstatspy.h
)

target_include_directories(incstatspy PUBLIC ${CMAKE_CURRENT_LIST_DIR})
