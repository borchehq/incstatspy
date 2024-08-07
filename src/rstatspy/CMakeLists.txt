target_sources(rstatspy
    PRIVATE
    rstatspy.c
    rstatspy.h
)

target_include_directories(rstatspy PUBLIC ${CMAKE_CURRENT_LIST_DIR})