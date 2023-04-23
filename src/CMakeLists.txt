
# directories
set (LIBELAS_SRC_DIR ${PROJECT_SOURCE_DIR}/thridparty/elas)

# include directory
include_directories("${LIBELAS_SRC_DIR}")

# sources
FILE(GLOB LIBELAS_SRC_FILES "../thridparty/elas/*.cpp")

add_library(zzslam SHARED
        frame.cpp
        mappoint.cpp
        map.cpp
        camera.cpp
        feature.cpp
        frontend.cpp
        backend.cpp
        viewer.cpp
        vo_system.cpp
        ${LIBELAS_SRC_FILES})

target_link_libraries(zzslam ${THIRD_PARTY_LIBS})



