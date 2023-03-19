


add_library(zzslam SHARED
        frame.cpp
        mappoint.cpp
        map.cpp
        camera.cpp
        feature.cpp
        frontend.cpp
        backend.cpp
        viewer.cpp
        vo_system.cpp)

target_link_libraries(zzslam ${THIRD_PARTY_LIBS})



