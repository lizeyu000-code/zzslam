# SET(TEST_SOURCES test_triangulation)

# FOREACH (test_src ${TEST_SOURCES})
#     ADD_EXECUTABLE(${test_src} ${test_src}.cpp)
#     TARGET_LINK_LIBRARIES(${test_src} ${THIRD_PARTY_LIBS} myslam)
#     ADD_TEST(${test_src} ${test_src})
# ENDFOREACH (test_src)


add_executable(test_triangulation test_triangulation.cpp)
target_link_libraries(test_triangulation zzslam ${THIRD_PARTY_LIBS})

# add_executable(sparse_ssd_match sparse_ssd_match.cpp)
# target_link_libraries(test_triangulation zzslam ${THIRD_PARTY_LIBS})

