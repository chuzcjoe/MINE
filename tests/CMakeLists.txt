file(GLOB test_src "LinearLayerTest/*.cpp")

add_executable(mine_tests ${test_src})

# target_include_directories(mine_tests PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/shaders)

# TODO: expose vulkan_shaders target to tests
# add_dependencies(mine_tests vulkan_shaders)

target_link_libraries(mine_tests PUBLIC gtest mine mat)