# compile vulkan shaders
set(GLSLC ${Vulkan_GLSLC_EXECUTABLE})

file(GLOB_RECURSE shader_src "shaders/vulkan/*.comp")

foreach(shader ${shader_src})
    get_filename_component(shader_name ${shader} NAME)
    set(shader_spv "${CMAKE_CURRENT_BINARY_DIR}/shaders/vulkan/${shader_name}.spv")
    add_custom_command(
        OUTPUT ${shader_spv}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/shaders/vulkan
        COMMAND ${GLSLC} --target-env=vulkan1.1 -mfmt=c ${shader} -o ${shader_spv}
        DEPENDS ${shader}
        COMMENT "Compiling shader ${shader_name} to SPIR-V"
        VERBATIM
    )
    list(APPEND shader_spv_files ${shader_spv})
endforeach(shader)

add_custom_target(mine_vulkan_shaders DEPENDS ${shader_spv_files})

# compile mine library
file(GLOB mine_src "src/*.cpp")

add_library(mine ${mine_src})

target_include_directories(mine
    PUBLIC include
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/shaders/vulkan)

add_dependencies(mine mine_vulkan_shaders)

if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    target_link_libraries(mine
        PUBLIC vulkan mat timer glm)
else()
    target_link_libraries(mine
        PUBLIC
        vulkan
        mat
        timer
        glm)
endif()