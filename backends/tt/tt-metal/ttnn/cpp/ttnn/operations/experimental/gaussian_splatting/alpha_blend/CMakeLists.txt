include(sources.cmake)

add_library(ttnn_op_experimental_gaussian_splatting_alpha_blend ${LIB_TYPE})
add_library(
    TTNN::Ops::Experimental::GaussianSplatting::AlphaBlend
    ALIAS ttnn_op_experimental_gaussian_splatting_alpha_blend
)

target_precompile_headers(ttnn_op_experimental_gaussian_splatting_alpha_blend REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_gaussian_splatting_alpha_blend)

set_target_properties(
    ttnn_op_experimental_gaussian_splatting_alpha_blend
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(GLOB_RECURSE kernels device/kernels/*)

target_sources(
    ttnn_op_experimental_gaussian_splatting_alpha_blend
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        ${TTNN_OP_EXPERIMENTAL_GAUSSIAN_SPLATTING_ALPHA_BLEND_SRCS}
)

target_include_directories(ttnn_op_experimental_gaussian_splatting_alpha_blend PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_gaussian_splatting_alpha_blend
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(
    TARGETS
        ttnn_op_experimental_gaussian_splatting_alpha_blend
    FILE_SET
    kernels
        DESTINATION
            ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/gaussian_splatting/alpha_blend
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_experimental_gaussian_splatting_alpha_blend LIBRARY COMPONENT tar)
