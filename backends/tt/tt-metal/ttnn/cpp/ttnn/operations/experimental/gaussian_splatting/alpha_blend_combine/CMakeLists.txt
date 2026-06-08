include(sources.cmake)

add_library(ttnn_op_experimental_gaussian_splatting_alpha_blend_combine ${LIB_TYPE})
add_library(
    TTNN::Ops::Experimental::GaussianSplatting::AlphaBlendCombine
    ALIAS ttnn_op_experimental_gaussian_splatting_alpha_blend_combine
)

target_precompile_headers(ttnn_op_experimental_gaussian_splatting_alpha_blend_combine REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_gaussian_splatting_alpha_blend_combine)

set_target_properties(
    ttnn_op_experimental_gaussian_splatting_alpha_blend_combine
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(GLOB_RECURSE kernels device/kernels/*)

target_sources(
    ttnn_op_experimental_gaussian_splatting_alpha_blend_combine
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
        ${TTNN_OP_EXPERIMENTAL_GAUSSIAN_SPLATTING_ALPHA_BLEND_COMBINE_SRCS}
)

target_include_directories(ttnn_op_experimental_gaussian_splatting_alpha_blend_combine PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_gaussian_splatting_alpha_blend_combine
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(
    TARGETS
        ttnn_op_experimental_gaussian_splatting_alpha_blend_combine
    FILE_SET
    kernels
        DESTINATION
            ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/gaussian_splatting/alpha_blend_combine
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_experimental_gaussian_splatting_alpha_blend_combine LIBRARY COMPONENT tar)
