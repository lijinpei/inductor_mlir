find_package(Python 3 COMPONENTS Interpreter Development REQUIRED)
add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/pybind11)
define_property(GLOBAL PROPERTY PYTHON_WRAPPER_SRCS)

function(add_python_wrapper_src)
	list(TRANSFORM ARGV APPEND ${CMAKE_CURRENT_SOURCE_DIR})
	set_property(GLOBAL APPEND PROPERTY PYTHON_WRAPPER_SRCS "${ARGV}")
endfunction()
