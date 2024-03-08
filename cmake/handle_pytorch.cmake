get_target_property(python3_interpreter_path Python::Interpreter IMPORTED_LOCATION)
execute_process(COMMAND ${python3_interpreter_path} "-c" "import torch; print(torch.utils.cmake_prefix_path)" OUTPUT_VARIABLE pytorch_cmake_prefix_path)
message(STATUS "pytorch cmake prefix path: ${pytorch_cmake_prefix_path}")
find_package(Torch CONFIG REQUIRED PATHS "${pytorch_cmake_prefix_path}/../../")
