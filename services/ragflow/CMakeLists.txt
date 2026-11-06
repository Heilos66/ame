add_executable(ame_worker worker.cpp)
target_link_libraries(ame_worker PRIVATE ame_engine)

find_package(Python3 3.10 REQUIRED COMPONENTS Interpreter)
add_test(NAME test_ragflow_service
  COMMAND "${Python3_EXECUTABLE}" -B -m unittest discover
    -s "${CMAKE_CURRENT_SOURCE_DIR}/tests" -v)
set_tests_properties(test_ragflow_service PROPERTIES
  ENVIRONMENT "AME_TEST_WORKER=$<TARGET_FILE:ame_worker>;PYTHONDONTWRITEBYTECODE=1"
  TIMEOUT 60)
