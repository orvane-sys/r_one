cmake_minimum_required(VERSION 3.5)
project(r_one)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Install Python module package
ament_python_install_package(${PROJECT_NAME})

if(BUILD_TESTING)
  set(ENV{PYTHONNOUSERSITE} "1")

  find_package(ament_cmake_pytest QUIET)
  if(ament_cmake_pytest_FOUND)
    ament_add_pytest_test(spatial_perception_tests test/test_spatial_perception.py
      ENV PYTHONNOUSERSITE=1
    )
    ament_add_pytest_test(launch_integration_tests test/test_launch_integration.py
      ENV PYTHONNOUSERSITE=1
    )
  endif()

  find_package(ament_cmake_flake8 QUIET)
  if(ament_cmake_flake8_FOUND)
    ament_flake8(r_one launch test)
  endif()

  find_package(ament_cmake_pep257 QUIET)
  if(ament_cmake_pep257_FOUND)
    ament_pep257(r_one launch test)
  endif()

  find_package(ament_cmake_xmllint QUIET)
  if(ament_cmake_xmllint_FOUND)
    ament_xmllint(package.xml)
  endif()

  find_package(ament_cmake_lint_cmake QUIET)
  if(ament_cmake_lint_cmake_FOUND)
    ament_lint_cmake()
  endif()
endif()

# Install non-code asset directories
install(
  DIRECTORY config description launch worlds
  DESTINATION share/${PROJECT_NAME}
)

# Install standard ROS 2 CLI executables (ros2 run r_one <node_name>)
install(
  PROGRAMS
    scripts/spatial_vision_node
    scripts/situational_awareness_node
    scripts/sensor_fusion_node
    scripts/mock_ai_node
    scripts/record_sample_bag.sh
    scripts/download_sample_bag.sh
  DESTINATION lib/${PROJECT_NAME}
)

# Install backwards-compatible legacy script aliases
install(
  PROGRAMS
    FakeSceneUnderStanding/spatial_vision_node.py
    FakeSceneUnderStanding/real_ai2.py
    FakeSceneUnderStanding/real_ai.py
    FakeSceneUnderStanding/situational_awareness.py
    FakeSceneUnderStanding/Sensor_fusion.py
    FakeSceneUnderStanding/mock_ai.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
