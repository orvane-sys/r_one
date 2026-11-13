cmake_minimum_required(VERSION 3.5)
project(r_one)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# Install Python module package
ament_python_install_package(${PROJECT_NAME})

if(BUILD_TESTING)
  # Keep tests hermetic: ignore packages in ~/.local (a stray pytest plugin there can break
  # pytest itself, which makes CMake skip the pytest tests below).
  set(ENV{PYTHONNOUSERSITE} "1")

  # Required, not QUIET: a missing test dependency must fail the build instead of
  # silently dropping tests (rosdep installs these from the test_depend entries).
  find_package(ament_cmake_pytest REQUIRED)
  find_package(ament_cmake_ros REQUIRED)
  find_package(ament_lint_auto REQUIRED)

  # This package has no copyright headers and no C++ sources
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  set(ament_cmake_uncrustify_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  # Unit tests that import the real r_one modules
  ament_add_pytest_test(unit_tests test/test_package_smoke.py
    ENV PYTHONNOUSERSITE=1
  )

  # Regression tests for the crash bugs
  ament_add_pytest_test(guard_tests test/test_spatial_vision_guards.py
    ENV PYTHONNOUSERSITE=1
  )

  # Tests for the run log and the scoring tools in the repository's tools/ folder
  ament_add_pytest_test(scoring_tests test/test_scoring.py
    ENV PYTHONNOUSERSITE=1
  )

  # Launch-based integration test, run on an isolated ROS_DOMAIN_ID
  ament_add_ros_isolated_pytest_test(launch_integration_tests test/test_launch_integration.py
    ENV PYTHONNOUSERSITE=1
    TIMEOUT 120
  )
endif()

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

ament_package()
