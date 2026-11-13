cmake_minimum_required(VERSION 3.5)
project(r_one_bringup)

find_package(ament_cmake REQUIRED)

if(BUILD_TESTING)
  # Keep tests hermetic: ignore packages in ~/.local (a stray pytest plugin there can break
  # pytest itself, which makes CMake skip the pytest tests below).
  set(ENV{PYTHONNOUSERSITE} "1")

  # Required, not QUIET: a missing test dependency must fail the build instead of
  # silently dropping tests (rosdep installs these from the test_depend entries).
  find_package(ament_cmake_pytest REQUIRED)
  find_package(ament_lint_auto REQUIRED)

  # This package has no copyright headers and no C++ sources
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  set(ament_cmake_uncrustify_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  # The launch files need the installed package
  ament_add_pytest_test(package_file_tests test/test_package_files.py
    ENV PYTHONNOUSERSITE=1
  )
  ament_add_pytest_test(launch_guard_tests test/test_launch_guards.py
    ENV PYTHONNOUSERSITE=1
  )
endif()

# Install non-code asset directories
install(
  DIRECTORY config description launch worlds
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
