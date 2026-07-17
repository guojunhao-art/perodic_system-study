cmake_minimum_required(VERSION 3.16)
project(pw_dft_split LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 QUIET)

add_library(pw_dft_core
    src/hamiltonian.cpp
    src/scf_modules.cpp
    src/eigensolver.cpp
    src/mixing.cpp
    src/potentials.cpp
    src/forces.cpp
)

target_include_directories(pw_dft_core PUBLIC include)

if (Eigen3_FOUND)
    target_link_libraries(pw_dft_core PUBLIC Eigen3::Eigen)
else()
    target_include_directories(pw_dft_core PUBLIC /usr/include/eigen3)
endif()

target_link_libraries(pw_dft_core PUBLIC fftw3 m)

add_executable(fft app/main.cpp)
target_link_libraries(fft PRIVATE pw_dft_core)

enable_testing()
add_executable(test_forces tests/test_forces.cpp)
target_link_libraries(test_forces PRIVATE pw_dft_core)
add_test(NAME analytic_forces COMMAND test_forces)

add_executable(test_scf_force_fd tests/test_scf_force_fd.cpp)
target_link_libraries(test_scf_force_fd PRIVATE pw_dft_core)
add_test(NAME scf_force_finite_difference COMMAND test_scf_force_fd)
