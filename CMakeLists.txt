cmake_minimum_required(VERSION 3.16)
project(pw_dft_split LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 QUIET)
find_package(PkgConfig QUIET)

if (PkgConfig_FOUND)
    pkg_check_modules(LIBXC QUIET IMPORTED_TARGET libxc)
endif()

if (NOT TARGET PkgConfig::LIBXC)
    find_path(LIBXC_INCLUDE_DIR NAMES xc.h)
    find_library(LIBXC_LIBRARY NAMES xc)
    if (NOT LIBXC_INCLUDE_DIR OR NOT LIBXC_LIBRARY)
        message(FATAL_ERROR
            "LibXC was not found. Install its development package "
            "(for example libxc-dev) or set LIBXC_INCLUDE_DIR and LIBXC_LIBRARY."
        )
    endif()
    add_library(LibXC::LibXC UNKNOWN IMPORTED)
    set_target_properties(LibXC::LibXC PROPERTIES
        IMPORTED_LOCATION "${LIBXC_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${LIBXC_INCLUDE_DIR}"
    )
endif()

add_library(pw_dft_core
    src/hamiltonian.cpp
    src/scf_modules.cpp
    src/scf.cpp
    src/eigensolver.cpp
    src/mixing.cpp
    src/ewald.cpp
    src/potentials.cpp
    src/forces.cpp
    src/radial_transform.cpp
    src/upf_reader.cpp
    src/upf_local_potential.cpp
    src/xc_functional.cpp
)

target_include_directories(pw_dft_core PUBLIC include)

if (Eigen3_FOUND)
    target_link_libraries(pw_dft_core PUBLIC Eigen3::Eigen)
else()
    target_include_directories(pw_dft_core PUBLIC /usr/include/eigen3)
endif()

target_link_libraries(pw_dft_core PUBLIC fftw3 m)

if (TARGET PkgConfig::LIBXC)
    target_link_libraries(pw_dft_core PUBLIC PkgConfig::LIBXC)
else()
    target_link_libraries(pw_dft_core PUBLIC LibXC::LibXC)
endif()

add_executable(fft app/main.cpp)
target_link_libraries(fft PRIVATE pw_dft_core)

add_executable(upf_info app/upf_info.cpp)
target_link_libraries(upf_info PRIVATE pw_dft_core)

add_executable(h2_opt app/h2_opt.cpp)
target_link_libraries(h2_opt PRIVATE pw_dft_core)

enable_testing()
add_executable(test_forces tests/test_forces.cpp)
target_link_libraries(test_forces PRIVATE pw_dft_core)
add_test(NAME analytic_forces COMMAND test_forces)

add_executable(test_scf_force_fd tests/test_scf_force_fd.cpp)
target_link_libraries(test_scf_force_fd PRIVATE pw_dft_core)
add_test(NAME scf_force_finite_difference COMMAND test_scf_force_fd)

add_executable(test_radial_transform tests/test_radial_transform.cpp)
target_link_libraries(test_radial_transform PRIVATE pw_dft_core)
add_test(NAME radial_fourier_bessel COMMAND test_radial_transform)

add_executable(test_upf_reader tests/test_upf_reader.cpp)
target_link_libraries(test_upf_reader PRIVATE pw_dft_core)
target_compile_definitions(
    test_upf_reader
    PRIVATE TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data"
)
add_test(NAME nc_upf_reader COMMAND test_upf_reader)

add_executable(test_xc_functional tests/test_xc_functional.cpp)
target_link_libraries(test_xc_functional PRIVATE pw_dft_core)
add_test(NAME libxc_lda COMMAND test_xc_functional)

add_executable(test_upf_local_potential tests/test_upf_local_potential.cpp)
target_link_libraries(test_upf_local_potential PRIVATE pw_dft_core)
add_test(NAME nc_upf_local_potential COMMAND test_upf_local_potential)

add_executable(test_ewald tests/test_ewald.cpp)
target_link_libraries(test_ewald PRIVATE pw_dft_core)
add_test(NAME point_ion_ewald COMMAND test_ewald)
