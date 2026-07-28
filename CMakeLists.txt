cmake_minimum_required(VERSION 3.16)
project(pw_dft_split LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(PWDFT_ENABLE_MPI "Enable process-level k-point MPI parallelism" OFF)
option(PWDFT_ENABLE_FFTW_THREADS "Enable threaded FFTW plans" ON)
option(PWDFT_ENABLE_OPENMP "Enable OpenMP real/reciprocal-grid loops" ON)
option(
    PWDFT_ENABLE_BLAS
    "Use an external BLAS backend for Eigen matrix products"
    OFF
)
set(
    PWDFT_BLAS_ROOT ""
    CACHE PATH
    "Optional BLAS/MKL root; searched without changing global CMake paths"
)
set(
    PWDFT_BLAS_LIBRARY ""
    CACHE FILEPATH
    "Optional full path to one BLAS runtime library (for example libmkl_rt.so)"
)
set(
    PWDFT_BLAS_VENDOR ""
    CACHE STRING
    "Optional FindBLAS BLA_VENDOR value when no explicit BLAS path is given"
)

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
    src/bands.cpp
    src/calculation.cpp
    src/fft_workspace.cpp
    src/relaxation.cpp
    src/hamiltonian.cpp
    src/input.cpp
    src/kpoint_scf.cpp
    src/scf_convergence.cpp
    src/scf_modules.cpp
    src/symmetry.cpp
    src/scf.cpp
    src/eigensolver.cpp
    src/mixing.cpp
    src/parallel.cpp
    src/ewald.cpp
    src/potentials.cpp
    src/forces.cpp
    src/radial_transform.cpp
    src/upf_reader.cpp
    src/upf_local_potential.cpp
    src/upf_nonlocal.cpp
    src/xc_functional.cpp
)

target_include_directories(pw_dft_core PUBLIC include)

if (Eigen3_FOUND)
    target_link_libraries(pw_dft_core PUBLIC Eigen3::Eigen)
else()
    target_include_directories(pw_dft_core PUBLIC /usr/include/eigen3)
endif()

target_link_libraries(pw_dft_core PUBLIC m)

if (PWDFT_ENABLE_BLAS)
    set(_pwdft_blas_description "")
    find_package(Threads REQUIRED)
    target_link_libraries(
        pw_dft_core PUBLIC Threads::Threads ${CMAKE_DL_LIBS}
    )

    if (PWDFT_BLAS_LIBRARY)
        if (NOT IS_ABSOLUTE "${PWDFT_BLAS_LIBRARY}")
            message(FATAL_ERROR
                "PWDFT_BLAS_LIBRARY must be an absolute library path."
            )
        endif()
        if (NOT EXISTS "${PWDFT_BLAS_LIBRARY}")
            message(FATAL_ERROR
                "PWDFT_BLAS_LIBRARY does not exist: "
                "${PWDFT_BLAS_LIBRARY}"
            )
        endif()
        add_library(PWDFT::BLAS UNKNOWN IMPORTED)
        set_target_properties(PWDFT::BLAS PROPERTIES
            IMPORTED_LOCATION "${PWDFT_BLAS_LIBRARY}"
        )
        target_link_libraries(pw_dft_core PUBLIC PWDFT::BLAS)
        set(_pwdft_blas_description "${PWDFT_BLAS_LIBRARY}")
    elseif (PWDFT_BLAS_ROOT)
        if (NOT IS_DIRECTORY "${PWDFT_BLAS_ROOT}")
            message(FATAL_ERROR
                "PWDFT_BLAS_ROOT is not a directory: "
                "${PWDFT_BLAS_ROOT}"
            )
        endif()

        # Keep this root out of CMAKE_PREFIX_PATH. In particular, an Intel
        # oneAPI root may also expose Intel MPI, which must not affect the
        # independently selected MPI compiler.
        unset(_PWDFT_BLAS_LIBRARY_FROM_ROOT CACHE)
        find_library(
            _PWDFT_BLAS_LIBRARY_FROM_ROOT
            NAMES mkl_rt openblas blas
            HINTS "${PWDFT_BLAS_ROOT}"
            PATH_SUFFIXES
                lib/intel64
                lib64
                lib
                "lib/${CMAKE_LIBRARY_ARCHITECTURE}"
                "lib/${CMAKE_LIBRARY_ARCHITECTURE}/openblas-pthread"
            NO_DEFAULT_PATH
        )
        if (NOT _PWDFT_BLAS_LIBRARY_FROM_ROOT)
            message(FATAL_ERROR
                "No mkl_rt, openblas, or blas library was found below "
                "PWDFT_BLAS_ROOT=${PWDFT_BLAS_ROOT}. Set "
                "PWDFT_BLAS_LIBRARY to the full library path instead."
            )
        endif()
        add_library(PWDFT::BLAS UNKNOWN IMPORTED)
        set_target_properties(PWDFT::BLAS PROPERTIES
            IMPORTED_LOCATION "${_PWDFT_BLAS_LIBRARY_FROM_ROOT}"
        )
        target_link_libraries(pw_dft_core PUBLIC PWDFT::BLAS)
        set(
            _pwdft_blas_description
            "${_PWDFT_BLAS_LIBRARY_FROM_ROOT}"
        )
    else()
        if (PWDFT_BLAS_VENDOR)
            set(BLA_VENDOR "${PWDFT_BLAS_VENDOR}")
        endif()
        find_package(BLAS REQUIRED)
        target_link_libraries(pw_dft_core PUBLIC ${BLAS_LIBRARIES})
        if (BLAS_LINKER_FLAGS)
            target_link_options(
                pw_dft_core PUBLIC ${BLAS_LINKER_FLAGS}
            )
        endif()
        set(_pwdft_blas_description "${BLAS_LIBRARIES}")
    endif()

    # Eigen retains matrix ownership and high-level expressions while
    # supported dynamic dense products are routed through BLAS (zgemm here).
    target_compile_definitions(pw_dft_core PUBLIC EIGEN_USE_BLAS=1)
    message(STATUS "PWDFT external BLAS backend: ${_pwdft_blas_description}")
endif()

if (PWDFT_ENABLE_FFTW_THREADS)
    find_package(Threads REQUIRED)
    find_library(FFTW3_THREADS_LIBRARY NAMES fftw3_threads)
    if (NOT FFTW3_THREADS_LIBRARY)
        message(FATAL_ERROR
            "Threaded FFTW was requested, but libfftw3_threads was not found."
        )
    endif()
    target_compile_definitions(
        pw_dft_core PUBLIC PWDFT_USE_FFTW_THREADS=1
    )
    target_link_libraries(
        pw_dft_core PUBLIC
        ${FFTW3_THREADS_LIBRARY}
        fftw3
        Threads::Threads
    )
else()
    target_link_libraries(pw_dft_core PUBLIC fftw3)
endif()

if (PWDFT_ENABLE_OPENMP)
    find_package(OpenMP QUIET COMPONENTS CXX)
    if (OpenMP_CXX_FOUND)
        target_link_libraries(pw_dft_core PUBLIC OpenMP::OpenMP_CXX)
    else()
        message(WARNING
            "OpenMP was requested but not found; grid loops remain serial."
        )
    endif()
endif()

if (PWDFT_ENABLE_MPI)
    find_package(MPI REQUIRED COMPONENTS CXX)
    target_compile_definitions(pw_dft_core PUBLIC PWDFT_USE_MPI=1)
    target_link_libraries(pw_dft_core PUBLIC MPI::MPI_CXX)
endif()

if (TARGET PkgConfig::LIBXC)
    target_link_libraries(pw_dft_core PUBLIC PkgConfig::LIBXC)
else()
    target_link_libraries(pw_dft_core PUBLIC LibXC::LibXC)
endif()

add_executable(pwdft app/main.cpp)
target_link_libraries(pwdft PRIVATE pw_dft_core)

add_executable(upf_info app/upf_info.cpp)
target_link_libraries(upf_info PRIVATE pw_dft_core)

add_executable(h2_opt app/h2_opt.cpp)
target_link_libraries(h2_opt PRIVATE pw_dft_core)

add_executable(si2_force_check app/si2_force_check.cpp)
target_link_libraries(si2_force_check PRIVATE pw_dft_core)

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

add_executable(test_input tests/test_input.cpp)
target_link_libraries(test_input PRIVATE pw_dft_core)
target_compile_definitions(
    test_input
    PRIVATE TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data"
)
add_test(NAME poscar_and_calculation_input COMMAND test_input)

add_executable(test_kpoints tests/test_kpoints.cpp)
target_link_libraries(test_kpoints PRIVATE pw_dft_core)
add_test(NAME multi_kpoint_scf COMMAND test_kpoints)
if (PWDFT_ENABLE_MPI)
    add_test(
        NAME multi_kpoint_scf_mpi_2ranks
        COMMAND
            ${MPIEXEC_EXECUTABLE}
            ${MPIEXEC_NUMPROC_FLAG} 2
            ${MPIEXEC_PREFLAGS}
            $<TARGET_FILE:test_kpoints>
            ${MPIEXEC_POSTFLAGS}
    )
    set_tests_properties(multi_kpoint_scf_mpi_2ranks PROPERTIES PROCESSORS 2)
endif()

add_executable(test_symmetry tests/test_symmetry.cpp)
target_link_libraries(test_symmetry PRIVATE pw_dft_core)
target_compile_definitions(
    test_symmetry
    PRIVATE TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data"
)
add_test(NAME kpoint_symmetry_reduction COMMAND test_symmetry)

add_executable(test_xc_functional tests/test_xc_functional.cpp)
target_link_libraries(test_xc_functional PRIVATE pw_dft_core)
add_test(NAME libxc_lda COMMAND test_xc_functional)

add_executable(test_upf_local_potential tests/test_upf_local_potential.cpp)
target_link_libraries(test_upf_local_potential PRIVATE pw_dft_core)
add_test(NAME nc_upf_local_potential COMMAND test_upf_local_potential)

add_executable(test_upf_nonlocal tests/test_upf_nonlocal.cpp)
target_link_libraries(test_upf_nonlocal PRIVATE pw_dft_core)
target_compile_definitions(
    test_upf_nonlocal
    PRIVATE TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/data"
)
add_test(NAME nc_upf_nonlocal COMMAND test_upf_nonlocal)

add_executable(test_ewald tests/test_ewald.cpp)
target_link_libraries(test_ewald PRIVATE pw_dft_core)
add_test(NAME point_ion_ewald COMMAND test_ewald)

add_executable(test_davidson tests/test_davidson.cpp)
target_link_libraries(test_davidson PRIVATE pw_dft_core)
add_test(NAME davidson_incremental_hpsi COMMAND test_davidson)

add_executable(test_batched_hamiltonian tests/test_batched_hamiltonian.cpp)
target_link_libraries(test_batched_hamiltonian PRIVATE pw_dft_core)
add_test(NAME batched_hamiltonian COMMAND test_batched_hamiltonian)

add_executable(test_relaxation tests/test_relaxation.cpp)
target_link_libraries(test_relaxation PRIVATE pw_dft_core)
add_test(NAME fixed_cell_bfgs_relaxation COMMAND test_relaxation)

add_executable(test_bands tests/test_bands.cpp)
target_link_libraries(test_bands PRIVATE pw_dft_core)
add_test(NAME fixed_potential_bands COMMAND test_bands)
