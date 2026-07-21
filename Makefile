CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17
CPPFLAGS ?= -Iinclude -I/usr/include/eigen3
LDFLAGS ?=
LDLIBS ?= -lfftw3 -lm
LIBXC_CFLAGS ?= $(shell pkg-config --cflags libxc 2>/dev/null)
LIBXC_LIBS ?= $(shell pkg-config --libs libxc 2>/dev/null)

ifeq ($(strip $(LIBXC_LIBS)),)
LIBXC_LIBS = -lxc
endif

CORE_CPPFLAGS = $(CPPFLAGS) $(LIBXC_CFLAGS)
CORE_LDLIBS = $(LDLIBS) $(LIBXC_LIBS)

CORE_SRC = $(wildcard src/*.cpp)

fft: $(CORE_SRC) app/main.cpp
	$(CXX) $(CORE_CPPFLAGS) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(CORE_LDLIBS)

test_forces: $(CORE_SRC) tests/test_forces.cpp
	$(CXX) $(CORE_CPPFLAGS) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(CORE_LDLIBS)

test_scf_force_fd: $(CORE_SRC) tests/test_scf_force_fd.cpp
	$(CXX) $(CORE_CPPFLAGS) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(CORE_LDLIBS)

test_xc_functional: $(CORE_SRC) tests/test_xc_functional.cpp
	$(CXX) $(CORE_CPPFLAGS) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(CORE_LDLIBS)

test_upf_local_potential: $(CORE_SRC) tests/test_upf_local_potential.cpp
	$(CXX) $(CORE_CPPFLAGS) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(CORE_LDLIBS)

test_upf_nonlocal: $(CORE_SRC) tests/test_upf_nonlocal.cpp
	$(CXX) $(CORE_CPPFLAGS) $(CXXFLAGS) -DTEST_DATA_DIR=\"tests/data\" $^ -o $@ $(LDFLAGS) $(CORE_LDLIBS)

test_ewald: $(CORE_SRC) tests/test_ewald.cpp
	$(CXX) $(CORE_CPPFLAGS) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(CORE_LDLIBS)

test_davidson: $(CORE_SRC) tests/test_davidson.cpp
	$(CXX) $(CORE_CPPFLAGS) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(CORE_LDLIBS)

test_radial_transform: src/radial_transform.cpp tests/test_radial_transform.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@

test_upf_reader: src/upf_reader.cpp tests/test_upf_reader.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -DTEST_DATA_DIR=\"tests/data\" $^ -o $@

upf_info: src/upf_reader.cpp src/radial_transform.cpp src/upf_local_potential.cpp app/upf_info.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(LDLIBS)

h2_opt: $(CORE_SRC) app/h2_opt.cpp
	$(CXX) $(CORE_CPPFLAGS) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(CORE_LDLIBS)

test: test_forces test_scf_force_fd test_radial_transform test_upf_reader test_xc_functional test_upf_local_potential test_upf_nonlocal test_ewald test_davidson
	./test_forces
	./test_scf_force_fd
	./test_radial_transform
	./test_upf_reader
	./test_xc_functional
	./test_upf_local_potential
	./test_upf_nonlocal
	./test_ewald
	./test_davidson

clean:
	rm -f fft upf_info h2_opt test_forces test_scf_force_fd test_radial_transform test_upf_reader test_xc_functional test_upf_local_potential test_upf_nonlocal test_ewald test_davidson
