CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17
CPPFLAGS ?= -Iinclude -I/usr/include/eigen3
LDFLAGS ?=
LDLIBS ?= -lfftw3 -lm

CORE_SRC = $(wildcard src/*.cpp)

fft: $(CORE_SRC) app/main.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(LDLIBS)

test_forces: $(CORE_SRC) tests/test_forces.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(LDLIBS)

test_scf_force_fd: $(CORE_SRC) tests/test_scf_force_fd.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@ $(LDFLAGS) $(LDLIBS)

test_radial_transform: src/radial_transform.cpp tests/test_radial_transform.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $^ -o $@

test: test_forces test_scf_force_fd test_radial_transform
	./test_forces
	./test_scf_force_fd
	./test_radial_transform

clean:
	rm -f fft test_forces test_scf_force_fd test_radial_transform
