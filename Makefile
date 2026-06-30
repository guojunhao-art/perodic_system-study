CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Iinclude -I/usr/include/eigen3 -I/home/guojunhao/anaconda3/envs/p4env/include -L/home/guojunhao/anaconda3/envs/p4env/lib
LDFLAGS ?= -lfftw3 -lm

SRC = src/*.cpp app/main.cpp

fft: $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o fft $(LDFLAGS)

clean:
	rm -f fft
