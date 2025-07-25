
DPCXX = icpx

SOURCES += $(wildcard *.cpp)
DPCXX_TARGETS := $(patsubst %.cpp, %-dpcpp, $(SOURCES))
TARGETS := $(DPCXX_TARGETS)

MKL_FLAGS = -I$(MKLROOT)/include

GOUDA_FLAGS = -I$(GOUDA_DIR)

CXXFLAGS = -O3 -std=c++2b $(GOUDA_FLAGS) $(MKL_FLAGS)

MKL_LIBS = -L$(MKLROOT)/lib/intel64 -lmkl_sycl -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core

LDLIBS = $(MKL_LIBS)

DPCPP_FLAGS = -fsycl

all: $(TARGETS)

run: all
	@for target in $(foreach target,$(TARGETS),./$(target)) ; do echo "Running \"$$target\"" ; $$target ; done


dpcpp: $(DPCXX_TARGETS)

%-dpcpp: %.cpp
	$(DPCXX) $(CXXFLAGS) -o $@ $^ $(LD_FLAGS) $(DPCPP_FLAGS) $(LDLIBS)

clean:
	rm -fv $(TARGETS)

