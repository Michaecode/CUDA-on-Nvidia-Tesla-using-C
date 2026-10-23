NVCC ?= nvcc
# Override for another GPU, e.g. make CUDA_ARCH=sm_75
CUDA_ARCH ?= sm_35
TARGET ?= nbody
NVCCFLAGS ?= -O3 -lineinfo

.PHONY: all clean run
all: $(TARGET)

$(TARGET): nbody.cu
	$(NVCC) $(NVCCFLAGS) -arch=$(CUDA_ARCH) -o $@ $<

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
