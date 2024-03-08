#include <iostream>

extern "C" void _mlir_my_func() {
  std::cerr << "my function called" << std::endl;
}

struct MemRef {
  void *data_alloc;
  void *data;
  int64_t offset;
  int64_t sizes[];
};

extern "C" void alloc_memref(int64_t rank, MemRef *memref, int64_t *sizes) {
  std::cerr << "calling-alloc_memref" << std::endl;
  std::cerr << "rank: " << rank << std::endl;
  int64_t total_size = 1;
  for (int64_t i = 0; i < rank; ++i) {
    auto size = sizes[i];
    std::cerr << "size " << i << " : " << size << std::endl;
    memref->sizes[i] = size;
    memref->sizes[i + rank] = total_size;
    total_size *= size;
  }
  void *data = malloc(sizeof(int) * total_size);
  memref->data_alloc = data;
  memref->data = data;
  memref->offset = 0;
}

extern "C" void _mlir_alloc_memref(void **) {}
