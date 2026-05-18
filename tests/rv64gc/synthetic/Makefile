CC := riscv64-linux-gnu-gcc
ARCH_FLAGS := -march=rv64gc -mabi=lp64d
ASM_CFLAGS := $(ARCH_FLAGS) -static -nostdlib -Wl,--build-id=none
CFLAGS := $(ARCH_FLAGS) -O2 -static -static-libgcc -Wl,--build-id=none
ASM_SRCS := $(wildcard *.S)
C_SRCS := $(wildcard *.c)
PROGS := $(patsubst %.S,bin/%,$(ASM_SRCS)) $(patsubst %.c,bin/%,$(C_SRCS))

all: $(PROGS)

bin/%: %.S
	mkdir -p bin
	$(CC) $(ASM_CFLAGS) -o $@ $<

bin/%: %.c
	mkdir -p bin
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -rf bin
