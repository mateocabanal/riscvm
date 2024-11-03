CC = riscv64-linux-gnu-gcc
CFLAGS = -march=rv64gc -static -static-libgcc -O3
SRCS = $(wildcard *.c)
PROGS = $(patsubst %.c,%,$(SRCS))

all: $(PROGS)

%: %.c
	$(CC) $(CFLAGS)  -o $@ $<
