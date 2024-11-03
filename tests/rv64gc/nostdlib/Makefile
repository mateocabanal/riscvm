CC = riscv64-linux-gnu-gcc
CFLAGS = -march=rv64gc -nostdlib
SRCS = $(wildcard *.c)
PROGS = $(patsubst %.c,%,$(SRCS))

all: $(PROGS)

%: %.c
	$(CC) $(CFLAGS)  -o $@ $<
