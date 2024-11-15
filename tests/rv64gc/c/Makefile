CC = riscv64-linux-gnu-gcc
CFLAGS = -march=rv64gc -static -static-libgcc -g
SRCS = $(wildcard *.c)
PROGS = $(patsubst %.c,%,$(SRCS))

all: $(PROGS)

clean:
	rm -r bin

%: %.c
	mkdir -p bin
	$(CC) $(CFLAGS) -o bin/$@ $<
