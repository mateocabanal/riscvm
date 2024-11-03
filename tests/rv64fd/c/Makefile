CC = riscv64-linux-gnu-gcc
CFLAGS = -march=rv64imaf -mabi=lp64f -nostdlib
SRCS = $(wildcard *.c)
PROGS = $(patsubst %.c,%,$(SRCS))

all: $(PROGS)

%: %.c
	$(CC) $(CFLAGS)  -o $@ $<
