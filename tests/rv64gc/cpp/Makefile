CC = riscv64-linux-gnu-g++
CFLAGS = -static -g
SRCS = $(wildcard *.cpp)
PROGS = $(patsubst %.cpp,%,$(SRCS))

all: $(PROGS)

%: %.cpp
	$(CC) $(CFLAGS)  -o $@ $<
