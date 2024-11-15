CC = riscv64-linux-gnu-g++
CFLAGS = -static -g -std=c++23
SRCS = $(wildcard *.cpp)
PROGS = $(patsubst %.cpp,%,$(SRCS))

all: $(PROGS)

clean: 
	rm -r bin

%: %.cpp
	mkdir -p bin
	$(CC) $(CFLAGS) -o bin/$@ $<
