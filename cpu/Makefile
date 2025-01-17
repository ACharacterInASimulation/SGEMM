CC = gcc
CFLAGS = -O2 -Wall
LDFLAGS = -lopenblas  # This should be used only in the linking step
AVXFLAGS = -march=native -mavx

OBJS = kernels.o matmul.o main.o

# Default target
all: benchmark


benchmark: $(OBJS)
	$(CC) -o benchmark $(OBJS) $(LDFLAGS)  

matmul.o: matmul.c kernels.h matmul.h
	$(CC) $(CFLAGS) -c matmul.c

kernels.o: kernels.c
	$(CC) $(CFLAGS)  $(AVXFLAGS) -c kernels.c  

main.o: main.c matmul.h
	$(CC) $(CFLAGS) -c main.c

clean:
	rm -f $(OBJS) benchmark
