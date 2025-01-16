CC = gcc
CFLAGS = -O2 -Wall
LDFLAGS = -lopenblas  # This should be used only in the linking step

OBJS = naive_matmul.o cblas_matmul.o main.o

# Default target
all: benchmark


benchmark: $(OBJS)
	$(CC) -o benchmark $(OBJS) $(LDFLAGS)  

naive_matmul.o: naive_matmul.c
	$(CC) $(CFLAGS) -c naive_matmul.c

cblas_matmul.o: cblas_matmul.c
	$(CC) $(CFLAGS) -c cblas_matmul.c  

main.o: main.c
	$(CC) $(CFLAGS) -c main.c

clean:
	rm -f $(OBJS) benchmark
