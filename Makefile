# Linux
CFLAGS=-g -std=c++11 -Wall -O3
LFLAGS=
# Mac
ifeq "$(shell uname)" "Darwin"
CFLAGS=-g -std=c++11 -Wall -O3
LFLAGS=
endif


all: neuralnet.o #mnist


neuralnet.o: neuralnet.cpp neuralnet.h
	g++ $(CFLAGS) -c $< -fopenmp


#mnist: mnist.cpp neuralnet.o connect.o layer.o
#	g++ $(CFLAGS) -o $@ $^ $(LFLAGS)


test: test.o
	./test.o

test.o: test.cpp neuralnet.o
	g++ $(CFLAGS) -o $@ $^ -fopenmp


clean:
	rm -f *.o mnist
