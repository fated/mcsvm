CXX ?= g++
CFLAGS = -Wall -Wconversion -O3 -fPIC
SHVER = 2
OS = $(shell uname)

all: mcsvm-offline mcsvm-cv

mcsvm-offline: mcsvm-offline.cpp utilities.o mcsvm.o
	$(CXX) $(CFLAGS) mcsvm-offline.cpp utilities.o mcsvm.o -o mcsvm-offline -lm

mcsvm-cv: mcsvm-cv.cpp utilities.o mcsvm.o
	$(CXX) $(CFLAGS) mcsvm-cv.cpp utilities.o mcsvm.o -o mcsvm-cv -lm

utilities.o: utilities.cpp utilities.h
	$(CXX) $(CFLAGS) -c utilities.cpp

mcsvm.o: mcsvm.cpp mcsvm.h
	$(CXX) $(CFLAGS) -c mcsvm.cpp

clean:
	rm -f utilities.o mcsvm.o mcsvm-offline mcsvm-cv
