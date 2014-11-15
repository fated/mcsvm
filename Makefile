CXX ?= g++
CFLAGS = -Wall -Wconversion -O3 -fPIC
SHVER = 2
OS = $(shell uname)

all: vm-offline

vm-offline: vm-offline.cpp utilities.o mcsvm.o
	$(CXX) $(CFLAGS) vm-offline.cpp utilities.o mcsvm.o -o vm-offline -lm

utilities.o: utilities.cpp utilities.h
	$(CXX) $(CFLAGS) -c utilities.cpp

mcsvm.o: mcsvm.cpp mcsvm.h
	$(CXX) $(CFLAGS) -c mcsvm.cpp

clean:
	rm -f utilities.o mcsvm.o vm-offline
