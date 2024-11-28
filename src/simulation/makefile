CPPX = g++
CPP_FLAG = -O2 -std=c++11

PIMD : main.cpp random.cpp simulation.cpp statistics.cpp
	$(CPPX) -o PIMD main.cpp random.cpp simulation.cpp statistics.cpp $(CPP_FLAG)

.PHONY : clean
clean :
	rm PIMD