CXX = g++
CXXFLAGS = -std=c++11 -stdlib=libc++

output: main.o connection.o node.o genome.o crossover.o funciones.o
	$(CXX) $(CXXFLAGS) main.o connection.o node.o genome.o crossover.o funciones.o -o output

main.o: main.cpp
	$(CXX) $(CXXFLAGS) -c main.cpp

connection.o: connection.cpp
	$(CXX) $(CXXFLAGS) -c connection.cpp

node.o: node.cpp
	$(CXX) $(CXXFLAGS) -c node.cpp

genome.o: genome.cpp
	$(CXX) $(CXXFLAGS) -c genome.cpp

crossover.o: crossover.cpp
	$(CXX) $(CXXFLAGS) -c crossover.cpp

funciones.o: funciones.cpp
	$(CXX) $(CXXFLAGS) -c funciones.cpp

clean:
	rm *.o output