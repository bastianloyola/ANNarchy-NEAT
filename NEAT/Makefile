CXX = g++
#CXXFLAGS = -std=c++11 -stdlib=libc++
CXXFLAGS = 
output: main.o connection.o node.o genome.o crossover.o funciones.o population.o
	$(CXX) $(CXXFLAGS) main.o connection.o node.o genome.o crossover.o funciones.o population.o -o output -lpython3.10

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

population.o: population.cpp
	$(CXX) $(CXXFLAGS) -c population.cpp

clean:
	rm *.o output
	rm -rf __pycache__
	rm -rf annarchy
