CXX = g++
CXXFLAGS = -std=c++11 -stdlib=libc++

PYTHON_INCLUDE = python3.12 # Relative path to the Python include directory

output: main.o connection.o node.o genome.o crossover.o funciones.o population.o
	$(CXX) $(CXXFLAGS) main.o connection.o node.o genome.o crossover.o funciones.o population.o -o output -L/Library/Frameworks/Python.framework/Versions/3.12/lib -lpython3.12

main.o: main.cpp
	$(CXX) $(CXXFLAGS) -c -I./$(PYTHON_INCLUDE) main.cpp

connection.o: connection.cpp
	$(CXX) $(CXXFLAGS) -c -I./$(PYTHON_INCLUDE) connection.cpp

node.o: node.cpp
	$(CXX) $(CXXFLAGS) -c -I./$(PYTHON_INCLUDE) node.cpp

genome.o: genome.cpp
	$(CXX) $(CXXFLAGS) -c -I./$(PYTHON_INCLUDE) genome.cpp

crossover.o: crossover.cpp
	$(CXX) $(CXXFLAGS) -c -I./$(PYTHON_INCLUDE) crossover.cpp

funciones.o: funciones.cpp
	$(CXX) $(CXXFLAGS) -c -I./$(PYTHON_INCLUDE) funciones.cpp

population.o: population.cpp
	$(CXX) $(CXXFLAGS) -c -I./$(PYTHON_INCLUDE) population.cpp

clean:
	rm *.o output
	rm -rf __pycache__
	rm -rf annarchy
