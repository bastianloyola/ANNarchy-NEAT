CXX = g++
CXXFLAGS = -std=c++11 -stdlib=libc++

output: main.o connection.o node.o genome.o
	$(CXX) $(CXXFLAGS) main.o connection.o node.o genome.o -o output

main.o: main.cpp
	$(CXX) $(CXXFLAGS) -c main.cpp

connection.o: connection.cpp
	$(CXX) $(CXXFLAGS) -c connection.cpp

node.o: node.cpp
	$(CXX) $(CXXFLAGS) -c node.cpp

genome.o: genome.cpp
	$(CXX) $(CXXFLAGS) -c genome.cpp

clean:
	rm *.o output