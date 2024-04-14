CXX = g++
CXXFLAGS = -Wall -std=c++11 -fopenmp -g
LIBS = -lpython3.10

SRCDIR = src
BINDIR = bin
HEADERDIR = headers

SOURCES := $(wildcard $(SRCDIR)/*.cpp)
OBJECTS := $(SOURCES:$(SRCDIR)/%.cpp=$(BINDIR)/%.o)

EXECUTABLE = NEAT

all: $(EXECUTABLE)

$(BINDIR)/%.o: $(SRCDIR)/%.cpp | $(BINDIR)
	$(CXX) $(CXXFLAGS) -I$(PYTHON_INCLUDE) -I$(NUMPY_INCLUDE) -c -o $@ $<

$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(EXECUTABLE) $(OBJECTS) $(LIBS)

$(BINDIR):
	mkdir -p $(BINDIR)
.PHONY: clean
clean:
	rm NEAT
	rm -fr $(BINDIR)
	rm -rf __pycache__
	rm -rf annarchy-*
Aclean:
	rm -rf annarchy-*