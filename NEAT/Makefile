CXX = g++
CXXFLAGS = -Wall -std=c++11 -g
LIBS = -lpython3.10
PYTHON_INCLUDE=/home/florencia/magister/trabajo_de_titulacion/repos/ANNarchy-NEAT/venv/include/python3.10
NUMPY_INCLUDE=/home/florencia/magister/trabajo_de_titulacion/repos/ANNarchy-NEAT/venv/lib/python3.10/site-packages/numpy/_core/include

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
	rm -f NEAT
	rm -fr $(BINDIR)
	rm -rf __pycache__
	rm -rf annarchy
	rm -rf results
Aclean:
	rm -rf annarchy