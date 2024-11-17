CXX = g++
CXXFLAGS = -Wall -std=c++11 -g
LIBS = -lpython3.10

SRCDIR = src
BINDIR = bin
HEADERDIR = headers

SOURCES := $(wildcard $(SRCDIR)/*.cpp)
OBJECTS := $(SOURCES:$(SRCDIR)/%.cpp=$(BINDIR)/%.o)

EXECUTABLE = NEAT

# Compilación normal (dinámica)
all: $(EXECUTABLE)

$(BINDIR)/%.o: $(SRCDIR)/%.cpp | $(BINDIR)
	$(CXX) $(CXXFLAGS) -I$(PYTHON_INCLUDE) -I$(NUMPY_INCLUDE) -c -o $@ $<

$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(EXECUTABLE) $(OBJECTS) $(LIBS)

$(BINDIR):
	mkdir -p $(BINDIR)

# Limpieza
.PHONY: clean
clean:
	rm -f $(EXECUTABLE)
	rm -fr $(BINDIR)
	rm -rf __pycache__
	rm -rf annarchy
	rm -rf results

Aclean:
	rm -rf annarchy

# Compilación estática para ejecutable autónomo
export: $(OBJECTS)
	$(CXX) $(CXXFLAGS) -static -o $(EXECUTABLE) $(OBJECTS) \
		/usr/lib/x86_64-linux-gnu/libpython3.10.a \
		/usr/lib/x86_64-linux-gnu/libz.a \
		/usr/lib/x86_64-linux-gnu/libexpat.a \
		/usr/lib/x86_64-linux-gnu/libm.a \
		/usr/lib/x86_64-linux-gnu/libpthread.a \
		/usr/lib/x86_64-linux-gnu/libc.a \
		/usr/lib/gcc/x86_64-linux-gnu/11/libgcc.a \
		/usr/lib/gcc/x86_64-linux-gnu/11/libstdc++.a
