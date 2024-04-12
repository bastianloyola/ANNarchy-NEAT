CC = g++
CFLAGS = -Wall -Wextra -std=c++11
INCLUDES = -I headers # Specify the directory containing header files

SRCS = $(wildcard src/*.cpp) # Specify the directory containing source files
OBJS = $(SRCS:.cpp=.o)

all: NEAT

NEAT: $(OBJS)
	$(CC) $(CFLAGS) $(INCLUDES) -o $@ $^ -lpython3.10
%.o: %.cpp
	$(CC) $(CFLAGS) $(INCLUDES) -c -o $@ $<

clean:
	rm -f $(OBJS) NEAT
	rm -rf __pycache__
	rm -rf annarchy-*