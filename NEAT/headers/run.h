#ifndef RUN_H
#define RUN_H

#include "all_headers.h"
#include "stdlib.h"

#include <vector>
#include <string>


std::vector<std::string> configNames(std::string directory);
void saveRun(Population* population, int n);
void saveResults(std::vector<int> bestFitnes, int n);
int run(int timesPerConfig);

#endif