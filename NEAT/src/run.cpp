#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "../headers/run.h"
#include <python3.10/numpy/arrayobject.h>
#include <iostream>
#include <vector>
#include <dirent.h>
#include <fstream>
#include <iomanip>

using namespace std;


std::vector<std::string> configNames(std::string directory) {
    // Config names
    vector<string> configNames;
    std::string path = "config";
    DIR* dir = opendir(path.c_str());
    if (dir == nullptr) {
        std::cerr << "No se pudo abrir el directorio." << std::endl;
    }

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        std::string name = entry->d_name;
        if (name != "." && name != "..") {
            configNames.push_back(name);
        }
    }

    closedir(dir);
    return configNames;
}

void saveRun(Population* population, int n, string filename) {
    Genome* best = population->getBest();
    ofstream outfile(filename, ios::app);

    if(!outfile) {
        cerr << "No se pudo abrir el archivo." << filename <<endl;
    }
    outfile << "--Results of run:" << n << " --\n";
    outfile << "Best Genome \n";
    outfile << "Genome id: " << best->getId() << "\n";
    outfile << "Genome fitness: " << best->getFitness() << "\n";

    best->sort_connections();
    vector<Connection> connections = best->getConnections();
    int nConnections = static_cast<int>(connections.size());

    outfile << std::setw(5) << "IN"
            << std::setw(5) << "OUT"
            << std::setw(10) << "Weight"
            << std::setw(7) << "Innov" << "\n";

    for (int i = 0; i < nConnections; i++) {
        if (connections[i].getEnabled()) {
            outfile << std::setw(5) << connections[i].getInNode()
                    << std::setw(5) << connections[i].getOutNode()
                    << std::setw(10) << connections[i].getWeight()
                    << std::setw(7) << connections[i].getInnovation() << "\n";
        }
    }
    outfile.close();
}

void saveResults(vector<int> bestFitness, int n, string filename) {
    ofstream outfile(filename, ios::app);

    if(!outfile) {
        cerr << "No se pudo abrir el archivo." << filename <<endl;
    }

    outfile << "Summerized results: \n";
    for (int i = 0; i < n; i++){
        outfile << "run: " << i << " bestFitness: " << bestFitness[i] << "\n";
    }
    outfile.close();
}

int getResultName(){
    std::string path = "results";
    DIR* dir = opendir(path.c_str());
    if (dir == nullptr) {
        std::cerr << "No se pudo abrir el directorio." << std::endl;
    }

    struct dirent* entry;
    int n = 0;
    while ((entry = readdir(dir)) != nullptr) {
        std::string name = entry->d_name;
        if (name != "." && name != "..") {
            n++;
        }
    }

    closedir(dir);
    return n;
}

int run(int timesPerConfig) {

    int n = getResultName();
    string filename = "results-"+to_string(n)+".txt";
    vector <string> names = configNames("config");
    int nConfig = static_cast<int>(names.size());
    int evolutions;

    vector <int> bestFitnes;
    // Run Cofigs
    for (int j = 0; j < nConfig; j++){
        ofstream outfile(filename, ios::app);
        if(!outfile) {
            cerr << "No se pudo abrir el archivo." << filename <<endl;
        }
        outfile << "---- Results of cofig: " << j << " ----\n";
        outfile.close();

        Parameters parameters("config/" + names[j]);
        for (int i = 0; i < timesPerConfig; i++){
            Population population(&parameters);
            evolutions = parameters.evolutions;

            population.evolution(evolutions);
            saveRun(&population, i, filename);
            bestFitnes.push_back(population.getBest()->getFitness());
        }
        saveResults(bestFitnes, timesPerConfig, filename);
        bestFitnes.clear();
    }
    return 0;
}