#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "../headers/run.h"
#include <python3.10/numpy/arrayobject.h>
#include <iostream>
#include <vector>
#include <dirent.h>
#include <fstream>
#include <iomanip>
#include <algorithm>

using namespace std;


std::vector<std::string> configNames(std::string directory) {
    // Config names
    vector<string> configNames;
    std::string path = "config";
    DIR* dir = opendir(path.c_str());
    if (dir == nullptr) {
        std::cerr << "configNames: No se pudo abrir el directorio." << std::endl;
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
void saveConfig(std::string filename, std::string configName) {
    std::ifstream file(configName); // Abrir el archivo
    std::string line;

    ofstream outfile(filename, ios::app);
    // Leer línea por línea del archivo
    while (std::getline(file, line)) {
        outfile << "  " << line << "\n";
    }
    outfile << "\n";
    outfile.close();
}

void saveRun(Population* population, int n, string filename) {
    Genome* best = population->getBest();
    ofstream outfile(filename, ios::app);

    if(!outfile) {
        cerr << "saveRun: No se pudo abrir el archivo." << filename <<endl;
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
    outfile << "\n";
    outfile.close();


    ofstream outfile2("results/best" + to_string(n) + ".txt", ios::app);
    if(!outfile2) {
        cerr << "saveRun: No se pudo abrir el archivo outfile2." <<endl;
    }

    for (int i = 0; i < nConnections; i++) {
        if (connections[i].getEnabled()) {
            outfile2 << connections[i].getInNode() << ";"
                     << connections[i].getOutNode() << ";"
                     << connections[i].getWeight() << "\n";
        }
    }
    outfile2.close();
}

void saveResults(vector<int> bestFitness, int n, string filename) {
    ofstream outfile(filename, ios::app);

    if(!outfile) {
        cerr << "saveResults: No se pudo abrir el archivo." << filename <<endl;
    }

    outfile << "Summerized results: \n";
    for (int i = 0; i < n; i++){
        outfile << "run: " << i << " bestFitness: " << bestFitness[i] << "\n";
    }
    //Percentage of max fitness
    int max = *max_element(bestFitness.begin(), bestFitness.end());
    int nMax = count(bestFitness.begin(), bestFitness.end(), max);
    float percentage = (nMax * 100.0) / n;
    outfile << "Max fitness: " << max << "\nPercentage: " << percentage << "% (" << nMax << " of " << n << ")\n";
    outfile << "--------------------------------------------------------------\n";
    outfile.close();
}

int run(int timesPerConfig) {

    string filename = "results/results.txt";
    string folder_path_1= "annarchy"; // Ruta de la carpeta que deseas borrar
    string folder_path_2 = "__pycache__"; // Ruta de la carpeta que deseas borrar
    printf("---- Running ----\n");
    vector <string> names = configNames("config");
    int nConfig = static_cast<int>(names.size());
    int evolutions;

    vector <int> bestFitnes;
    // Run Cofigs
    for (int j = 0; j < nConfig; j++){
        printf("---- Config: %s ----\n", names[j].c_str());
        ofstream outfile(filename, ios::app);
        if(!outfile) {
            cerr << "run: No se pudo abrir el archivo." << filename <<endl;
        }
        outfile << "\n---- Results of cofig: " << j << " ----\n";
        outfile.close();
        saveConfig(filename, "config/" + names[j]);
        
        printf("---- Loading Config: %s ----\n", names[j].c_str());
        Parameters parameters("config/" + names[j]);
        printf("---- Loaded Config: %s ----\n", names[j].c_str());
        for (int i = 0; i < timesPerConfig; i++){
            printf("---- Run: %d ----\n", i);
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