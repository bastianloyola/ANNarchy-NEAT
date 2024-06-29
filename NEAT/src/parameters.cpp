#include "../headers/parameters.h"
#include <fstream> // Para leer archivos
#include <sstream> // Para dividir líneas
#include <string> // Para manejar cadenas de caracteres

// Función para cargar los parámetros desde el archivo cfg
void Parameters::loadFromCfg(const std::string& filename) {
    std::ifstream file(filename); // Abrir el archivo
    std::string line;

    // Leer línea por línea del archivo
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string key;
        if (std::getline(iss, key, '=')) {
            std::string value;
            if (std::getline(iss, value)) {
                // Asignar valor al parámetro correspondiente
                if (key == "keep") keep = std::stof(value);
                else if (key == "threshold") threshold = std::stof(value);
                else if (key == "probabilityInterespecies") probabilityInterespecies = std::stof(value);
                else if (key == "probabilityNoCrossoverOff") percentageNoCrossoverOff = std::stof(value);
                else if (key == "probabilityWeightMutated") probabilityWeightMutated = std::stof(value);
                else if (key == "probabilityAddNodeSmall") probabilityAddNodeSmall = std::stof(value);
                else if (key == "probabilityAddLink_small") probabilityAddLinkSmall = std::stof(value);
                else if (key == "probabilityAddNodeLarge") probabilityAddNodeLarge = std::stof(value);
                else if (key == "probabilityAddLink_Large") probabilityAddLinkLarge = std::stof(value);
                else if (key == "largeSize") largeSize = std::stoi(value);
                else if (key == "c1") c1 = std::stof(value);
                else if (key == "c2") c2 = std::stof(value);
                else if (key == "c3") c3 = std::stof(value);
                else if (key == "initial_weight") initial_weight = std::stof(value);
                else if (key == "numberGenomes") numberGenomes = std::stoi(value);
                else if (key == "numberInputs") numberInputs = std::stoi(value);
                else if (key == "numberOutputs") numberOutputs = std::stoi(value);
                else if (key == "evolutions") evolutions = std::stoi(value);
                else if (key == "process_max") process_max = std::stoi(value);
                else if (key == "n_max") n_max = std::stoi(value);
                else if (key == "learningRate") learningRate = std::stof(value);
                else if (key == "vectorWeights") { 
                    std::istringstream weightsStream(value);
                    std::string weight;
                    while (std::getline(weightsStream, weight, ';')) {
                        inputWeights.push_back(std::stof(weight));
                    }
                }
                else if (key == "weightsRange") {
                    std::istringstream rangeStream(value);
                    std::string rangePart;
                    float minWeight, maxWeight;
                    std::getline(rangeStream, rangePart, ';');
                    minWeight = std::stof(rangePart);
                    std::getline(rangeStream, rangePart, ';');
                    maxWeight = std::stof(rangePart);
                    weightsRange[0] = minWeight;
                    weightsRange[1] = maxWeight;
                }
            }
        }
    }
}

// Constructor que carga los parámetros desde el archivo cfg
Parameters::Parameters(const std::string& cfgFilename) {
    // Cargar parámetros desde el archivo cfg
    loadFromCfg(cfgFilename);
}

// Constructor por defecto
Parameters::Parameters() {}

// Constructor con parámetros
Parameters::Parameters(int numberGenomes, int numberInputs, int numberOutputs, float keep, float threshold,
            float probabilityInterespecies, float percentageNoCrossoverOff, float probabilityWeightMutated, 
            float probabilityAddNodeSmall, float probabilityAddLinkSmall, float probabilityAddNodeLarge, float probabilityAddLinkLarge,
            int largeSize, float c1, float c2, float c3, float initial_weight)
    :numberGenomes(numberGenomes),numberInputs(numberInputs),numberOutputs(numberOutputs),keep(keep),threshold(threshold),
    probabilityInterespecies(probabilityInterespecies),percentageNoCrossoverOff(percentageNoCrossoverOff),
    probabilityWeightMutated(probabilityWeightMutated),probabilityAddNodeSmall(probabilityAddNodeSmall),
    probabilityAddLinkSmall(probabilityAddLinkSmall),probabilityAddNodeLarge(probabilityAddNodeLarge),
    probabilityAddLinkLarge(probabilityAddLinkLarge),largeSize(largeSize),c1(c1),c2(c2),c3(c3),initial_weight(initial_weight){}
