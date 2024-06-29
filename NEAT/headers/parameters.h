#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <vector>
#include <string> // Para manejar cadenas de caracteres

class Parameters{
public:
    Parameters();
    Parameters(const std::string& cfgFilename); // Constructor que carga los parámetros desde el archivo cfg
    Parameters(int numberGenomes, int numberInputs, int numberOutputs, 
        float keep=0.9, float threshold=0.01,
        float probabilityInterespecies=0.001, float probabilityNoCrossoverOff=0.25,
        float probabilityWeightMutated=0.8, 
        float probabilityAddNodeSmall=0.9, float probabilityAddLinkSmall=0.9,
        float probabilityAddNodeLarge=0.9, float probabilityAddLinkLarge=0.9,
        int largeSize=20,
        float c1=1.0, float c2=1.0, float c3=0.4, float initial_weight=110.0);
    int numberGenomes;
    int numberInputs;
    int numberOutputs;
    float keep;
    float threshold;
    float probabilityInterespecies;
    float percentageNoCrossoverOff;
    float probabilityWeightMutated;
    float probabilityAddNodeSmall;
    float probabilityAddLinkSmall;
    float probabilityAddNodeLarge;
    float probabilityAddLinkLarge;
    int largeSize;
    float c1;
    float c2;
    float c3;
    float initial_weight;
    int evolutions;
    int process_max;
    int n_max;
    float learningRate;
    std::vector<float> inputWeights;
    std::vector<float> weightsRange;

private:
    // Función para cargar los parámetros desde el archivo cfg
    void loadFromCfg(const std::string& filename);
};

#endif
