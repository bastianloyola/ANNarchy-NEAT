#include <iostream>
#include <vector>
#include <string>
#include <fstream>  // Asegúrate de incluir este encabezado 
#include <cstdlib>  // para std::atof
#include <filesystem>
#include <iostream>
#include <sstream>


#define PY_SSIZE_T_CLEAN
#include <python3.10/Python.h>
#include "../headers/menu.h"

using namespace std;

int main(int argc, char *argv[]) {

  //verificar el número de argumentos
  if (argc == 0) {
    // Crear el directorio "results/trial-0"
    std::string path = "results/trial-0";
    if (mkdir(path.c_str(), 0777) == 0) {
        std::cout << "Directory created: " << path << std::endl;
    } else {
        std::cerr << "Error creating directory: " << path << std::endl;
    }

    // Copiar el archivo "config/config.cfg" a "results/trial-0/config.cfg"
    std::string src = "config/config.cfg";
    std::string dest = path + "/config.cfg";

    std::ifstream srcFile(src, std::ios::binary);
    std::ofstream destFile(dest, std::ios::binary);

    if (srcFile && destFile) {
        destFile << srcFile.rdbuf();
        std::cout << "File copied from " << src << " to " << dest << std::endl;
    } else {
        std::cerr << "Error copying file from " << src << " to " << dest << std::endl;
    }

    setenv("PYTHONPATH", ".", 1);
    Py_Initialize();
  
    //menu();
    std::cout << "starting" << endl;
    //float fitness = run(1);
    float fitness = run(1);
    cout << "finalized" << endl;
    Py_Finalize();
    return fitness;
  }else{
    // Recibir parametros de la interfaz de usuario
    float keep=std::atof(argv[1]);
    float threshold=std::atof(argv[2]);
    float probabilityInterespecies=std::atof(argv[3]);
    float probabilityNoCrossoverOff=std::atof(argv[4]);
    float probabilityWeightMutated=std::atof(argv[5]);
    float probabilityAddNodeSmall=std::atof(argv[6]);
    float probabilityAddLink_small=std::atof(argv[7]);
    float probabilityAddNodeLarge=std::atof(argv[8]);
    float probabilityAddLink_Large=std::atof(argv[9]);
    int largeSize=20;
    float c1=std::atof(argv[10]);
    float c2=std::atof(argv[11]);
    float c3=std::atof(argv[12]);
    int trialNumber=std::atoi(argv[13]);

    // Parametros constantes leidos desde el archivo config/config.cfg
    std::ifstream file("config/config.cfg"); // Abrir el archivo
    std::string line;
    int numberGenomes;
    int numberInputs;
    int numberOutputs;
    int evolutions;
    float learningRate;
    std::vector<float> inputWeights;
    float weightsRange[2] = {0.0f, 0.0f};
    float inputWeights_min;
    float inputWeights_max;
    float weightsRange_min;
    float weightsRange_max;
    int n_max;
    int process_max;
    string function;

    // Leer línea por línea del archivo
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string key;
        if (std::getline(iss, key, '=')) {
            std::string value;
            if (std::getline(iss, value)) {
                try {
                    if (key == "numberGenomes") numberGenomes = std::stoi(value);
                    else if (key == "numberInputs") numberInputs = std::stoi(value);
                    else if (key == "numberOutputs") numberOutputs = std::stoi(value);
                    else if (key == "evolutions") evolutions = std::stoi(value);
                    else if (key == "process_max") process_max = std::stoi(value);
                    else if (key == "n_max") n_max = std::stoi(value);
                    else if (key == "learningRate") learningRate = std::stof(value);
                    else if (key == "inputWeights") { 
                        std::istringstream weightsStream(value);
                        std::string weight;
                        std::getline(weightsStream, weight, ',');
                        inputWeights_min = std::stof(weight);
                        std::getline(weightsStream, weight, ',');
                        inputWeights_max = std::stof(weight);

                    }
                    else if (key == "weightsRange") {
                        std::istringstream rangeStream(value);
                        std::string rangePart;
                        std::getline(rangeStream, rangePart, ',');
                        float minWeight = std::stof(rangePart);
                        std::getline(rangeStream, rangePart, ',');
                        float maxWeight = std::stof(rangePart);
                        weightsRange_min = minWeight;
                        weightsRange_max = maxWeight;
                    }
                    else if (key == "function") {
                        function = value;
                    }

                }catch (const std::exception& e) {
                    std::cerr << "Error parsing key: " << key << ", value: " << value << ". Exception: " << e.what() << std::endl;
                }
            }
        }
    }
    // Escribir en el archivo config.cfg
    string folder = "results/trial-" + std::to_string(trialNumber);
    string filename = folder + "/config.cfg";
    // Crear la carpeta
    //std::filesystem::create_directories(folder);
    // Crear y abrir el archivo en modo truncado
    ofstream config_file(filename, ofstream::trunc);
    if (!config_file.is_open()) {
      cerr << "No se pudo abrir el archivo config.cfg para escribir." << endl;
      return 1;
    }
    config_file << "keep=" << keep << "\n";
    config_file << "threshold=" << threshold << "\n";
    config_file << "interespeciesRate=" << probabilityInterespecies << "\n";
    config_file << "noCrossoverOff=" << probabilityNoCrossoverOff << "\n";
    config_file << "probabilityWeightMutated=" << probabilityWeightMutated << "\n";
    config_file << "probabilityAddNodeSmall=" << probabilityAddNodeSmall << "\n";
    config_file << "probabilityAddLink_small=" << probabilityAddLink_small << "\n";
    config_file << "probabilityAddNodeLarge=" << probabilityAddNodeLarge << "\n";
    config_file << "probabilityAddLink_Large=" << probabilityAddLink_Large << "\n";
    config_file << "largeSize=" << largeSize << "\n";
    config_file << "c1=" << c1 << "\n";
    config_file << "c2=" << c2 << "\n";
    config_file << "c3=" << c3 << "\n";
    config_file << "numberGenomes=" << numberGenomes << "\n";
    config_file << "numberInputs=" << numberInputs << "\n";
    config_file << "numberOutputs=" << numberOutputs << "\n";
    config_file << "evolutions=" << evolutions << "\n";
    config_file << "n_max=" << n_max << "\n";
    config_file << "learningRate=" << learningRate << "\n";
    config_file << "inputWeights=" << inputWeights_min << "," << inputWeights_max << "\n";
    config_file << "weightsRange=" << weightsRange_min << "," << weightsRange_max << "\n";
    config_file << "process_max=" << process_max << "\n";
    config_file << "function=" << function << "\n";
    config_file << "folder=" << folder << "\n";
    config_file.close();

    setenv("PYTHONPATH", ".", 1);
    Py_Initialize();
  
    //menu();
    std::cout << "starting" << endl;
    //float fitness = run(1);
    float fitness = run2(folder, trialNumber);
    cout << "finalized" << endl;


    Py_Finalize();
    return fitness;
  }
}
