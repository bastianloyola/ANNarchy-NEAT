#include <iostream>
#include <vector>
#include <string>
#include <fstream>  // Asegúrate de incluir este encabezado 
#include <cstdlib>  // para std::atof
#include <filesystem>
#include <sys/stat.h>  // Para mkdir
#include <sys/types.h> // Para mkdir


#define PY_SSIZE_T_CLEAN
#include <python3.10/Python.h>
#include "../headers/menu.h"

using namespace std;

int main(int argc, char *argv[]) {
  if (argc != 14)
  {
    string folder = "results/trial-0";
    std::string parentFolder = "results";
    std::string subFolder = parentFolder + "/trial-0";

    // Crear la carpeta padre si no existe
    if (mkdir(parentFolder.c_str(), 0777) == 0 || errno == EEXIST) {
        std::cout << "Carpeta padre creada o ya existe: " << parentFolder << std::endl;
        // Crear la subcarpeta
        if (mkdir(subFolder.c_str(), 0777) == 0) {
            std::cout << "Subcarpeta creada exitosamente: " << subFolder << std::endl;
        } else {
            std::cout << "Error al crear la subcarpeta o ya existe." << std::endl;
        }
    } else {
        std::cout << "Error al crear la carpeta padre." << std::endl;
    }

    setenv("PYTHONPATH", ".", 1);
    Py_Initialize();
    //menu();
    std::cout << "starting" << endl;
    //float fitness = run(1);
    float fitness = run3();
    cout << "finalized" << endl;
    Py_Finalize();
    return fitness;
  }
  
//int main() {
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
  /*
  float keep = 0.5;
  float threshold = 3.0;
  float probabilityInterespecies = 0.001;
  float probabilityNoCrossoverOff = 0.75;
  float probabilityWeightMutated = 0.8;
  float probabilityAddNodeSmall = 0.03;
  float probabilityAddLink_small = 0.05;
  float probabilityAddNodeLarge = 0.03;
  float probabilityAddLink_Large = 0.3;
  int largeSize = 20;
  float c1 = 1.0;
  float c2 = 1.0;
  float c3 = 0.4;
  int trialNumber = 0;
  */  

  // Parametros constantes
  int numberGenomes=1;
  keep=1.0;
  int numberInputs=80;
  int numberOutputs=40;
  int evolutions=1;
  float learningRate=10.0;
  float inputWeights_min=0.0;
  float inputWeights_max=150.0;
  float weightsRange_min=-20.0;
  float weightsRange_max=80.0;
  int n_max=200;
  int process_max=1;
  string function="cartpole2";

  // Parametros constantes
  /*
  int numberGenomes=4;
  int numberInputs=2;
  int numberOutputs=1;
  int evolutions=20;
  float learningRate=10.0;
  float inputWeights_min=1.0;
  float inputWeights_max=1.0;
  float weightsRange_min=110.0;
  float weightsRange_max=110.0;
  int n_max=100;
  int process_max=6;
  string function="xor";
  */

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