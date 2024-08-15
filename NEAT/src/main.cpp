#include <iostream>
#include <vector>
#include <string>
#include <fstream>  // Asegúrate de incluir este encabezado 
#include <cstdlib>  // para std::atof
#include <filesystem>


#define PY_SSIZE_T_CLEAN
#include <python3.10/Python.h>
#include "../headers/menu.h"

using namespace std;

int main(int argc, char *argv[]) {

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

  // Parametros constantes
  float initial_weights=110.0;
  int numberGenomes=5;
  int numberInputs=2;
  int numberOutputs=1;
  int evolutions=2;
  float learningRate=10.0;
  float inputWeights_min=1.0;
  float inputWeights_max=1.0;
  float weightsRange_min=110.0;
  float weightsRange_max=110.0;
  int n_max=10;
  int process_max=8;
  string function="xor";

  // Escribir en el archivo config.cfg
  string folder = "results/trial-" + std::to_string(trialNumber);
  string filename = folder + "/config.cfg";
  // Crear la carpeta
  std::filesystem::create_directories(folder);
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
  config_file << "initial_weight=" << initial_weights << "\n";
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
