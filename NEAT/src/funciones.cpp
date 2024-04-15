#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN
#include "../headers/funciones.h"
#include <python3.10/numpy/arrayobject.h>
#include <iostream>
#include <thread>
#include <vector>
#include <random>
#include <atomic>
#include <mutex>

using namespace std;

std::mutex genome_mutex;

void single_evaluation(Genome &genome, PyObject *load_module, int in, int out){
    //Inicializar varibles necesarias
    int n = static_cast<int>(genome.get_nodes().size());

    //Obtener npArray
    double data[n*n];
    for (int i = 0; i < n * n; ++i) {
        data[i] = NAN;
    }
    cout << 14 << endl;
    for (const auto& connection : genome.get_connections()) {
        int in_node = connection.get_InNode();
        int out_node = connection.get_OutNode();
        double weight = connection.get_weight();
        if (in_node >= 0 && in_node < n && out_node >= 0 && out_node < 3) {
            int index = out_node * n + in_node;
            data[index] = weight;
            cout << index << weight << endl;
        }
    }
    _import_array();
    npy_intp dims[2] = {n, n};
    PyObject* numpy_array = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, data);

    //Llamado a función
    PyObject* func = PyObject_GetAttrString(load_module, "snn");

    PyObject* args = PyTuple_Pack(5, PyFloat_FromDouble(double(in)), PyFloat_FromDouble(double(out)), PyFloat_FromDouble(double(n)), PyFloat_FromDouble(double(genome.get_id())), numpy_array);

    PyObject* callfunc = PyObject_CallObject(func, args);

    //Set de fit
    double value = PyFloat_AsDouble(callfunc);
    cout << "Fitness: " << value << endl;
    genome.set_fitness(value);

    //Decref de variables necesarias
    Py_DECREF(numpy_array);
    Py_DECREF(args);
}

void evaluate(Population &population){

    // Importar modulo
    PyObject* name = PyUnicode_FromString("annarchy");
    PyObject* load_module = PyImport_Import(name);

    std::vector<std::thread> threads;

    vector<Genome> genomes = population.get_genomes();

    // Contador atómico para rastrear el número de hilos completados
    std::atomic<int> threads_completed(0);

    
    // Función para ejecutar en un hilo
    auto evaluate_genome = [&](Genome& genome) {
        std::lock_guard<std::mutex> lock(genome_mutex);
        cout << population.get_n_inputs() << " " << population.get_n_outputs() << endl;
        single_evaluation(genome, load_module, population.get_n_inputs(), population.get_n_outputs());
         // Bloquear el mutex antes de modificar threads_completed
        threads_completed++;
    };
    

    
    // Iniciar un hilo para cada genoma
    for (auto& genome : genomes) {
        threads.emplace_back(evaluate_genome, std::ref(genome));
    }
    
    // Esperar a que todos los hilos hayan terminado
    for (auto& thread : threads) {
        thread.join();
    }

    // Decref
    Py_DECREF(name);
}

// void evolution(Population &population){
//     //probabilidades
//     double weight_mutated = 0.8;
//     double uniform_weight = 0.9; //falta implementar
//     double disable_iherited = 0.75; //falta implementar
//     double offspring_cross = 0.75;
//     double interspecies = 0.001;
//     double add_node_small = 0.03;
//     double add_link_small = 0.05;
//     double add_node_large = 0.03; //no aparece
//     double add_link_large = 0.03;

//     //
//     int size_pop = population.get_genomes().size();
//     int fitter = 0;
//     int fit_fitter = population.get_genomes()[0].get_fitness();
//     for (int i = 1; i < static_cast<int>(population.get_genomes().size()); i++){
//         int mutated_id;
//         // compare fitness
//         if (fit_fitter < population.get_genomes()[i].get_fitness()){
//             mutated_id = fitter;
//             fitter = i;
//             fit_fitter = population.get_genomes()[i].get_fitness();
//         }else mutated_id = i;
        
//         if (getBooleanWithProbability(weight_mutated)){
//             int n = population.get_genomes()[mutated_id].get_connections().size();

//             int innovation_id =  (int)(rand() % n)+1;
//             while (!population.get_genomes()[mutated_id].get_connections()[innovation_id].get_enable())
//             {
//                 int innovation_id =  (int)(rand() % n)+1;
//             }
//             int weight = (rand() %10)/10;
//             population.get_genomes()[mutated_id].change_weight(innovation_id,weight);
//         }
//     }
// }

// bool getBooleanWithProbability(double probability) {
//     // Generador de números aleatorios
//     random_device rd;
//     mt19937 gen(rd());
//     uniform_real_distribution<> dis(0, 1); // Distribución uniforme entre 0 y 1

//     // Generar un número aleatorio entre 0 y 1
//     double randomValue = dis(gen);

//     // Comparar el número aleatorio con la probabilidad dada
//     return randomValue < probability;
// }