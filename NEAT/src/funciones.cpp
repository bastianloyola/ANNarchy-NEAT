#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN
#include "../headers/funciones.h"
#include <python3.10/numpy/arrayobject.h>
#include <iostream>
#include <vector>

using namespace std;

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

    // Antes de llamar a PyObject_CallObject
    cout << "Load: " << load_module << endl;
    cout << "Función Python: " << func << endl;
    cout << "Argumentos: " << args << endl;

    PyObject* callfunc = PyObject_CallObject(func, args);

    //Set de fit
    double value = PyFloat_AsDouble(callfunc);
    genome.set_fitness(value);

    //Decref de variables necesarias
    Py_DECREF(numpy_array);
    Py_DECREF(args);

}

void evaluate(Population &population){

    // Importar modulo
    PyObject* name = PyUnicode_FromString("annarchy");
    PyObject* load_module = PyImport_Import(name);

    // Llamada paralela a single_evaluation

    vector<Genome> genomes = population.get_genomes();

    
    for (int i = 0; i < static_cast<int>(genomes.size()); i++){
        cout << i << " - " << genomes[i].get_id() << endl;
        single_evaluation( genomes[i], load_module, population.get_n_inputs(), population.get_n_outputs());
    }

    // Decref
    Py_DECREF(name);
}