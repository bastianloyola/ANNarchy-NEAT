#include "funciones.h"
#include "population.h"
#include <iostream>
#include <vector>

using namespace std;

PyObject* vectorGenome_to_TupleList( vector<Genome> genome) {

  PyObject* genome_connections_list = PyList_New( genome.size() );

  for (int i = 0; i < genome.size(); i++){
    vector <Connection> data = genome[i].get_connections();
    PyObject* listObj = PyList_New( data.size() +1);
    
    double num = genome[i].get_nodes().size();
    PyObject *n = PyFloat_FromDouble(num);

    PyList_SET_ITEM(listObj, 0, n);

    int count = 1;
	  for (int j = 0; j < data.size(); j++) {
      if (data[j].get_enable()){
        PyObject* tuple = PyTuple_New(3);

        PyObject *in = PyFloat_FromDouble( (double) data[j].get_InNode() );
        PyObject *out = PyFloat_FromDouble( (double) data[j].get_OutNode() );
        PyObject *weight = PyFloat_FromDouble( (double) data[j].get_weight() );

        PyTuple_SET_ITEM(tuple, 0, in);
        PyTuple_SET_ITEM(tuple, 1, out);
        PyTuple_SET_ITEM(tuple, 2, weight);

        PyList_SET_ITEM(listObj, count, tuple);
        count++;
      }
	  }
    PyList_SET_ITEM(genome_connections_list, i, listObj);
  }
	return genome_connections_list;
}

void createSNN(vector <Genome> g){
  //Py_Initialize();
  int in_nodes = g[0].get_in_nodes();
  int out_nodes = g[0].get_out_nodes();
  PyObject* input_index  = PyList_New( in_nodes );
  PyObject* output_index  = PyList_New( out_nodes );
  for (int i = 0; i < in_nodes; i++){
    PyObject *in = PyFloat_FromDouble( (double) i);
    PyList_SET_ITEM(input_index, i, in);
  }
  int j = 0;
  for (int i = in_nodes; i < in_nodes + out_nodes; i++){
    PyObject *output = PyFloat_FromDouble( (double) i);
    PyList_SET_ITEM(output_index, j, output);
    j++;
  }

  PyObject* list = vectorGenome_to_TupleList(g);

  PyObject *name, *load_module, *func, *callfunc, *args, *n, *obj;
  
  name = PyUnicode_FromString("annarchy");
  printf("llegue\n");
  load_module = PyImport_Import(name);
  printf("llegue\n");

  func = PyObject_GetAttrString(load_module, (char*)"snn");
  printf("llegue\n");

  args = PyTuple_Pack(3, input_index, output_index,list);
  
  callfunc = PyObject_CallObject(func,args);
  
  vector <double> fits;

  // Acceder a los elementos de la lista devuelta

  if (PyList_Check(callfunc)) {
    Py_ssize_t len = PyList_Size(callfunc);
    for (Py_ssize_t i = 0; i < len; ++i) {
      PyObject* item = PyList_GetItem(callfunc, i);
      if (PyLong_Check(item)) {
        double value = PyFloat_AsDouble(item);
        cout << value << endl;
        fits.push_back(value);
        // Usa el valor según sea necesario
      } else {
        // Manejo de error si no es un flotante
      }
    }
  }

  Py_DECREF(name);
  Py_DECREF(load_module);
  Py_DECREF(func);
  Py_DECREF(callfunc);
  Py_DECREF(args);
  Py_DECREF(list);
  
  //Py_Finalize();
}

vector<Genome> menu() {
     // Menu to test mutators
  int in_node, out_node, new_weight, new_id, new_type, innovation, n_genomes;
  char option;

  //Ingresar la cantidad de genomes de la poblacion
  cout << "Enter n_genomes: ";
  cin >> n_genomes;

  //Ingresar cantidad de nodos de entrada y de salida
  int in, out;
  cout << "Enter in_nodes, out_nodes: ";
  cin >> in >> out;

  // Crear poblacion
  Population p(n_genomes, in, out);
  
  vector<Genome> g = p.get_genomes();
  
  int max_innovation = g.front().get_local_max();
  
  do {
    cout << "Choose an option:  a. create_connection  b. create_node  c. change weight d. print genome e. exit f. print population g. create snn z. crossover" << endl;
    cin >> option;
    switch (option) {
    case 'a':
      int genome_id;
      //Select genome
      cout << "Enter genome id: ";
      cin >> genome_id;

      cout << "Enter in_node, out_node, new_weight: ";
      cin >> in_node >> out_node >> new_weight;
      g[genome_id].create_connection(in_node, out_node, new_weight, max_innovation);
      max_innovation = g[genome_id].get_local_max();
      p.increase_max_innovation();
      p.set_genomes(g);
      break;
    case 'b':
      //Select genome
      cout << "Enter genome id: ";
      cin >> genome_id;
      //select connection
      cout << "Enter in_node, out_node: ";
      cin >> in_node >> out_node;
      g[genome_id].create_node(in_node, out_node);
      max_innovation = g[genome_id].get_local_max();
      p.increase_max_innovation();
      p.set_genomes(g);
      break;
    case 'c':
      //Select genome
      cout << "Enter genome id: ";
      cin >> genome_id;
      cout << "Enter innovation, new_weight: ";
      cin >> innovation >> new_weight;
      g[genome_id].change_weight(innovation, new_weight);
      p.set_genomes(g);
      break;
    case 'd':
      //Select genome
      cout << "Enter genome id: ";
      cin >> genome_id;
      g[genome_id].print_genome();
      break;
    case 'f':
      //print all genomes 
      for(int i = 0; i < g.size(); i++){
        cout << "Genoma " << i << endl;
        g[i].print_genome();
        cout << "---------------------------------------------"<< endl;
      }
      break;
    case 'g':
      //create snn 
      cout << "Creando " << endl;
      createSNN(g);
      cout << "---------------------------------------------"<< endl;
      break;
    case 'z':
      //Select two genomes
      int genome1, genome2;
      cout << "Enter first genome id: ";
      cin >> genome1;
      cout << "Enter second genome id: ";
      cin >> genome2;
      Genome g3 = crossover(g[genome1], g[genome2]);

      g3.print_genome();

      //Add new genome
      g.push_back(g3);
      p.set_genomes(g);
      p.set_n_genomes(g.size());

      break;
    }
  } while (option != 'e');
  return g;
}