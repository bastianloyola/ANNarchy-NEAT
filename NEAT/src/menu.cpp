#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "../headers/menu.h"
#include <python3.10/numpy/arrayobject.h>
#include <iostream>
#include <vector>


using namespace std;

vector<Genome> menu() {
     // Menu to test mutators
  int in_node, out_node, new_weight, innovation, n_genomes;
  char option;

  //Ingresar la cantidad de genomes de la poblacion
  cout << "Enter n_genomes: ";
  cin >> n_genomes;

  //Ingresar cantidad de nodos de entrada y de salida
  int in, out;
  cout << "Enter in_nodes: ";
  cin >> in;
  cout << "Enter out_nodes: ";
  cin >> out;

  // Crear poblacion
  Population p(n_genomes, in, out);
  
  vector<Genome> g = p.get_genomes();
  
  int max_innovation = g.front().get_local_max();
  cout << "max_id" << p.get_max_id() << endl;
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
      for(int i = 0; i < static_cast<int>(g.size()); i++){
        cout << "Genoma " << i << endl;
        g[i].print_genome();
        cout << "---------------------------------------------"<< endl;
      }
      break;
    case 'g':
      //create snn 
      cout << "Creando " << endl;
      evaluate(p);
      cout << "---------------------------------------------"<< endl;
      break;
    case 'z':
      //Select two genomes
      int genome1, genome2;
      cout << "Enter first genome id: ";
      cin >> genome1;
      cout << "Enter second genome id: ";
      cin >> genome2;
      Genome g3 = crossover(g[genome1], g[genome2],p.get_max_id());
      p.increase_max_id();

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