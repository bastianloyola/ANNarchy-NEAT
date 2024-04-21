#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "../headers/menu.h"
#include <python3.10/numpy/arrayobject.h>
#include <iostream>
#include <vector>

using namespace std;

void menu() {
     // Menu to test mutators
  int in_node, out_node, new_weight, innovation, n_genomes, innov_c;
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

  do {
    cout << "Choose an option:  a. create_connection  b. create_node "
        << "c. change weight d. print genome e. exit f. print population"
        <<"g. create snn h. evolucionar i. mutar z. crossover" << endl;
    cin >> option;
    switch (option) {
    case 'a':
      int genome_id;
      //Select genome
      cout << "Enter genome id: ";
      cin >> genome_id;
      cout << "Enter in_node: ";
      cin >> in_node;
      cout << "Enter out_node: ";
      cin >> new_weight;
      cout << "Enter new_weight: ";
      p.genomes[genome_id].createConnection(in_node, out_node, new_weight,p.innov);
      break;
    case 'b':
      //Select genome
      cout << "Enter genome id: ";
      cin >> genome_id;
      //select connection
      cout << "Enter innovation: ";
      cin >> innov_c;
      p.genomes[genome_id].createNode(innov_c,p.innov);
      break;
    case 'c':
      //Select genome
      cout << "Enter genome id: ";
      cin >> genome_id;
      cout << "Enter innovation: ";
      cin >> innovation;
      cout << "Enter new_weight: ";
      cin >> new_weight;
      p.genomes[genome_id].changeWeight(innovation, new_weight);
      break;
    case 'd':
      //Select genome
      cout << "Enter genome id: ";
      cin >> genome_id;
      p.genomes[genome_id].printGenome();
      break;
    case 'f':
      //print all genomes
      p.print();
      break;
    case 'g':
      //create snn 
      cout << "Creando " << endl;
      p.evaluate();
      cout << "---------------------------------------------"<< endl;
      break;
    case 'h':
      //evolutionate
      int n;
      cout << "Enter number of evolutions: ";
      cin >> n;
      cout << "Evolucionando " << endl;
      p.evolution(n);
      cout << "---------------------------------------------"<< endl;
      break;
    case 'i':
      //evolutionate
      cout << "Mutate " << endl;
      p.mutations();
      cout << "---------------------------------------------"<< endl;
      break;
    case 'j':
      //evolutionate
      cout << "Eliminate " << endl;
      p.eliminate();
      cout << "---------------------------------------------"<< endl;
      break;
    case 'z':
      //Select two genomes
      int genome1, genome2;
      cout << "Enter first genome id: ";
      cin >> genome1;
      cout << "Enter second genome id: ";
      cin >> genome2;
      Genome g3 = p.crossover(p.genomes[genome1], p.genomes[genome2]);
      p.maxGenome++;

      g3.printGenome();

      //Add new genome
      p.genomes.push_back(g3);
      break;
    }
  } while (option != 'e');
}