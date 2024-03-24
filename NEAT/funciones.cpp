#include "funciones.h"

#include <iostream>
#include <vector>

using namespace std;



void menu() {
     // Menu to test mutators
  int in_node, out_node, new_weight, new_id, new_type, innovation;
  char option;

  //Ingresar cantidad de nodos de entrada y de salida
  int in, out;
  cout << "Enter in_nodes, out_nodes: ";
  cin >> in >> out;


  Genome g1(in, out, 0, 0);
  Genome g2(in, out, 0, 0);
  Genome g3(in, out, 0, 0);
  int max_innovation = g1.get_local_max();
  

  do {
    cout << "Choose an option:  a. create_connection  b. create_node  c. change_weight  d. print_genome  f. create_connection  g. create_node  h. change_weight  i. print_genome z. cross" << endl;
    cin >> option;
    switch (option) {
    case 'a':
      cout << "Enter in_node, out_node, new_weight: ";
      cin >> in_node >> out_node >> new_weight;
      g1.create_connection(in_node, out_node, new_weight, max_innovation);
      max_innovation = g1.get_local_max();
      g1.set_max(max_innovation);
      g2.set_max(max_innovation);
      g3.set_max(max_innovation);
      break;
    case 'b':
      //select connection
      cout << "Enter in_node, out_node: ";
      cin >> in_node >> out_node;
      g1.create_node(in_node, out_node);
      max_innovation = g1.get_local_max();
      g1.set_max(max_innovation);
      g2.set_max(max_innovation);
      g3.set_max(max_innovation);
      break;
    case 'c':
      cout << "Enter innovation, new_weight: ";
      cin >> innovation >> new_weight;
      g1.change_weight(innovation, new_weight);
      break;
    case 'd':
      g1.print_genome();
      break;
    case 'f':
      cout << "Enter in_node, out_node, new_weight: ";
      cin >> in_node >> out_node >> new_weight;
      g2.create_connection(in_node, out_node, new_weight, max_innovation);
      max_innovation = g2.get_local_max();
      g1.set_max(max_innovation);
      g2.set_max(max_innovation);
      g3.set_max(max_innovation);
      break;
    case 'g':
      //select connection
      cout << "Enter in_node, out_node: ";
      cin >> in_node >> out_node;
      g2.create_node(in_node, out_node);
      max_innovation = g2.get_local_max();
      g1.set_max(max_innovation);
      g2.set_max(max_innovation);
      g3.set_max(max_innovation);
      break;
    case 'h':
      cout << "Enter innovation, new_weight: ";
      cin >> innovation >> new_weight;
      g2.change_weight(innovation, new_weight);
      break;
    case 'i':
      g2.print_genome();
      break;
    case 'z':
      g3 = crossover(g1, g2);
      g1.set_max(max_innovation);
      g2.set_max(max_innovation);
      g3.set_max(max_innovation);
      g3.print_genome();
      break;
  }
  } while (option != 'e');
}