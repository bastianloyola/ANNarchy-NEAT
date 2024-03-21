#include <iostream>
#include <vector>

#include "node.h"
#include "connection.h"
#include "genome.h"

using namespace std;

// Crossover
Genome crossover(Genome a, Genome b, bool equal_fit=false){

    vector<Connection> connections_a = a.get_connections();
    vector<Connection> connections_b = b.get_connections();

    vector<Connection> connections;

    int in_nodes = a.get_in_nodes();
    int out_nodes = a.get_out_nodes();

    int connection_size_a = connections_a.size();
    int connection_size_b = connections_b.size();

    int max;
    if (connections_a[connection_size_a -1].get_Innovation() > connections_b[connection_size_b -1].get_Innovation())
    {
        max = connections_a[connection_size_a -1].get_Innovation();
    }else{
        max = connections_b[connection_size_b -1].get_Innovation();
    }
    int count_a=0, count_b = 0, random;

    Genome offspring(in_nodes, out_nodes);

    // excess and disjoint fron fiiter parent (a)
    offspring.set_nodes(a.get_nodes());
    for (int i = 0; i <= max; i++){
        if (connections_a[count_a].get_Innovation() == i){
            if (connections_b[count_b].get_Innovation() == i){
                random = rand() % 2;
                if (random == 0){
                    connections.push_back(connections_a[count_a]);
                }else{
                    connections.push_back(connections_b[count_b]);
                }
                count_b++;
            }else{
                connections.push_back(connections_a[count_a]);
            }
            count_a++;
        }
    }
    offspring.set_connections(connections);

    return offspring;
}

void testing_all_classes_and_methods() {

    Genome g(3,1);
     // testing all classes and methods
    Connection c(1, 2, 0.5, true, 1);
    cout << c.get_InNode() << endl;
    cout << c.get_OutNode() << endl;
    cout << c.get_weight() << endl;
    cout << c.get_enable() << endl;
    cout << c.get_Innovation() << endl;

    Node n(3, 1);
    Node n2(4, 1);
    cout << n.get_id() << endl;
    cout << n.get_type() << endl;
    g.add_connection(c);
    g.add_node(n);
    g.add_node(n2);
    
    vector<Connection> connections = g.get_connections();
    vector<Node> nodes = g.get_nodes();

    cout << connections.front().get_InNode() << endl;
    cout << connections.front().get_OutNode() << endl;
    cout << connections.front().get_weight() << endl;
    cout << connections.front().get_enable() << endl;
    cout << connections.front().get_Innovation() << endl;

    g.print_genome();
}

void menu() {
     // Menu to test mutators
  int in_node, out_node, new_weight, new_id, new_type, innovation;
  char option;

  //Ingresar cantidad de nodos de entrada y de salida
  int in, out;
  cout << "Enter in_nodes, out_nodes: ";
  cin >> in >> out;


  Genome g1(in, out);
  Genome g2(in, out);
  Genome g3(in, out);
  do {
    cout << "Choose an option:  a. create_connection  b. create_node  c. change_weight  d. print_genome  f. create_connection  g. create_node  h. change_weight  i. print_genome z. cross" << endl;
    cin >> option;
    switch (option) {
    case 'a':
      cout << "Enter in_node, out_node, new_weight: ";
      cin >> in_node >> out_node >> new_weight;
      g1.create_connection(in_node, out_node, new_weight);
      break;
    case 'b':
      //select connection
      cout << "Enter in_node, out_node: ";
      cin >> in_node >> out_node;
      g1.create_node(in_node, out_node);
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
      g2.create_connection(in_node, out_node, new_weight);
      break;
    case 'g':
      //select connection
      cout << "Enter in_node, out_node: ";
      cin >> in_node >> out_node;
      g2.create_node(in_node, out_node);
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
      g3.print_genome();
      break;
  }
  } while (option != 'e');
}

int main() {
    //testing_all_classes_and_methods();
    menu();
  return 0;
}
