#include <iostream>
#include <vector>

using namespace std;

// Clase de las conexiones entre nodos
class Connection {
    int in_node;
    int out_node;
    float weight; //no se si es float         
    bool enable;
    int innovation;    

  public:  

    // Constructor de la clase
    Connection(int c_in_node, int c_out_node, float c_weight, bool c_enable, int c_innovation){
        in_node = c_in_node;
        out_node = c_out_node;
        weight = c_weight;
        enable = c_enable;
        innovation = c_innovation; // tenemos que tener una variable global que permita ir poniendo este valor
    }

    // Getters         
    int get_InNode(){
        return in_node;
    }
    int get_OutNode(){
        return out_node;
    }
    float get_weight(){
        return weight;
    }
    bool get_enable(){
        return enable;
    }
    int get_Innovation(){
        return innovation;
    }

    // Setters 
    void set_enable(bool new_enable){
        enable = new_enable;
    }
    void set_weight(int new_weight){
        weight = new_weight;
    }
    void change_enable(){
        enable = !enable;
    }
};

// Clase de nodos
class Node {       
    int id;
    int type; //1, 2 o 3

    public:

        // Constructor de la clase
        Node(int c_id, int c_type){
            id = c_id;
            type = c_type;
        }

        // Getters
        int get_id(){
            return id;
        }
        int get_type(){
            return type;
        }
};

// Clase para Genoma
class Genome {
    int in_nodes;
    int out_nodes; 
    vector<Connection> connections;
    vector<Node> nodes;
    
    public:  

        Genome(int in, int out){
            in_nodes = in;
            out_nodes = out;
            for (int i = 0; i < in_nodes; i++){
                Node n(i, 0);
                nodes.push_back(n);
            }
            for (int i = 0; i < out_nodes; i++){
                Node n(i + in_nodes, 2);
                nodes.push_back(n);
            }
            //Crear conexiones entre todos los nodos de entrada y todos los nodos de salida
            int new_innovation = 0;
            for (int i = 0; i < nodes.size(); i++){
                if (nodes[i].get_type() == 0){
                    for (int j = 0; j < nodes.size(); j++){
                        if (nodes[j].get_type() == 2){
                            Connection c(nodes[i].get_id(), nodes[j].get_id(), 1, true, new_innovation);
                            connections.push_back(c);
                            new_innovation++;
                        }
                    }
                }
            }

        }
        void add_connection(Connection c){
            connections.push_back(c);
        }
        void add_node(Node c){
            nodes.push_back(c);
        }  

        vector<Connection> get_connections(){
            return connections;
        }       

        vector<Node> get_nodes(){
            return nodes;
        }

        int get_in_nodes(){
            return in_nodes;
        }

        int get_out_nodes(){
            return out_nodes;
        }

        Connection get_connection(int in_node, int out_node){
            //Find connection in vector
            for(int i = 0; i < connections.size(); i++){
                if(connections[i].get_InNode() == in_node && connections[i].get_OutNode() == out_node){
                    return connections[i];
                }
            }
            return Connection(0,0,0,false,0);
        }

        Node get_node(int id){
            for(int i = 0; i < nodes.size(); i++){
                if(nodes.front().get_id() == id){
                    return nodes.front();
                }
            }
            return Node(0,0);
        }

        void set_connections(vector<Connection> new_connections){
            connections = new_connections;
        }

        void set_nodes(vector<Node> new_nodes){
            nodes = new_nodes;
        }

        void set_in_nodes(int new_in){
            in_nodes = new_in;
        }

        void set_out_nodes(int new_out){
            out_nodes = new_out;
        }

        // Mutators

        // Change weight, this depends
        void change_weight(int innovation, float new_weight){
                connections[innovation-1].set_weight(new_weight);
        }

        // Create new connection
        void create_connection(int in_node, int out_node, float new_weight){
            int new_innovation = connections.size();
            Connection c(in_node, out_node, new_weight, true, new_innovation);
            connections.push_back(c);
        }

        // Create new node
        void create_node(int in_node, int out_node){
            // Find connection and disable
            float new_weight = 1; // Valor default en caso que no exista una conexion previa
            for(int i = 0; i < connections.size(); i++){
                if(connections[i].get_InNode() == in_node && connections.front().get_OutNode() == out_node){
                    connections[i].set_enable(false);
                    new_weight = connections[i].get_weight();
                }
            }
            // get last id
            int new_id = nodes.back().get_id() + 1;
            // Add node
            Node n(new_id, 2);
            nodes.push_back(n);
            // last innovation
            int new_innovation = connections.size();
            // Add two new connections
            Connection c1(in_node, new_id, 1, true, new_innovation);
            Connection c2(new_id, out_node, new_weight, true, new_innovation+1);
            connections.push_back(c1);
            connections.push_back(c2);
        }

        // Print genome
        void print_genome(){
            for(int i = 0; i < connections.size(); i++){
                cout << connections[i].get_InNode() << " " << connections[i].get_OutNode() << " " << connections[i].get_weight() << " " << connections[i].get_Innovation() << endl;
            }
        }
};

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
