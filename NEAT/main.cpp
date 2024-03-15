#include <iostream>
#include <vector>

using namespace std;

class Connection {
    int in_node;
    int out_node;
    float weight; //no se si es float         
    bool enable;
    int innovation;    

  public:  

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

class Node {       
    int id;
    int type; //1, 2 o 3

    public:

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

class Genome {     // Falta poner el constructor  
    vector<Connection> connections;
    vector<Node> nodes;


  public:  

    Genome(){
        connections = vector<Connection>();
        nodes = vector<Node>();
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
    }

    void set_connections(vector<Connection> new_connections){
        connections = new_connections;
    }

    void set_nodes(vector<Node> new_nodes){
        nodes = new_nodes;
    }

    //Mutators
    
    //Change weight, this depends
    void change_weight(int innovation, float new_weight){
            connections[innovation-1].set_weight(new_weight);
    }

    //Create new connection
    void create_connection(int in_node, int out_node, float new_weight){
        int new_innovation = connections.size() + 1;
        Connection c(in_node, out_node, new_weight, true, new_innovation);
        connections.push_back(c);
    }

    //Create new node
    void create_node(int in_node, int out_node){
        //Find connection and disable
        float new_weight = 1; //Valor default en caso que no exista una conexion previa
        for(int i = 0; i < connections.size(); i++){
            if(connections[i].get_InNode() == in_node && connections.front().get_OutNode() == out_node){
                connections[i].set_enable(false);
                new_weight = connections[i].get_weight();
            }
        }
        //get last id
        int new_id = nodes.back().get_id() + 1;
        //Add node
        Node n(new_id, 2);
        nodes.push_back(n);
        //Add two new connections
        Connection c1(in_node, new_id, 1, true, 1);
        Connection c2(new_id, out_node, new_weight, true, 1);
        connections.push_back(c1);
        connections.push_back(c2);

    
    }

    //Print genome
    void print_genome(){
        for(int i = 0; i < connections.size(); i++){
            cout << connections[i].get_InNode() << " " << connections[i].get_OutNode() << " " << connections[i].get_weight() << connections[i].get_Innovation() << endl;
        }

    }


    
};

void testing_all_classes_and_methods() {
    // testing all classes and methods
    Connection c(1, 2, 0.5, true, 1);
    cout << c.get_InNode() << endl;
    cout << c.get_OutNode() << endl;
    cout << c.get_weight() << endl;
    cout << c.get_enable() << endl;
    cout << c.get_Innovation() << endl;

    Node n(1, 1);
    Node n2(2, 1);
    cout << n.get_id() << endl;
    cout << n.get_type() << endl;

    Genome g;
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
  Genome g;
  do {
    cout << "Choose an option:  a. create_connection  b. create_node  c. change_weight  d. print_genome" << endl;
    cin >> option;
    switch (option) {
    case 'a':
      cout << "Enter in_node, out_node, new_weight: ";
      cin >> in_node >> out_node >> new_weight;
      g.create_connection(in_node, out_node, new_weight);
      break;
    case 'b':
      //select connection
      cout << "Enter in_node, out_node: ";
      cin >> in_node >> out_node;
      g.create_node(in_node, out_node);
      break;
    case 'c':
      cout << "Enter in_node, out_node, new_weight: ";
      cin >> innovation >> new_weight;
      g.change_weight(innovation, new_weight);
      break;
    case 'd':
      g.print_genome();
      break;
  }
  } while (option != 'e');
  
}

int main() {
    testing_all_classes_and_methods();
    menu();
  return 0;
}
