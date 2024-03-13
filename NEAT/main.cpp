#include <iostream>
#include <queue>

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
};

class Node {       
    int id;
    int type; //1, 2 o 3

    public:

        Node(int c_id, int c_type){
            id = c_id; // tenemos que tener una variable global que permita ir poniendo este valor 
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

class genome {     // Falta poner el constructor  
    queue<Connection> connections;
    queue<Node> nodes;

  public:             
    
};


int main() {
  return 0;
}
