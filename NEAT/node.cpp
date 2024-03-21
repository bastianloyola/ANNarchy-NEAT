#include "node.h"

// Constructor de la clase
Node::Node(int c_id, int c_type){
    id = c_id;
    type = c_type;
}

// Getters
int Node::get_id(){
    return id;
}
int Node::get_type(){
    return type;
}

/*

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
*/