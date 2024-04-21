#include "../headers/node.h"

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
