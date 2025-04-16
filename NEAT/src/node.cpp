#include "../headers/node.h"

// Class constructor
Node::Node(int c_id, int c_type){
    id = c_id;
    type = c_type;
}

Node::Node(){
}

// Getters
int Node::get_id(){
    return id;
}
int Node::get_type(){
    return type;
}
