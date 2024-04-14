#ifndef CONNECTION_H
#define CONNECTION_H

// Clase de las conexiones entre nodos
class Connection {    

  public:  
    // Constructor de la clase
    Connection(int c_in_node, int c_out_node, float c_weight, bool c_enable, int c_innovation);

    // Getters         
    int get_InNode() const;
    int get_OutNode() const;
    float get_weight() const;
    bool get_enable();
    int get_Innovation();

    // Setters 
    void set_enable(bool new_enable);
    void set_weight(int new_weight);
    void change_enable();

    private:
        int in_node;
        int out_node;
        float weight;        
        bool enable;
        int innovation;
};

#endif