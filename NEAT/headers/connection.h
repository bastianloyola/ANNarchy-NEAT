#ifndef CONNECTION_H
#define CONNECTION_H

// Class of the connections between nodes
class Connection {    

  public:  
    // Class constructor
    Connection(int c_in_node, int c_out_node, float c_weight, bool c_enabled, int innovation);
    Connection();
    // Getters         
    int getInNode() const;
    int getOutNode() const;
    float getWeight() const;
    bool getEnabled();
    int getInnovation();

    // Setters 
    void setEnabled(bool new_enable);
    void setWeight(int new_weight);
    void changeEnabled();
    void changeWeight(float new_weight);

  private:
    int in;
    int out;
    float weight;        
    bool enabled;
    int innovation;
};

#endif