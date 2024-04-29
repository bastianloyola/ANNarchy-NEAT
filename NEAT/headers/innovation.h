#ifndef INNOVATION_H
#define INNOVATION_H

#include <vector>


struct Split{
    int in;
    int out;
    int id;
};
struct Link{
    int in;
    int out;
    int id;
};

class Innovation{
    public:
        Innovation();
        Innovation(int in, int out);
        std::vector<Split> splits;
        std::vector<Link> links;

        int addConnection(int in, int out);
        int addNode(int in, int out);

    private:
        int maxNode;
        int maxConnection;
};

#endif