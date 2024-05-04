#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <vector>

class Parameters{
    public:
        Parameters();
        Parameters(int numberGenomes, int numberInputs, int numberOutputs, 
            float keep=0.9, float threshold=0.4,
            float probabilityInterespecies=0.001, float probabilityNoCrossoverOff=0.25,
            float probabilityWeightMutated=0.8, 
            float probabilityAddNodeSmall=0.9, float probabilityAddLink_small=0.9,
            float probabilityAddNodeLarge=0.9, float probabilityAddLink_Large=0.9,
            int largeSize=20,
            float c1=1.0, float c2=1.0, float c3=0.4, float initial_weight=110.0);
        int numberGenomes;
        int numberInputs;
        int numberOutputs;
        float keep;
        float threshold;
        float probabilityInterespecies;
        float percentageNoCrossoverOff;
        float probabilityWeightMutated;
        float probabilityAddNodeSmall;
        float probabilityAddLinkSmall;
        float probabilityAddNodeLarge;
        float probabilityAddLinkLarge;
        int largeSize;
        float c1;
        float c2;
        float c3;
        float initial_weight;

    private:
};

#endif