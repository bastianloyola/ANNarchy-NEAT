#include "../headers/parameters.h"
#include <vector>
#include <iostream>
Parameters::Parameters(){}
Parameters::Parameters(int numberGenomes, int numberInputs, int numberOutputs, float keep, float threshold,
            float probabilityInterespecies, float percentageNoCrossoverOff, float probabilityWeightMutated, 
            float probabilityAddNodeSmall, float probabilityAddLinkSmall, float probabilityAddNodeLarge, float probabilityAddLinkLarge,
            int largeSize, float c1, float c2, float c3)
    :numberGenomes(numberGenomes),numberInputs(numberInputs),numberOutputs(numberOutputs),keep(keep),threshold(threshold),
    probabilityInterespecies(probabilityInterespecies),percentageNoCrossoverOff(percentageNoCrossoverOff),
    probabilityWeightMutated(probabilityWeightMutated),probabilityAddLinkSmall(probabilityAddLinkSmall),
    probabilityAddLinkLarge(probabilityAddLinkLarge),probabilityAddNodeSmall(probabilityAddNodeSmall),
    probabilityAddNodeLarge(probabilityAddNodeLarge),largeSize(largeSize),c1(c1),c2(c2),c3(c3){}
