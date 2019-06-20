#include "interfaces.hpp"
#include "MR_likelihood.hpp"
#include <iostream>

int main(int argc, char *argv[])
{
    if (argc > 2) {
        std::string input_file = argv[1];
        std::string config_file = argv[2];

        set_ini(config_file); // pass config file to likelihood computation
        run_polychord(loglikelihood,setup_loglikelihood,input_file) ;
        return 0;
    }
    else{
        std::cerr << "polychord_MR should be called with two arguments specifying INI configuration files: the first file contains the algorithm configuration and the second file contains the model configuration" << std::endl;
        return 1;
    }
}

