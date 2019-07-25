#include "interfaces.hpp"
#include "MR_likelihood.hpp"
#include <iostream>

int main(int argc, char *argv[])
{
    if (argc == 2) {
        std::string config_file = argv[1];

        set_ini(config_file); // pass config file to likelihood computation
        run_polychord(loglikelihood,setup_loglikelihood, config_file) ;
        return 0;
    }
    else{
        std::cerr << "polychord_MR should be called with exactly one file specifying a INI configuration file, which should be created using the functions provided in the BayesMR package" << std::endl;
        return 1;
    }
}

