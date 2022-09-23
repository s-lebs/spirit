#pragma once
#ifndef SPIRIT_CORE_DATA_PARAMETERS_METHOD_MMF_HPP
#define SPIRIT_CORE_DATA_PARAMETERS_METHOD_MMF_HPP

#include <data/Parameters_Method_Solver.hpp>

#include <random>
#include <vector>

namespace Data
{

// LLG_Parameters contains all LLG information about the spin system
struct Parameters_Method_MMF : public Parameters_Method_Solver
{
    // Which mode to follow (based on some conditions)
    int n_mode_follow = 0;
    // Number of lowest modes to calculate
    int n_modes = 10;
    // Number of steps for the Gradient Decent algorithm
    int n_GD_iterations = 1;
    // Number of steps of GD between sparse hessian partial steps +1
    int n_GD_steps = 3162;
    // Is the Hamiltonian sparse
    bool sparse = false;     
    //Save computed eigenmodes in system->modes
    bool save_modes = true;

    // ----------------- Output --------------
    // Energy output settings
    bool output_energy_step                  = false;
    bool output_energy_archive               = false;
    bool output_energy_spin_resolved         = false;
    bool output_energy_divide_by_nspins      = true;
    bool output_energy_add_readability_lines = false;
    // Spin configurations output settings
    bool output_configuration_step    = false;
    bool output_configuration_archive = false;

};

} // namespace Data

#endif
