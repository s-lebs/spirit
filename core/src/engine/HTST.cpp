#ifndef SPIRIT_SKIP_HTST

#include <engine/HTST.hpp>
#include <engine/Hamiltonian_Heisenberg.hpp>
#include <engine/Manifoldmath.hpp>
#include <engine/Vectormath.hpp>
#include <engine/Eigenmodes.hpp>
#include <utility/Constants.hpp>
#include <utility/Logging.hpp>
#include <Spirit/Configurations.h>
#include <Spirit/Simulation.h>
#include <Spirit/System.h>
#include <data/State.hpp>
#include <Spirit/IO.h>

#include <Eigen/Core>
#include <Eigen/Dense>
// #include <Eigen/Array>
#include <Eigen/Eigenvalues>
//#include <unsupported/Eigen/CXX11/Tensor>
#include <GenEigsRealShiftSolver.h>
#include <GenEigsSolver.h> // Also includes <MatOp/DenseGenMatProd.h>

#include <fmt/format.h>
#include <fmt/ostream.h>

#include <iostream>
#include <iomanip>

using namespace std;

namespace C = Utility::Constants;

namespace Engine
{
namespace HTST
{

// Note the two images should correspond to one minimum and one saddle point
// Non-extremal images may yield incorrect Hessians and thus incorrect results
void Calculate( Data::HTST_Info & htst_info, int n_eigenmodes_keep )
{
    Log( Utility::Log_Level::All, Utility::Log_Sender::HTST, "---- Prefactor calculation" );
    htst_info.sparse           = false;
    const scalar epsilon       = 2e-3;
    const scalar epsilon_force = 4e-7;

    auto & image_minimum = *htst_info.minimum->spins;
    auto & image_sp      = *htst_info.saddle_point->spins;

    int nos = image_minimum.size();

    if( n_eigenmodes_keep < 0 )
        n_eigenmodes_keep = 2 * nos;
    n_eigenmodes_keep           = std::min( 2 * nos, n_eigenmodes_keep );
    htst_info.n_eigenmodes_keep = n_eigenmodes_keep;
    Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST,
         fmt::format( "    Saving the first {} eigenvectors.", n_eigenmodes_keep ) );

    vectorfield force_tmp( nos, { 0, 0, 0 } );
    std::vector<std::string> block;

    // TODO
    bool is_afm = false;

    // The gradient (unprojected)
    Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST,
         "    Evaluation of the gradient at the initial configuration..." );
    vectorfield gradient_minimum( nos, { 0, 0, 0 } );
    htst_info.minimum->hamiltonian->Gradient( image_minimum, gradient_minimum );

    // Check if the configuration is actually an extremum
    Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST,
         "    Checking if initial configuration is an extremum..." );
    Vectormath::set_c_a( 1, gradient_minimum, force_tmp );
    Manifoldmath::project_tangential( force_tmp, image_minimum );
    scalar fmax_minimum = Vectormath::max_norm( force_tmp );
    if( fmax_minimum > epsilon_force )
    {
        Log( Utility::Log_Level::Error, Utility::Log_Sender::All,
             fmt::format(
                 "HTST: the initial configuration is not a converged minimum, its max. torque is above the threshold "
                 "({} > {})!",
                 fmax_minimum, epsilon_force ) );
        return;
    }

    // The gradient (unprojected)
    Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST,
         "    Evaluation of the gradient at the transition configuration..." );
    vectorfield gradient_sp( nos, { 0, 0, 0 } );
    htst_info.saddle_point->hamiltonian->Gradient( image_sp, gradient_sp );

    // Check if the configuration is actually an extremum
    Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST,
         "    Checking if transition configuration is an extremum..." );
    Vectormath::set_c_a( 1, gradient_sp, force_tmp );
    Manifoldmath::project_tangential( force_tmp, image_sp );
    scalar fmax_sp = Vectormath::max_norm( force_tmp );
    if( fmax_sp > epsilon_force )
    {
        Log( Utility::Log_Level::Error, Utility::Log_Sender::All,
             fmt::format(
                 "HTST: the transition configuration is not a converged saddle point, its max. torque is above the "
                 "threshold ({} > {})!",
                 fmax_sp, epsilon_force ) );
        return;
    }

    ////////////////////////////////////////////////////////////////////////
    // Saddle point
    {
        Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST, "Calculation for the Saddle Point" );

        // Evaluation of the Hessian...
        MatrixX hessian_sp = MatrixX::Zero( 3 * nos, 3 * nos );
        Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST, "    Evaluation of the Hessian..." );
        htst_info.saddle_point->hamiltonian->Hessian( image_sp, hessian_sp );

        // Eigendecomposition
        Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST, "    Eigendecomposition..." );
        MatrixX hessian_geodesic_sp_3N = MatrixX::Zero( 3 * nos, 3 * nos );
        MatrixX hessian_geodesic_sp_2N = MatrixX::Zero( 2 * nos, 2 * nos );
        htst_info.eigenvalues_sp       = VectorX::Zero( 2 * nos );
        htst_info.eigenvectors_sp      = MatrixX::Zero( 2 * nos, 2 * nos );
        Geodesic_Eigen_Decomposition(
            image_sp, gradient_sp, hessian_sp, hessian_geodesic_sp_3N, hessian_geodesic_sp_2N, htst_info.eigenvalues_sp,
            htst_info.eigenvectors_sp );

        // Print some eigenvalues
        block = std::vector<std::string>{ "10 lowest eigenvalues at saddle point:" };
        for( int i = 0; i < 10; ++i )
            block.push_back( fmt::format(
                "ew[{}]={:^20e}   ew[{}]={:^20e}", i, htst_info.eigenvalues_sp[i], i + 2 * nos - 10,
                htst_info.eigenvalues_sp[i + 2 * nos - 10] ) );
        Log.SendBlock( Utility::Log_Level::Info, Utility::Log_Sender::HTST, block, -1, -1 );

        // Check if lowest eigenvalue < 0 (else it's not a SP)
        Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST, "Checking if actually a saddle point..." );
        if( htst_info.eigenvalues_sp[0] > -epsilon )
        {
            Log( Utility::Log_Level::Error, Utility::Log_Sender::All,
                 fmt::format(
                     "HTST: the transition configuration is not a saddle point, its lowest eigenvalue is above the "
                     "threshold ({} > {})!",
                     htst_info.eigenvalues_sp[0], -epsilon ) );
            return;
        }

        // Check if second-lowest eigenvalue < 0 (higher-order SP)
        Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST, "Checking if higher order saddle point..." );
        int n_negative = 0;
        for( int i = 0; i < htst_info.eigenvalues_sp.size(); ++i )
        {
            if( htst_info.eigenvalues_sp[i] < -epsilon )
                ++n_negative;
        }
        if( n_negative > 1 )
        {
            Log( Utility::Log_Level::Error, Utility::Log_Sender::All,
                 fmt::format( "HTST: the image you passed is a higher order saddle point (N={})!", n_negative ) );
            return;
        }

        // Perpendicular velocity
        Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST,
             "Calculating perpendicular velocity at saddle point ('a' factors)..." );
        // Calculation of the 'a' parameters...
        htst_info.perpendicular_velocity = VectorX::Zero( 2 * nos );
        MatrixX basis_sp                 = MatrixX::Zero( 3 * nos, 2 * nos );
        Manifoldmath::tangent_basis_spherical( image_sp, basis_sp );
        // Manifoldmath::tangent_basis(image_sp, basis_sp);
        // Calculate_Perpendicular_Velocity_2N(image_sp, hessian_geodesic_sp_2N, basis_sp, htst_info.eigenvectors_sp,
        // perpendicular_velocity_sp);
        Calculate_Perpendicular_Velocity(
            image_sp, htst_info.saddle_point->geometry->mu_s, hessian_geodesic_sp_3N, basis_sp,
            htst_info.eigenvectors_sp, htst_info.perpendicular_velocity );

        // Reduce the number of saved eigenmodes
        htst_info.eigenvalues_sp.conservativeResize( 2 * nos );
        htst_info.eigenvectors_sp.conservativeResize( 2 * nos, n_eigenmodes_keep );
    }
    // End saddle point
    ////////////////////////////////////////////////////////////////////////

    // Checking for zero modes at the saddle point...
    Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST, "Checking for zero modes at the saddle point..." );
    int n_zero_modes_sp = 0;
    int n_ev=htst_info.eigenvalues_sp.size();
    for( int i = 0; i < n_ev; ++i )
    {
        if( std::abs( htst_info.eigenvalues_sp[i] ) <= epsilon )
            ++n_zero_modes_sp;
    }

    // Deal with zero modes if any (calculate volume)
    htst_info.volume_sp = 1;
    if( n_zero_modes_sp > 0 )
    {
        Log( Utility::Log_Level::All, Utility::Log_Sender::HTST,
             fmt::format( "ZERO MODES AT SADDLE POINT (N={})", n_zero_modes_sp ) );

        if( is_afm )
            htst_info.volume_sp = Calculate_Zero_Volume( htst_info.saddle_point, htst_info.eigenvalues_sp, htst_info.eigenvectors_sp,epsilon, 5, &htst_info.rmode_sp);
        else
            htst_info.volume_sp = Calculate_Zero_Volume( htst_info.saddle_point, htst_info.eigenvalues_sp, htst_info.eigenvectors_sp,epsilon, 5, &htst_info.rmode_sp);
    }

    // Calculate "s"
    htst_info.s = 0;
    for( int i = n_zero_modes_sp + 1; i < 2 * nos; ++i )
        htst_info.s += std::pow( htst_info.perpendicular_velocity[i], 2 ) / htst_info.eigenvalues_sp[i];
    htst_info.s = std::sqrt( htst_info.s );

    ////////////////////////////////////////////////////////////////////////
    // Initial state minimum
    {
        Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST, "Calculation for the Minimum" );

        // Evaluation of the Hessian...
        MatrixX hessian_minimum = MatrixX::Zero( 3 * nos, 3 * nos );
        Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST, "    Evaluation of the Hessian..." );
        htst_info.minimum->hamiltonian->Hessian( image_minimum, hessian_minimum );

        // Eigendecomposition
        Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST, "    Eigendecomposition..." );
        MatrixX hessian_geodesic_minimum_3N = MatrixX::Zero( 3 * nos, 3 * nos );
        MatrixX hessian_geodesic_minimum_2N = MatrixX::Zero( 2 * nos, 2 * nos );
        htst_info.eigenvalues_min           = VectorX::Zero( 2 * nos );
        htst_info.eigenvectors_min          = MatrixX::Zero( 2 * nos, 2 * nos );
        Geodesic_Eigen_Decomposition(
            image_minimum, gradient_minimum, hessian_minimum, hessian_geodesic_minimum_3N, hessian_geodesic_minimum_2N,
            htst_info.eigenvalues_min, htst_info.eigenvectors_min );

        // Print some eigenvalues
        block = std::vector<std::string>{ "10 lowest eigenvalues at minimum:" };
        for( int i = 0; i < 10; ++i )
            block.push_back( fmt::format(
                "ew[{}]={:^20e}   ew[{}]={:^20e}", i, htst_info.eigenvalues_min[i], i + 2 * nos - 10,
                htst_info.eigenvalues_min[i + 2 * nos - 10] ) );
        Log.SendBlock( Utility::Log_Level::Info, Utility::Log_Sender::HTST, block, -1, -1 );

        // Check for eigenvalues < 0 (i.e. not a minimum)
        Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST, "Checking if actually a minimum..." );
        if( htst_info.eigenvalues_min[0] < -epsilon )
        {
            Log( Utility::Log_Level::Error, Utility::Log_Sender::All,
                 fmt::format(
                     "HTST: the initial configuration is not a minimum, its lowest eigenvalue is below the threshold "
                     "({} < {})!",
                     htst_info.eigenvalues_min[0], -epsilon ) );
            return;
        }

        // Reduce the number of saved eigenmodes
        htst_info.eigenvalues_min.conservativeResize( 2 * nos );
        htst_info.eigenvectors_min.conservativeResize( 2 * nos, n_eigenmodes_keep );
    }
    // End initial state minimum
    ////////////////////////////////////////////////////////////////////////

    // Checking for zero modes at the minimum...
    Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST, "Checking for zero modes at the minimum..." );
    int n_zero_modes_minimum = 0;
    n_ev=htst_info.eigenvalues_min.size();
    for( int i = 0; i < n_ev; ++i )
    {
        if( std::abs( htst_info.eigenvalues_min[i] ) <= epsilon )
            ++n_zero_modes_minimum;
    }

    // Deal with zero modes if any (calculate volume)
    htst_info.volume_min = 1;
    if( n_zero_modes_minimum > 0 )
    {
        Log( Utility::Log_Level::All, Utility::Log_Sender::HTST,
             fmt::format( "ZERO MODES AT MINIMUM (N={})", n_zero_modes_minimum ) );

        if( is_afm )
            htst_info.volume_min = Calculate_Zero_Volume( htst_info.minimum, htst_info.eigenvalues_min, htst_info.eigenvectors_min,epsilon, 5, &htst_info.rmode_min);//5=n_ev
        else
            htst_info.volume_min = Calculate_Zero_Volume( htst_info.minimum, htst_info.eigenvalues_min, htst_info.eigenvectors_min,epsilon, 5, &htst_info.rmode_min);
    }

    ////////////////////////////////////////////////////////////////////////
    // Calculation of the prefactor...
    Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST, "Calculating prefactor..." );

    // Calculate the exponent for the temperature-dependence of the prefactor
    //      The exponent depends on the number of zero modes at the different states
    htst_info.temperature_exponent = 0.5 * ( n_zero_modes_minimum - n_zero_modes_sp );

    // Calculate "me"
    htst_info.me = std::pow( 2 * C::Pi * C::k_B, htst_info.temperature_exponent );

    // Calculate Omega_0, i.e. the entropy contribution
    htst_info.Omega_0 = 1;
    if( n_zero_modes_minimum > n_zero_modes_sp + 1 )
    {
        for( int i = n_zero_modes_sp + 1; i < n_zero_modes_minimum; ++i )
            htst_info.Omega_0 /= std::sqrt( htst_info.eigenvalues_sp[i] );
    }
    else if( n_zero_modes_minimum < n_zero_modes_sp + 1 )
    {
        for( int i = n_zero_modes_minimum; i < ( n_zero_modes_sp + 1 ); ++i )
            htst_info.Omega_0 *= std::sqrt( htst_info.eigenvalues_min[i] );
    }
    for( int i = std::max( n_zero_modes_minimum, n_zero_modes_sp + 1 ); i < 2 * nos; ++i )
        htst_info.Omega_0 *= std::sqrt( htst_info.eigenvalues_min[i] / htst_info.eigenvalues_sp[i] );
/*
    // Calculate the prefactor
    htst_info.prefactor_dynamical = htst_info.me * htst_info.volume_sp / htst_info.volume_min * htst_info.s;
    htst_info.prefactor
        = C::g_e / ( C::hbar * 1e-12 ) * htst_info.Omega_0 * htst_info.prefactor_dynamical / ( 2 * C::Pi );

    Log.SendBlock(
        Utility::Log_Level::All, Utility::Log_Sender::HTST,
        { "---- Prefactor calculation successful!",
          fmt::format( "exponent    = {:^20e}", htst_info.temperature_exponent ),
          fmt::format( "me          = {:^20e}", htst_info.me ),
          fmt::format( "m = Omega_0 = {:^20e}", htst_info.Omega_0 ),
          fmt::format( "s           = {:^20e}", htst_info.s ),
          fmt::format( "volume_sp   = {:^20e}", htst_info.volume_sp ),
          fmt::format( "volume_min  = {:^20e}", htst_info.volume_min ),
          fmt::format( "hbar[meV*s] = {:^20e}", C::hbar * 1e-12 ),
          fmt::format( "v = dynamical prefactor = {:^20e}", htst_info.prefactor_dynamical ),
          fmt::format( "prefactor               = {:^20e}", htst_info.prefactor ) },
        -1, -1 );*/
}

void End_HTST( Data::HTST_Info & htst_info)
{
    // Calculate the prefactor
    htst_info.prefactor_dynamical = htst_info.me * htst_info.volume_sp / htst_info.volume_min * htst_info.s;
    htst_info.prefactor
        = C::g_e / ( C::hbar * 1e-12 ) * htst_info.Omega_0 * htst_info.prefactor_dynamical / ( 2 * C::Pi );

    Log.SendBlock(
        Utility::Log_Level::All, Utility::Log_Sender::HTST,
        { "---- Prefactor calculation successful!",
          fmt::format( "exponent    = {:^20e}", htst_info.temperature_exponent ),
          fmt::format( "me          = {:^20e}", htst_info.me ),
          fmt::format( "m = Omega_0 = {:^20e}", htst_info.Omega_0 ),
          fmt::format( "s           = {:^20e}", htst_info.s ),
          fmt::format( "volume_sp   = {:^20e}", htst_info.volume_sp ),
          fmt::format( "volume_min  = {:^20e}", htst_info.volume_min ),
          fmt::format( "hbar[meV*s] = {:^20e}", C::hbar * 1e-12 ),
          fmt::format( "v = dynamical prefactor = {:^20e}", htst_info.prefactor_dynamical ),
          fmt::format( "prefactor               = {:^20e}", htst_info.prefactor ) },
        -1, -1 );
}

/*
The translational zero mode volume is calculated by approximating the translational modes from the spin configuration
*/

scalar Calculate_Zero_Volume( const std::shared_ptr<Data::Spin_System> system , VectorX eigenvalues, MatrixX eigenvectors,scalar epsilon, int n_modes, bool* rot)
{
    int nos                = system->geometry->nos;
    auto & n_cells         = system->geometry->n_cells;
    auto & spins           = *system->spins;
    auto & spin_positions  = system->geometry->positions;
    auto & geometry        = *system->geometry;
    auto & bravais_vectors = system->geometry->bravais_vectors;

    // Dimensionality of the zero mode
    int zero_mode_dimensionality = 0;
    Vector3 zero_mode_length{ 0, 0, 0 };
    std::cout << "start trans " << std::endl;
    vectorfield spins_before( nos, Vector3{ 0, 0, 0 } );
    spins_before=spins;


    int dx1,dx2;
    int dy1,dy2;
    int dz1,dz2;
    int dimx=geometry.n_cells[0];
    int dimy=geometry.n_cells[1];
    int dimz=geometry.n_cells[2];
    int N=geometry.n_cell_atoms;

    MatrixX transmodes(3*nos,3);

    
    int it=0;

    // Compute the translational modes from the spin configuration:
    for(int d=0; d<nos; d++)
    {
        dx1=d+N;
        dx2=d-N;
        if(dx1%(N*dimx)==0)
        {
            dx1=dx1-N*dimx;
        }
        if(d%(N*dimx)==0)
        {
            dx2=dx2+N*dimx;
        }
        dy1=d+N*dimx;
        dy2=d-N*dimx;
        if(N*dimx*(dimy-1)<=d)
        {
            dy1=dy1-N*dimy*dimx;
        }
        if(d<dimx*N)
        {
            dy2=dy2+N*dimx*dimy;
        }
        dz1=d+N*dimy*dimx;
        dz2=d-N*dimy*dimx;
        if(N*dimx*dimy*(dimz-1)<=d)
        {
            dz1=dz1-N*dimx*dimy*dimz;
        }
        if(d<dimx*dimy)
        {
            dz2=dz2+N*dimx*dimy*dimz;
        }

        transmodes.col(0)[3*d]=0.5*(spins[dx1][0]-spins[dx2][0]);
        transmodes.col(0)[1+3*d]=0.5*(spins[dx1][1]-spins[dx2][1]);
        transmodes.col(0)[2+3*d]=0.5*(spins[dx1][2]-spins[dx2][2]);
        transmodes.col(1)[3*d]=0.5*(spins[dy1][0]-spins[dy2][0]);
        transmodes.col(1)[1+3*d]=0.5*(spins[dy1][1]-spins[dy2][1]);
        transmodes.col(1)[2+3*d]=0.5*(spins[dy1][2]-spins[dy2][2]);
        transmodes.col(2)[3*d]=0.5*(spins[dz1][0]-spins[dz2][0]);
        transmodes.col(2)[1+3*d]=0.5*(spins[dz1][1]-spins[dz2][1]);
        transmodes.col(2)[2+3*d]=0.5*(spins[dz1][2]-spins[dz2][2]);

    }

    double tnorm0=transmodes.col(0).norm();
    double tnorm1=transmodes.col(1).norm();
    double tnorm2=transmodes.col(2).norm();

   
    if( tnorm0 == 0.0 )
        tnorm0 = 1.0;
    if( tnorm1 == 0.0 )
        tnorm1 = 1.0;
    if( tnorm2 == 0.0 )
        tnorm2 = 1.0;

    // Orthogonalize the translational modes

    transmodes.col( 1 )
        = transmodes.col( 1 )
          - ( transmodes.col( 0 ).dot( transmodes.col( 1 ) ) ) / ( tnorm0*tnorm0 ) * transmodes.col( 0 );

    tnorm1=transmodes.col(1).norm();

    transmodes.col( 2 )
        = transmodes.col( 2 )
          - ( transmodes.col( 0 ).dot( transmodes.col( 2 ) ) ) / ( tnorm0*tnorm0  ) * transmodes.col( 0 )
          - ( transmodes.col( 2 ).dot(transmodes.col( 1 ) ) ) / ( tnorm1*tnorm1 ) * transmodes.col( 1 );

    
    tnorm2=transmodes.col(2).norm();

    if( tnorm1 == 0.0 )
        tnorm1 = 1.0;
    if( tnorm2 == 0.0 )
        tnorm2 = 1.0;


    MatrixX eigenmodes(3*nos,n_modes);

    SpMatrixX basis_3Nx2N   = SpMatrixX( 3 * nos, 2 * nos );

    Manifoldmath::sparse_tangent_basis_spherical( spins, basis_3Nx2N);

    // Get the eigenmodes

    for(int i=0;i<n_modes;i++)
    {
        eigenmodes.col(i)=basis_3Nx2N * eigenvectors.col(i);
        eigenmodes.col(i).normalize();
    }


    int dim=system->hamiltonian->boundary_conditions[0]+system->hamiltonian->boundary_conditions[1]+system->hamiltonian->boundary_conditions[2];

    // Find out, which eigenvalues are zero
    VectorX zms;
    for(int i=0;i<n_modes;i++)
    {
        if(epsilon>abs(eigenvalues[i]))
        {
            zms.conservativeResize(zms.size()+1);
            zms[zms.size()-1]=i;
        }
    }

    int z_modes=zms.size();

    // Find the number of translational and non-translational modes

    double sum=0;
    for(int i=0; i<z_modes;i++)
    {
        sum+=pow(transmodes.col(0).dot(eigenmodes.col(zms[i]))/tnorm0,2);
        sum+=pow(transmodes.col(1).dot(eigenmodes.col(zms[i]))/tnorm1,2);
        sum+=pow(transmodes.col(2).dot(eigenmodes.col(zms[i]))/tnorm2,2);
    }

    int n_t=round(sum);
    int n_nt=z_modes-n_t;
    scalar zero_volume = 1;

    Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST,
         fmt::format( "Found {} translational modes and {} other", n_t, n_nt ) );


    // The "handmade" transmodes can be used to calculate the ZMV
    if( n_t == dim)
    {

        for( int ibasis = 0; ibasis < 3; ++ibasis )
        {
            // Only a periodical direction can be a true zero mode
            if( system->hamiltonian->boundary_conditions[ibasis] && geometry.n_cells[ibasis] > 1 )
            {
                zero_volume *= transmodes.col( ibasis ).norm();
            }
        }
    }
    VectorX rotmode( 3 * nos );
    // If there is a non-translational mode, the zero mode volume needs to be updated later by the non-translational zero mode volume
    if( n_nt == 1)
        *rot=true;

    // Get the rotational mode (if needed for translational mode)
    if( n_nt == 1 && 0<n_t)
    {
        VectorX ntmode( 3 * nos );
        double max_rot = 0;

        for( int i = 0; i < z_modes; i++ )
        {
            ntmode = eigenmodes.col( zms[i] )
                     - ( eigenmodes.col( zms[i] ).dot( transmodes.col( 0 ) ) )/(tnorm0*tnorm0) * transmodes.col( 0 )
                     - ( eigenmodes.col( zms[i] ).dot( transmodes.col( 1 ) ) )/(tnorm1*tnorm1) * transmodes.col( 1 )
                     - ( eigenmodes.col( zms[i] ).dot( transmodes.col( 2 ) ) )/(tnorm2*tnorm2) * transmodes.col( 2 );
            if( max_rot < ntmode.dot( ntmode ) )
            {
                rotmode = ntmode;
                max_rot = ntmode.dot( ntmode );
            }
        }
        rotmode.normalize();
    }
    // At the moment, the zero mode volume can only be calculated for one non-translational mode
    else if(1<n_nt)
    {
        Log( Utility::Log_Level::Error, Utility::Log_Sender::HTST,
             fmt::format( "There is more than one non-translational mode!" ) );
        return 0;
    }


    // Move on to translational zero mode volume (if not computed yet)
    if(0 < n_t &&n_t< dim)
    {
        // Remove the rotational mode from the translational mode
        MatrixX tmodes( 3 * nos, z_modes );

        for( int i = 0; i < z_modes; i++ )
        {
            tmodes.col( i )
                = eigenmodes.col( zms[i] ) - eigenmodes.col( zms[i] ).dot( rotmode ) * eigenmodes.col( zms[i] );
        }

        // Give modes the right length

        for( int i = 0; i < z_modes; i++ )
        {
            tmodes.col( i ) = 1 / ( tmodes.col( i ).norm() * tnorm0 )
                                  * ( tmodes.col( i ).dot( transmodes.col( 0 ) ) ) * transmodes.col( 0 )
                              + 1 / ( tmodes.col( i ).norm() * tnorm1 )
                                    * ( tmodes.col( i ).dot( transmodes.col( 1 ) ) ) * transmodes.col( 1 )
                              + 1 / ( tmodes.col( i ).norm() * tnorm2)
                                    * ( tmodes.col( i ).dot( transmodes.col( 2 ) ) ) * transmodes.col( 2 );
        }

        // Orthogonalize translational modes

        if( n_t == 1 )
        {
            if( tmodes.col( 0 ).norm() < 0.1 )
            {
                tmodes.col( 0 ) = tmodes.col( 1 );
            }
        }

        if( n_t == 2 )
        {
            if( tmodes.col( 0 ).norm()  < 0.1 )
            {
                tmodes.col( 0 ) = tmodes.col( 1 );
                tmodes.col( 1 ) = tmodes.col( 2 )
                                  - 1 / ( tmodes.col( 0 ).norm() * tmodes.col( 0 ).norm() )
                                        * ( tmodes.col( 0 ).dot( tmodes.col( 2 ) ) ) * tmodes.col( 0 );
            }
            else
            {
                tmodes.col( 1 ) = tmodes.col( 1 )
                                  - 1 / ( tmodes.col( 0 ).norm() * tmodes.col( 0 ).norm() )
                                        * ( tmodes.col( 0 ).dot( tmodes.col( 1 ) ) ) * tmodes.col( 0 );
                if( tmodes.col( 1 ).norm() < 0.1 )
                {
                    tmodes.col( 1 ) = tmodes.col( 2 )
                                      - 1 / ( tmodes.col( 0 ).norm() * tmodes.col( 0 ).norm() )
                                            * ( tmodes.col( 0 ).dot( tmodes.col( 2 ) ) ) * tmodes.col( 0 );
                }
            }
        }


        for( int i = 0; i < n_t; i++ )
        {
            zero_volume *= tmodes.col( i ).norm();
            std::cout <<"zm " <<i<<": "<<tmodes.col( i ).norm()<< std::endl;
        }
    }

    Log.SendBlock(
        Utility::Log_Level::Info, Utility::Log_Sender::HTST,
        { fmt::format( "ZV zero mode dimensionality = {}", zero_mode_dimensionality ),
          fmt::format( "ZV         zero mode length = {}", zero_mode_length.transpose() ),
          fmt::format( "ZV = {}", zero_volume ) },
        -1, -1 );

    std::cout << "end trans " << std::endl;
    // Return
    return zero_volume;
}

/*
In case that there is a non-translational zero mode, this iterative method updates the zero mode volume
*/

scalar UpdateZMV(State * state, int idx_image_minimum, int idx_image_sp, int idx_chain, char type)
{
    std::shared_ptr<Data::Spin_System> image;
    std::shared_ptr<Data::Spin_System_Chain> chain;
    int idx_image;
    if(type=='s')
    {
        from_indices( state, idx_image_sp, idx_chain, image, chain );

        idx_image = idx_image_sp;  
    }
    if(type=='m')
    {
        from_indices( state, idx_image_minimum, idx_chain, image, chain );

        idx_image = idx_image_minimum;
    }

    int nos = image->geometry->nos;
    auto & geometry = *image->geometry;
    //auto & spins= *image->spins;
    //std::shared_ptr<vectorfield> spins(nos,3);
    vectorfield positions( nos, Vector3{ 0.0, 0.0, 0.0 } );
    positions=(image->geometry->positions);
    Vector3 center={0.0,0.0,0.0};
    double mass=0.0;
    Vector3 centerN={0.0,0.0,0.0};
    double massN=0.0;
    Vector3 diff={0.0,0.0,0.0};
    vectorfield spinsN( nos, Vector3{ 0.0, 0.0, 0.0 } );
    vectorfield spins( nos, Vector3{ 0.0, 0.0, 0.0 } );
    spins=*image->spins;//( *image->spins )[0].data();
    vectorfield starting_spins( nos, Vector3{ 0.0, 0.0, 0.0 } );
    starting_spins=spins;
    vectorfield NewSpins( nos, Vector3{ 0.0, 0.0, 0.0 } );
    auto & bravais_vectors = image->geometry->bravais_vectors;

    std::cout << "start rot " << std::endl;

    auto p = image->mmf_parameters;
    
    if(p!=0)
    {

        image->Lock();
        p->n_mode_follow = 0;
        image->Unlock();
                Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST,
         fmt::format( "MMF currently follows mode {}. Set MMF mode to follow = {}", p->n_mode_follow, image->mmf_parameters->n_mode_follow) );
    }
    

    // Calculate the eigenmodes using the SHSM method
    Engine::Eigenmodes::Calculate_Eigenmodes( image, idx_image, idx_chain );

    double norm=0.0;
    double prevNorm=-1.0;
    double spNorm=-2.0;
    int it=0;

    double Volume=0;
    double dVol;

    VectorX ntmode( 3 * nos );
    VectorX rotmode( 3 * nos );
    VectorX last_mode( 3 * nos );

    int z_modes;
    double max_rot;
    int rot_idx;
    int n_modes;


    // Apply the non-translational mode, until the initial configuration is reached again
    while(spNorm<prevNorm || prevNorm>norm)
    {
        // Update the eigenmodes using Gradient Decent
        Engine::Eigenmodes::Calculate_EigenmodesGD( image, idx_image, idx_chain ,20000);
        //Engine::Eigenmodes::Calculate_Eigenmodes( image, idx_image, idx_chain );
        

        int dx1, dx2;
        int dy1,dy2;
        int dz1,dz2;
        int dimx=geometry.n_cells[0];
        int dimy=geometry.n_cells[1];
        int dimz=geometry.n_cells[2];
        int N=geometry.n_cell_atoms;

        MatrixX transmodes(3*nos,3);

        // Compute the translational modes from the lattice:
        for( int d = 0; d < nos; d++ )
        {
            dx1 = d + N;
            dx2 = d - N;
            if( dx1 % ( N * dimx ) == 0 )
            {
                dx1 = dx1 - N * dimx;
            }
            if( d % ( N * dimx ) == 0 )
            {
                dx2 = dx2 + N * dimx;
            }
            dy1 = d + N * dimx;
            dy2 = d - N * dimx;
            if( N * dimx * ( dimy - 1 ) <= d )
            {
                dy1 = dy1 - N * dimy * dimx;
            }
            if( d < dimx * N )
            {
                dy2 = dy2 + N * dimx * dimy;
            }
            dz1 = d + N * dimy * dimx;
            dz2 = d - N * dimy * dimx;
            if( N * dimx * dimy * ( dimz - 1 ) <= d )
            {
                dz1 = dz1 - N * dimx * dimy * dimz;
            }
            if( d < dimx * dimy )
            {
                dz2 = dz2 + N * dimx * dimy * dimz;
            }

            transmodes.col( 0 )[3 * d]     = spins[dx1][0] - spins[dx2][0];
            transmodes.col( 0 )[1 + 3 * d] = spins[dx1][1] - spins[dx2][1];
            transmodes.col( 0 )[2 + 3 * d] = spins[dx1][2] - spins[dx2][2];
            transmodes.col( 1 )[3 * d]     = spins[dy1][0] - spins[dy2][0];
            transmodes.col( 1 )[1 + 3 * d] = spins[dy1][1] - spins[dy2][1];
            transmodes.col( 1 )[2 + 3 * d] = spins[dy1][2] - spins[dy2][2];
            transmodes.col( 2 )[3 * d]     = spins[dz1][0] - spins[dz2][0];
            transmodes.col( 2 )[1 + 3 * d] = spins[dz1][1] - spins[dz2][1];
            transmodes.col( 2 )[2 + 3 * d] = spins[dz1][2] - spins[dz2][2];
        }

        double tnorm0 = transmodes.col( 0 ).norm();
        double tnorm1 = transmodes.col( 1 ).norm();
        double tnorm2 = transmodes.col( 2 ).norm();

        if( tnorm0 == 0.0 )
            tnorm0 = 1.0;
        if( tnorm1 == 0.0 )
            tnorm1 = 1.0;
        if( tnorm2 == 0.0 )
            tnorm2 = 1.0;
        

        transmodes.col( 1 )
            = transmodes.col( 1 )
              - ( transmodes.col( 0 ).dot( transmodes.col( 1 ) ) ) / ( tnorm0 * tnorm0 ) * transmodes.col( 0 );

        tnorm1 = transmodes.col( 1 ).norm();
        if( tnorm1 == 0.0 )
            tnorm1 = 1.0;

        transmodes.col( 2 )
            = transmodes.col( 2 )
              - ( transmodes.col( 0 ).dot( transmodes.col( 2 ) ) ) / ( tnorm0 * tnorm0 ) * transmodes.col( 0 )
              - ( transmodes.col( 2 ).dot( transmodes.col( 1 ) ) ) / ( tnorm1 * tnorm1 ) * transmodes.col( 1 );

        tnorm2 = transmodes.col( 2 ).norm();

        if( tnorm2 == 0.0 )
            tnorm2 = 1.0;

        VectorX zms;

        // Find the non-translational mode as the mode with the biggest norm after the translational modes were removed
        if(it==0)
        {
            n_modes=image->eigenvalues.size();
            // Find out, which eigenvalues are zero

            for( int i=0; i < n_modes; i++ )
            {
                if( 1e-6 > abs( image->eigenvalues[i] ) )
                {
                    zms.conservativeResize( zms.size() + 1 );
                    zms[zms.size() - 1] = i;
                }
            }

            std::cout <<zms<< std::endl;

            z_modes=zms.size();
            max_rot = 0;
            rot_idx=0;

            for( int i = 0; i < z_modes; i++ )
            {
                for( int k = 0; k < nos; k++ )
                {
                    for( int j = 0; j < 3; j++ )
                    {
                        ntmode[3 * k + j] = ( *image->modes[zms[i]] )[k][j];
                    }
                }

                ntmode = ntmode                             
                    - ( ntmode.dot( transmodes.col( 0 ) ) )/(tnorm0*tnorm0) * transmodes.col( 0 )
                    - ( ntmode.dot( transmodes.col( 1 ) ) )/(tnorm1*tnorm1) * transmodes.col( 1 )
                    - ( ntmode.dot( transmodes.col( 2 ) ) )/(tnorm2*tnorm2) * transmodes.col( 2 ); 
                if( max_rot < abs(ntmode.dot( ntmode ) ))
                {
                    rot_idx=i;
                    rotmode = ntmode;
                    max_rot = ntmode.dot( ntmode );
                }
            }
            rotmode.normalize();
            
        }


        // For the higher iterations the non-translational mode is determined as the mode with the biggest scalar product with the last non-translational mode
        if(it!=0)
        {
            n_modes=image->eigenvalues.size();

            // Find out, which eigenvalues are zero
            for( int i=0; i < n_modes; i++ )
            {
                if( 1e-5 > abs( image->eigenvalues[i] ) )
                {
                    zms.conservativeResize( zms.size() + 1 );
                    zms[zms.size() - 1] = i;
                }
            }
            max_rot = 0;
            for( int i = 0; i < z_modes; i++ )
            {
                for( int k = 0; k < nos; k++ )
                {
                    for( int j = 0; j < 3; j++ )
                    {
                        ntmode[3*k+j]=(*image->modes[zms[i]])[k][j];
                    }
                }

                if( max_rot < abs(ntmode.dot( last_mode ) ))
                {
                    rotmode = ntmode;
                    max_rot = abs(ntmode.dot( last_mode ));
                    rot_idx = i;
                }
            }
            rotmode = rotmode
                    - ( rotmode.dot( transmodes.col( 0 ) ) )/(tnorm0*tnorm0) * transmodes.col( 0 )
                    - ( rotmode.dot( transmodes.col( 1 ) ) )/(tnorm1*tnorm1) * transmodes.col( 1 )
                    - ( rotmode.dot( transmodes.col( 2 ) ) )/(tnorm2*tnorm2) * transmodes.col( 2 );
            rotmode.normalize();
              
        }
        
        if(rotmode.dot(last_mode)<0 && it!=0)
        {
            rotmode=-rotmode;
            last_mode=rotmode;
        }
        else
        {
            last_mode=rotmode;
        }
        
        vectorfield rotfield( nos );
        for( int k = 0; k < nos; k++ )
        {
            for( int j = 0; j < 3; j++ )
            {
                rotfield[k][j]=rotmode[k*3+j];
            }
        }

        // Line 917-1015 is a first try to be able to remove the translational modes better from the non-translational modes, it does not work yet

        center={0.0,0.0,0.0};
        mass=0.0;

        for(int i=0; i< nos; i++)
        {
            center=center+positions[i]*spins[i][2];
            mass=mass+spins[i][2];
        }

        center=center/mass;

        scalarfield angles( nos );
        vectorfield axes( nos );

        // Find the angles and axes of rotation
        for( int idx = 0; idx < nos; idx++ )
        {
            angles[idx] = rotfield[idx].norm();
            axes[idx]   = spins[idx].cross( rotfield[idx] ).normalized();
        }

        // Scale the angles
        Engine::Vectormath::scale( angles, 1 );

        // Rotate around axes by certain angles
        Engine::Vectormath::rotate( spins, axes, angles, spinsN );



        centerN={0.0,0.0,0.0};
        massN=0.0;

        for(int i=0; i< nos; i++)
        {
            centerN=centerN+positions[i]*spinsN[i][2];
            massN=massN+spinsN[i][2];
        }

        centerN=centerN/massN; 

        std::cout <<center[0]<< " "<<centerN[0] <<" "<<diff[0]<< std::endl;
        std::cout <<center[1]<< " "<<centerN[1] <<" "<<diff[1]<< std::endl;
        std::cout <<center[2]<< " "<<centerN[2] <<" "<<diff[2]<< std::endl;
        std::cout <<" "<< std::endl;

        diff=(centerN-center);

        while( 0.00001 < diff.norm() )
        {

            double displacementA = diff.dot( bravais_vectors[0].normalized() );
            double displacementB = diff.dot((bravais_vectors[1]-(bravais_vectors[1].dot(bravais_vectors[0].normalized()))*bravais_vectors[0].normalized()).normalized())/( bravais_vectors[0].normalized().cross( bravais_vectors[1].normalized() ) ).norm();
            displacementA = displacementA / bravais_vectors[0].norm();

            rotmode = rotmode - transmodes.col( 0 ) * displacementA - transmodes.col( 1 ) * displacementB;

            for( int k = 0; k < nos; k++ )
            {
                for( int j = 0; j < 3; j++ )
                {
                    rotfield[k][j] = rotmode[k * 3 + j];
                }
            }

            // Find the angles and axes of rotation
            for( int idx = 0; idx < nos; idx++ )
            {
                angles[idx] = rotfield[idx].norm();
                axes[idx]   = spins[idx].cross( rotfield[idx] ).normalized();
            }

            // Scale the angles
            Engine::Vectormath::scale( angles, 1 );

            // Rotate around axes by certain angles
            Engine::Vectormath::rotate( spins, axes, angles, spinsN );

            centerN = { 0.0, 0.0, 0.0 };
            massN   = 0.0;

            for( int i = 0; i < nos; i++ )
            {
                centerN = centerN + positions[i] * spinsN[i][2];
                massN   = massN + spinsN[i][2];
            }

            centerN = centerN / massN;


            diff = ( centerN - center );

            std::cout <<center[0]<< " "<<centerN[0] <<" "<<diff[0]<< std::endl;
            std::cout <<center[1]<< " "<<centerN[1] <<" "<<diff[1]<< std::endl;
            std::cout <<center[2]<< " "<<centerN[2] <<" "<<diff[2]<< std::endl;
            std::cout <<" "<< std::endl;

        }

        // Apply the non-translational mode that was determined previously 
                
        *image->modes[zms[rot_idx]]=rotfield;
        Configuration_Displace_Eigenmode(state, zms[rot_idx], idx_image, idx_chain );
        std::cout << norm<<" "<<prevNorm<<" "<<spNorm<< std::endl;

        spins=*image->spins;
        center={0.0,0.0,0.0};
        mass=0.0;

        for(int i=0; i< nos; i++)
        {
            center=center+positions[i]*spins[i][2];
            mass=mass+spins[i][2];
        }

        center=center/mass;

        std::cout <<center<< " "<<centerN << std::endl;

        // Move the configuration back to the saddle point/minimum configuration by MMF and LLG simulation 

        if(type=='s')
        {
            Simulation_MMF_Start( state, 5, 20000, -1, false, nullptr, idx_image, idx_chain );
        }
        if(type=='m')
        {
            Simulation_LLG_Start(state, 7, 20000, -1, false, nullptr, idx_image, idx_chain );
        }

        // Update the zero mode volume as the changes is the spins

        NewSpins=*image->spins;                            
        //dVol=(spins-NewSpins).norm();
        dVol=0;
        for( int i = 0; i < nos; i++ )
        {
            for( int j = 0; j < 3; j++ )
            {
                dVol=dVol+(spins[i][j]-NewSpins[i][j])*(spins[i][j]-NewSpins[i][j]);
            }
        }
        dVol=sqrt(dVol);

        spins=NewSpins;

        spNorm=prevNorm;
        prevNorm=norm;
        norm=0;
        for( int i = 0; i < nos; i++ )
        {
            for( int j = 0; j < 3; j++ )
            {
                norm=norm+(spins[i][j]-starting_spins[i][j])*(spins[i][j]-starting_spins[i][j]);
            }
        }
        norm=sqrt(norm);

        std::cout <<dVol<<" "<< norm<<" "<<prevNorm<<" "<<spNorm<< std::endl;

        Volume=Volume+dVol;
        std::cout <<"Volume: "<<Volume<< std::endl;
        it++;
    }
    
    dVol   = 0;
    for( int i = 0; i < nos; i++ )
    {
        for( int j = 0; j < 3; j++ )
        {
            dVol = dVol + ( spins[i][j] - starting_spins[i][j] ) * ( spins[i][j] - starting_spins[i][j] );
        }
    }
    dVol = sqrt( dVol );

    Volume=Volume-dVol;

    return Volume;
}


void Calculate_Perpendicular_Velocity(
    const vectorfield & spins, const scalarfield & mu_s, const MatrixX & hessian, const MatrixX & basis,
    const MatrixX & eigenbasis, VectorX & perpendicular_velocity )
{
    int nos = spins.size();

    Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST,
         "  Calculate_Perpendicular_Velocity: calculate velocity matrix" );

    // Calculate the velocity matrix in the 3N-basis
    MatrixX velocity( 3 * nos, 3 * nos );
    Calculate_Dynamical_Matrix( spins, mu_s, hessian, velocity );

    Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST,
         "  Calculate_Perpendicular_Velocity: project velocity matrix" );

    // Project the velocity matrix into the 2N tangent space
    MatrixX velocity_projected( 2 * nos, 2 * nos );
    velocity_projected = basis.transpose() * velocity * basis;

    Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST, "  Calculate_Perpendicular_Velocity: calculate a" );

    // The velocity components orthogonal to the dividing surface
    perpendicular_velocity = eigenbasis.col( 0 ).transpose() * ( velocity_projected * eigenbasis );

    // std::cerr << "  Calculate_Perpendicular_Velocity: sorting" << std::endl;
    // std::sort(perpendicular_velocity.data(),perpendicular_velocity.data()+perpendicular_velocity.size());

    std::vector<std::string> block( 0 );
    for( int i = 0; i < 10; ++i )
        block.push_back( fmt::format( "  a[{}] = {}", i, perpendicular_velocity[i] ) );
    Log.SendBlock( Utility::Log_Level::Info, Utility::Log_Sender::HTST, block, -1, -1 );

    // std::cerr << "without units:" << std::endl;
    // for (int i=0; i<10; ++i)
    //     std::cerr << "  a[" << i << "] = " << perpendicular_velocity[i]/C::mu_B << std::endl;
}

void Calculate_Dynamical_Matrix(
    const vectorfield & spins, const scalarfield & mu_s, const MatrixX & hessian, MatrixX & velocity )
{
    velocity.setZero();
    int nos = spins.size();

    for( int i = 0; i < nos; ++i )
    {
        Vector3 beff{ 0, 0, 0 };

        for( int j = 0; j < nos; ++j )
        {
            velocity( 3 * i, 3 * j )
                = spins[i][1] * hessian( 3 * i + 2, 3 * j ) - spins[i][2] * hessian( 3 * i + 1, 3 * j );
            velocity( 3 * i, 3 * j + 1 )
                = spins[i][1] * hessian( 3 * i + 2, 3 * j + 1 ) - spins[i][2] * hessian( 3 * i + 1, 3 * j + 1 );
            velocity( 3 * i, 3 * j + 2 )
                = spins[i][1] * hessian( 3 * i + 2, 3 * j + 2 ) - spins[i][2] * hessian( 3 * i + 1, 3 * j + 2 );

            velocity( 3 * i + 1, 3 * j )
                = spins[i][2] * hessian( 3 * i, 3 * j ) - spins[i][0] * hessian( 3 * i + 2, 3 * j );
            velocity( 3 * i + 1, 3 * j + 1 )
                = spins[i][2] * hessian( 3 * i, 3 * j + 1 ) - spins[i][0] * hessian( 3 * i + 2, 3 * j + 1 );
            velocity( 3 * i + 1, 3 * j + 2 )
                = spins[i][2] * hessian( 3 * i, 3 * j + 2 ) - spins[i][0] * hessian( 3 * i + 2, 3 * j + 2 );

            velocity( 3 * i + 2, 3 * j )
                = spins[i][0] * hessian( 3 * i + 1, 3 * j ) - spins[i][1] * hessian( 3 * i, 3 * j );
            velocity( 3 * i + 2, 3 * j + 1 )
                = spins[i][0] * hessian( 3 * i + 1, 3 * j + 1 ) - spins[i][1] * hessian( 3 * i, 3 * j + 1 );
            velocity( 3 * i + 2, 3 * j + 2 )
                = spins[i][0] * hessian( 3 * i + 1, 3 * j + 2 ) - spins[i][1] * hessian( 3 * i, 3 * j + 2 );

            beff -= hessian.block<3, 3>( 3 * i, 3 * j ) * spins[j];
        }

        velocity( 3 * i, 3 * i + 1 ) -= beff[2];
        velocity( 3 * i, 3 * i + 2 ) += beff[1];
        velocity( 3 * i + 1, 3 * i ) += beff[2];
        velocity( 3 * i + 1, 3 * i + 2 ) -= beff[0];
        velocity( 3 * i + 2, 3 * i ) -= beff[1];
        velocity( 3 * i + 2, 3 * i + 1 ) += beff[0];

        velocity.row( 3 * i ) /= mu_s[i];
        velocity.row( 3 * i + 1 ) /= mu_s[i];
        velocity.row( 3 * i + 2 ) /= mu_s[i];
    }
}

void hessian_bordered_3N(
    const vectorfield & image, const vectorfield & gradient, const MatrixX & hessian, MatrixX & hessian_out )
{
    // Calculates a 3Nx3N matrix in the bordered Hessian approach and transforms it into the tangent basis,
    // making the result a 2Nx2N matrix. The bordered Hessian's Lagrange multipliers assume a local extremum.

    int nos     = image.size();
    hessian_out = hessian;

    VectorX lambda( nos );
    for( int i = 0; i < nos; ++i )
        lambda[i] = image[i].normalized().dot( gradient[i] );

    for( int i = 0; i < nos; ++i )
    {
        for( int j = 0; j < 3; ++j )
        {
            hessian_out( 3 * i + j, 3 * i + j ) -= lambda[i];
        }
    }
}

// NOTE WE ASSUME A SELFADJOINT MATRIX
void Eigen_Decomposition( const MatrixX & matrix, VectorX & evalues, MatrixX & evectors )
{
    // Create a Spectra solver
    Eigen::SelfAdjointEigenSolver<MatrixX> matrix_solver( matrix );
    evalues  = matrix_solver.eigenvalues().real();
    evectors = matrix_solver.eigenvectors().real();
}

void Eigen_Decomposition_Spectra(
    int nos, const MatrixX & matrix, VectorX & evalues, MatrixX & evectors, int n_decompose = 1 )
{
    int n_steps = std::max( 2, nos );

    //      Create a Spectra solver
    Spectra::DenseGenMatProd<scalar> op( matrix );
    Spectra::GenEigsSolver<scalar, Spectra::SMALLEST_REAL, Spectra::DenseGenMatProd<scalar>> matrix_spectrum(
        &op, n_decompose, n_steps );
    matrix_spectrum.init();

    //      Compute the specified spectrum
    int nconv = matrix_spectrum.compute();

    if( matrix_spectrum.info() == Spectra::SUCCESSFUL )
    {
        evalues  = matrix_spectrum.eigenvalues().real();
        evectors = matrix_spectrum.eigenvectors().real();
        // Eigen::Ref<VectorX> evec = evectors.col(0);
    }
    else
    {
        Log( Utility::Log_Level::Error, Utility::Log_Sender::All, "Failed to calculate eigenvectors of the Matrix!" );
        evalues.setZero();
        evectors.setZero();
    }
}

void Geodesic_Eigen_Decomposition(
    const vectorfield & image, const vectorfield & gradient, const MatrixX & hessian, MatrixX & hessian_geodesic_3N,
    MatrixX & hessian_geodesic_2N, VectorX & eigenvalues, MatrixX & eigenvectors )
{
    Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST, "---------- Geodesic Eigen Decomposition" );

    int nos = image.size();

    // Calculate geodesic Hessian in 3N-representation
    hessian_geodesic_3N = MatrixX::Zero( 3 * nos, 3 * nos );
    hessian_bordered_3N( image, gradient, hessian, hessian_geodesic_3N );

    // Transform into geodesic Hessian
    Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST, "    Transforming Hessian into geodesic Hessian..." );
    hessian_geodesic_2N = MatrixX::Zero( 2 * nos, 2 * nos );
    // Manifoldmath::hessian_bordered(image, gradient, hessian, hessian_geodesic_2N);
    // Manifoldmath::hessian_projected(image, gradient, hessian, hessian_geodesic_2N);
    // Manifoldmath::hessian_weingarten(image, gradient, hessian, hessian_geodesic_2N);
    // Manifoldmath::hessian_spherical(image, gradient, hessian, hessian_geodesic_2N);
    // Manifoldmath::hessian_covariant(image, gradient, hessian, hessian_geodesic_2N);

    // Do this manually
    MatrixX basis = MatrixX::Zero( 3 * nos, 2 * nos );
    Manifoldmath::tangent_basis_spherical( image, basis );
    // Manifoldmath::tangent_basis(image, basis);
    hessian_geodesic_2N = basis.transpose() * hessian_geodesic_3N * basis;

    // Calculate full eigenspectrum
    Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST, "    Calculation of full eigenspectrum..." );
    // std::cerr << hessian_geodesic_2N.cols() << "   " << hessian_geodesic_2N.rows() << std::endl;
    eigenvalues  = VectorX::Zero( 2 * nos );
    eigenvectors = MatrixX::Zero( 2 * nos, 2 * nos );
    Eigen_Decomposition( hessian_geodesic_2N, eigenvalues, eigenvectors );
    // Eigen_Decomposition_Spectra(hessian_geodesic_2N, eigenvalues, eigenvectors);

    Log( Utility::Log_Level::Info, Utility::Log_Sender::HTST, "---------- Geodesic Eigen Decomposition Done" );
}

} // end namespace HTST
} // end namespace Engine

#endif