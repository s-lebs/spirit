#include <engine/Eigenmodes.hpp>
#include <engine/Manifoldmath.hpp>
#include <engine/Vectormath.hpp>
// #include <engine/Backend_par.hpp>

#include <MatOp/SparseSymMatProd.h> // Also includes <MatOp/DenseSymMatProd.h>
#include <SymEigsSolver.h>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <fmt/format.h>
#include <fmt/ostream.h>

#include <iostream>
#include <fstream>
#include <math.h>

using Utility::Log_Level;
using Utility::Log_Sender;

namespace Engine
{
namespace Eigenmodes
{

void Check_Eigenmode_Parameters( std::shared_ptr<Data::Spin_System> system )
{
    int nos        = system->nos;
    auto & n_modes = system->ema_parameters->n_modes;
    if( n_modes > 2 * nos - 2 )
    {
        n_modes = 2 * nos - 2;
        system->modes.resize( n_modes );
        system->eigenvalues.resize( n_modes );

        Log( Log_Level::Warning, Log_Sender::EMA,
             fmt::format(
                 "Number of eigenmodes declared in "
                 "EMA Parameters is too large. The number is set to {}",
                 n_modes ) );
    }
    if( n_modes != system->modes.size() )
        system->modes.resize( n_modes );

    // Initial check of selected_mode
    auto & n_mode_follow = system->ema_parameters->n_mode_follow;
    if( n_mode_follow > n_modes - 1 )
    {
        Log( Log_Level::Warning, Log_Sender::EMA,
             fmt::format(
                 "Eigenmode number {} is not "
                 "available. The largest eigenmode ({}) is used instead",
                 n_mode_follow, n_modes - 1 ) );
        n_mode_follow = n_modes - 1;
    }
}

void Calculate_Eigenmodes( std::shared_ptr<Data::Spin_System> system, int idx_img, int idx_chain )
{
    int nos = system->nos;

    //Check_Eigenmode_Parameters( system );

    auto & n_modes = system->ema_parameters->n_modes;

    // vectorfield mode(nos, Vector3{1, 0, 0});
    vectorfield spins_initial = *system->spins;

    Log( Log_Level::Info, Log_Sender::EMA, fmt::format( "Started calculation of {} Eigenmodes ", n_modes ), idx_img,
         idx_chain );

    // Calculate the Eigenmodes
    vectorfield gradient( nos );

    // The gradient (unprojected)
    system->hamiltonian->Gradient( spins_initial, gradient );
    auto mask = system->geometry->mask_unpinned.data();
    auto g    = gradient.data();
    // Backend::par::apply(gradient.size(), [g, mask] SPIRIT_LAMBDA (int idx) {
    //     g[idx] = mask[idx]*g[idx];
    // });
    Vectormath::set_c_a( 1, gradient, gradient, system->geometry->mask_unpinned );

    VectorX eigenvalues;
    MatrixX eigenvectors;
    SpMatrixX tangent_basis = SpMatrixX( 3 * nos, 2 * nos );

    bool sparse = true;//system->ema_parameters->sparse;
    bool successful;
    if( sparse )
    {
        // The Hessian (unprojected)
        SpMatrixX hessian( 3 * nos, 3 * nos );
        system->hamiltonian->Sparse_Hessian( spins_initial, hessian );
        //std::cout <<hessian<<std::endl;
        // Get the eigenspectrum
        SpMatrixX hessian_constrained = SpMatrixX( 2 * nos, 2 * nos );

        successful = Eigenmodes::Sparse_Hessian_Partial_Spectrum(
            system->ema_parameters, spins_initial, gradient, hessian, n_modes, tangent_basis, hessian_constrained,
            eigenvalues, eigenvectors );
        //std::cout <<tangent_basis<<std::endl;
        //std::cout <<hessian_constrained<<std::endl;
    }
    else
    {
        // The Hessian (unprojected)
        MatrixX hessian( 3 * nos, 3 * nos );
        system->hamiltonian->Hessian( spins_initial, hessian );
        std::cout <<hessian<<std::endl;
        // Get the eigenspectrum
        MatrixX hessian_constrained = MatrixX::Zero( 2 * nos, 2 * nos );
        MatrixX _tangent_basis      = MatrixX( tangent_basis );

        successful = Eigenmodes::Hessian_Partial_Spectrum(
            system->ema_parameters, spins_initial, gradient, hessian, n_modes, _tangent_basis, hessian_constrained,
            eigenvalues, eigenvectors );
        //std::cout <<hessian_constrained<<std::endl;
        tangent_basis = _tangent_basis.sparseView();
        //std::cout <<tangent_basis<<std::endl;
    }

    

    if( successful )
    {
        // get every mode and save it to system->modes
        for( int i = 0; i < n_modes; i++ )
        {
            // Extract the minimum mode (transform evec_lowest_2N back to 3N)
            VectorX evec_3N = tangent_basis * eigenvectors.col( i );

            // dynamically allocate the system->modes
            system->modes[i] = std::shared_ptr<vectorfield>( new vectorfield( nos, Vector3{ 1, 0, 0 } ) );

            // Set the modes
            for( int j = 0; j < nos; j++ )
                ( *system->modes[i] )[j] = { evec_3N[3 * j], evec_3N[3 * j + 1], evec_3N[3 * j + 2] };

            // dynamically allocate the system->modes
            system->modes2N[i] = std::shared_ptr<vectorfield>( new vectorfield( nos, Vector3{ 1, 0,0} ) );
            // Set the modes
            for( int j = 0; j < nos; j++ )
                ( *system->modes2N[i] )[j] = { eigenvectors.col(i)[2 * j], eigenvectors.col(i)[2 * j + 1],0};

            // get the eigenvalues
            system->eigenvalues[i] = eigenvalues( i );
        }

        Log( Log_Level::Info, Log_Sender::All, fmt::format( "Finished calculation of {} Eigenmodes ", n_modes ),
             idx_img, idx_chain );

        int ev_print = std::min( n_modes, 100 );
        Log( Log_Level::Info, Log_Sender::EMA,
             fmt::format( "Eigenvalues: {}", eigenvalues.head( ev_print ).transpose() ), idx_img, idx_chain );
    }
    else
    {
        //// TODO: What to do then?
        Log( Log_Level::Warning, Log_Sender::All, "Something went wrong in eigenmode calculation...", idx_img,
             idx_chain );
    }
}

bool Hessian_Full_Spectrum(
    const std::shared_ptr<Data::Parameters_Method> parameters, const vectorfield & spins, const vectorfield & gradient,
    const MatrixX & hessian, MatrixX & tangent_basis, MatrixX & hessian_constrained, VectorX & eigenvalues,
    MatrixX & eigenvectors )
{
    std::size_t nos = spins.size();

    // Calculate the final Hessian to use for the minimum mode
    // TODO: add option to choose different Hessian calculation
    hessian_constrained = MatrixX::Zero( 2 * nos, 2 * nos );
    tangent_basis       = MatrixX::Zero( 3 * nos, 2 * nos );
    Manifoldmath::hessian_bordered( spins, gradient, hessian, tangent_basis, hessian_constrained );
    // Manifoldmath::hessian_projected(spins, gradient, hessian, tangent_basis, hessian_constrained);
    // Manifoldmath::hessian_weingarten(spins, gradient, hessian, tangent_basis, hessian_constrained);
    // Manifoldmath::hessian_spherical(spins, gradient, hessian, tangent_basis, hessian_constrained);
    // Manifoldmath::hessian_covariant(spins, gradient, hessian, tangent_basis, hessian_constrained);

    // Create and initialize a Eigen solver. Note: the hessian matrix should be symmetric!
    Eigen::SelfAdjointEigenSolver<MatrixX> hessian_spectrum( hessian_constrained );

    // Extract real eigenvalues
    eigenvalues = hessian_spectrum.eigenvalues().real();
    // Retrieve the real eigenvectors
    eigenvectors = hessian_spectrum.eigenvectors().real();

    // Return whether the calculation was successful
    return true;
}

bool Hessian_Partial_Spectrum(
    const std::shared_ptr<Data::Parameters_Method> parameters, const vectorfield & spins, const vectorfield & gradient,
    const MatrixX & hessian, std::size_t n_modes, MatrixX & tangent_basis, MatrixX & hessian_constrained,
    VectorX & eigenvalues, MatrixX & eigenvectors )
{
    std::size_t nos = spins.size();

    // Restrict number of calculated modes to [1,2N)
    n_modes = std::max( static_cast<std::size_t>( 1 ), std::min( 2 * nos - 2, n_modes ) );

    // If we have only one spin, we can only calculate the full spectrum
    if( n_modes == nos )
        return Hessian_Full_Spectrum(
            parameters, spins, gradient, hessian, tangent_basis, hessian_constrained, eigenvalues, eigenvectors );

    // Calculate the final Hessian to use for the minimum mode
    // TODO: add option to choose different Hessian calculation
    hessian_constrained = MatrixX::Zero( 2 * nos, 2 * nos );
    tangent_basis       = MatrixX::Zero( 3 * nos, 2 * nos );
    Manifoldmath::hessian_bordered( spins, gradient, hessian, tangent_basis, hessian_constrained );
// Manifoldmath::hessian_projected(spins, gradient, hessian, tangent_basis, hessian_constrained);
// Manifoldmath::hessian_weingarten(spins, gradient, hessian, tangent_basis, hessian_constrained);
// Manifoldmath::hessian_spherical(spins, gradient, hessian, tangent_basis, hessian_constrained);
// Manifoldmath::hessian_covariant(spins, gradient, hessian, tangent_basis, hessian_constrained);

// Remove degrees of freedom of pinned spins
#ifdef SPIRIT_ENABLE_PINNING
    for( std::size_t i = 0; i < nos; ++i )
    {
        // TODO: pinning is now in Data::Geometry
        // if (!parameters->pinning->mask_unpinned[i])
        // {
        //     // Remove interaction block
        //     for (int j=0; j<nos; ++j)
        //     {
        //         hessian_constrained.block<2,2>(2*i,2*j).setZero();
        //         hessian_constrained.block<2,2>(2*j,2*i).setZero();
        //     }
        //     // Set diagonal matrix entries of pinned spins to a large value
        //     hessian_constrained.block<2,2>(2*i,2*i).setZero();
        //     hessian_constrained.block<2,2>(2*i,2*i).diagonal().setConstant(nos*1e5);
        // }
    }
#endif // SPIRIT_ENABLE_PINNING

    // Create the Spectra Matrix product operation
    Spectra::DenseSymMatProd<scalar> op( hessian_constrained );
    // Create and initialize a Spectra solver
    Spectra::SymEigsSolver<scalar, Spectra::SMALLEST_ALGE, Spectra::DenseSymMatProd<scalar>> hessian_spectrum(
        &op, n_modes, 2 * nos );
    hessian_spectrum.init();

    // Compute the specified spectrum, sorted by smallest real eigenvalue
    int nconv = hessian_spectrum.compute( 1000, 1e-10, int( Spectra::SMALLEST_ALGE ) );

    // Extract real eigenvalues
    eigenvalues = hessian_spectrum.eigenvalues().real();

    // Retrieve the real eigenvectors
    eigenvectors = hessian_spectrum.eigenvectors().real();

    // Return whether the calculation was successful
    return ( hessian_spectrum.info() == Spectra::SUCCESSFUL ) && ( nconv > 0 );
}

bool Sparse_Hessian_Partial_Spectrum(
    const std::shared_ptr<Data::Parameters_Method> parameters, const vectorfield & spins, const vectorfield & gradient,
    const SpMatrixX & hessian, int n_modes, SpMatrixX & tangent_basis, SpMatrixX & hessian_constrained,
    VectorX & eigenvalues, MatrixX & eigenvectors )
{
    int nos = spins.size();

    // Restrict number of calculated modes to [1,2N)
    n_modes = std::max( 1, std::min( 2 * nos - 2, n_modes ) );

    // Calculate the final Hessian to use for the minimum mode
    Manifoldmath::sparse_tangent_basis_spherical( spins, tangent_basis );

    SpMatrixX hessian_constrained_3N = SpMatrixX( 3 * nos, 3 * nos );
    Manifoldmath::sparse_hessian_bordered_3N( spins, gradient, hessian, hessian_constrained_3N );

    hessian_constrained = tangent_basis.transpose() * hessian_constrained_3N * tangent_basis;

    // TODO: Pinning (see non-sparse function for)

    hessian_constrained.makeCompressed();
    int ncv = std::min(2*nos, std::max(2*n_modes + 1, 20)); // This is the default value used by scipy.sparse
    int max_iter = 100*nos;

    // Create the Spectra Matrix product operation
    Spectra::SparseSymMatProd<scalar> op( hessian_constrained );
    // Create and initialize a Spectra solver
    Spectra::SymEigsSolver<scalar, Spectra::SMALLEST_ALGE, Spectra::SparseSymMatProd<scalar>> hessian_spectrum(
        &op, n_modes, ncv);
    hessian_spectrum.init();

    // Compute the specified spectrum, sorted by smallest real eigenvalue
    int nconv = hessian_spectrum.compute( max_iter, 1e-10, int( Spectra::SMALLEST_ALGE ) );

    // Extract real eigenvalues
    eigenvalues = hessian_spectrum.eigenvalues().real();

    // Retrieve the real eigenvectors
    eigenvectors = hessian_spectrum.eigenvectors().real();

    // Return whether the calculation was successful
    return ( hessian_spectrum.info() == Spectra::SUCCESSFUL ) && ( nconv > 0 );
}

/*
 Calculate the lowest n_modes using Gradient Decent
*/
bool computeLowEV(const std::shared_ptr<Data::Parameters_Method> parameters, const vectorfield & spins, const vectorfield & gradient,
    const SpMatrixX & hessian, int n_modes, SpMatrixX & tangent_basis, SpMatrixX & hessian_constrained,
    VectorX & eigenvalues, MatrixX & eigenvectors, MatrixX & prev, int GDIterations)
{
    int nos = spins.size();

    // Restrict number of calculated modes to [1,2N)
    n_modes = std::max( 1, std::min( 2 * nos - 2, n_modes ) );

    // Calculate the final Hessian to use for the minimum mode
    Manifoldmath::sparse_tangent_basis_spherical( spins, tangent_basis );

    SpMatrixX hessian_constrained_3N = SpMatrixX( 3 * nos, 3 * nos );
    Manifoldmath::sparse_hessian_bordered_3N( spins, gradient, hessian, hessian_constrained_3N );

    hessian_constrained = tangent_basis.transpose() * hessian_constrained_3N * tangent_basis;

    hessian_constrained.makeCompressed();

    SpMatrixX matrix = hessian_constrained;

    int dimM = matrix.rows();
    double R;
    bool successful=1;
    int nit = 10000000;
    int l=0;
    double maxev;
    double step;
    

    VectorX Vec(dimM);
    VectorX Vec1(dimM);
    VectorX MatV(dimM);
    VectorX gradR;

    // get the biggest eigenvalue
    maxev=15;
    
    for(int i=0;i<=dimM;i++)
    {
        Vec(i)=rand()%100;
    }

    Vec.normalize();

    for(int s=0; s<500;s++)
    {
        MatV=matrix*Vec;

        R=Vec.transpose()*MatV;//*1/(Vec.norm()*Vec.norm())
        /*
        if(s%1000==0)
            {
                std::cout <<s<<": maxev="<<R<<"; gradient: "<<gradR.norm()<<std::endl;

            }
        */
        gradR=MatV-R*Vec;
            
        if(gradR.norm()<5e-7 || s==GDIterations/100)
        {
            maxev=R;
            //std::cout <<s<<": maxev="<<R<<std::endl;
            break;
        }
        if(10000<gradR.norm()&& 100<s)
        {
            std::cout <<"gradient diverged after "<<s<<" steps"<<std::endl;   
            successful=0;
            break;
        }

        if(true)//s<100)
        {
            step=0.01;
        }
        else
        {
            step=1.0/double(s);
        }

        Vec=Vec+step*gradR;
        Vec.normalize();
    }
    

    if(prev.size()==0)
    {
        for(int i=0;i<=dimM;i++)
        {
            Vec1(i)=rand()%100;
        }
    
    Vec1.normalize();
    }
    //std::cout <<"Starting Calculation of "<<n_modes<<" eigenvalues"<< std::endl;

    for(int n=1;n<=n_modes;n+=1)
    {	
        if(prev.size()==0)
    	{
       	    Vec=Vec1;
	    }
	    else
	    {
	        Vec=prev.col(n-1);
	    }

        Vec.normalize();

        for(int s=0; s<nit;s++)
        {
            for(int m=0;m<n;m+=1)
            {
                if(m==0)
                {
                    MatV=matrix*Vec;
                }
                else
                {
                    MatV+=(0.99*abs(maxev-eigenvalues(m-1)))*eigenvectors.col(m-1).dot(Vec)*eigenvectors.col(m-1);
                }
            }

            R=Vec.transpose()*MatV;//*1/(Vec.norm()*Vec.norm())

            gradR=MatV-R*Vec;

            //if(s%10000==0)
            //{
            //    std::cout <<"eigenvalue " << n<<" "<<R<<" "<<gradR.norm()<<std::endl;
            //}
            
            
            if(gradR.norm()<5e-7 || s==GDIterations)
            {
                eigenvalues.conservativeResize(eigenvalues.size()+1);
                eigenvalues(n-1)=R;
                eigenvectors.conservativeResize(dimM,eigenvectors.cols()+1);
                eigenvectors.col(n-1)=Vec;
                //std::cout <<"eigenvalue " << n<<" converged after "<< s <<" steps as R="<<R<<std::endl;
                break;
            }

            if(10000<gradR.norm())/* something to check if the gradient is diverging*/
            {   
                std::cout <<"The gradient does not converge: |gR|="<< gradR.norm()<< std::endl;
                successful=0;
                break;
	        }

            step=1.0/(100.0*ceil((s+1)/10000.0));

            //if(s%10000==0)
            //{
            //    std::cout <<step<<std::endl;
            //}
 
            Vec=Vec-step*gradR;

            Vec.normalize();
            
        }
    }

    // sort eigenvectors and eigenvalues (small->big)
    float curr, min;
    int minj;
    
    for (int i=0; i<n_modes-1;i++)
    {
        curr=eigenvalues(i);
        min=curr;
        minj=i;
        for (int j=i;j<n_modes;j++)
        {
            if(eigenvalues(j)<min)
            {
                min=eigenvalues(j);
                minj=j;
            }
        }
        if (min!=curr)
        {
            Vec=eigenvectors.col(i);
            eigenvectors.col(i)=eigenvectors.col(minj);
            eigenvalues(i)=eigenvalues(minj);
            eigenvectors.col(minj)=Vec;
            eigenvalues(minj)=curr;

        }
    }

    // Return whether the calculation was successful
    return ( successful );
}

void Transfer_Eigenmodes( std::shared_ptr<Data::Spin_System> system, int idx_img, int idx_chain )
{
    int nos = system->nos;
    MatrixX eigenvectors;
    SpMatrixX tangent_basis = SpMatrixX( 3 * nos, 2 * nos );

    int n_modes=(system->modes2N).size();
    vectorfield spins = *system->spins;

    // calculate tangent_basis
    Manifoldmath::sparse_tangent_basis_spherical( spins, tangent_basis );

    // write 2N modes into long vectors (like in Sparse Hessian Spectra Matrix)
    for(int i=0;i< n_modes;i++)
    {
        eigenvectors.conservativeResize(2*nos,eigenvectors.cols()+1);
        for( int j = 0; j < nos; j++ )
        {
            (eigenvectors.col(i))[2*j]=((*system->modes2N[i])[j])[0];
            (eigenvectors.col(i))[2*j+1]=((*system->modes2N[i] )[j])[1];
        }
    }   
    // get every mode and save it to system->modes
    for( int i = 0; i < n_modes; i++ )
    {
        // Extract the minimum mode (transform evec_lowest_2N back to 3N)
        VectorX evec_3N = tangent_basis * eigenvectors.col( i );
        // dynamically allocate the system->modes
        system->modes[i] = std::shared_ptr<vectorfield>( new vectorfield( nos, Vector3{ 1, 0, 0 } ) );
        // Set the modes
        for( int j = 0; j < nos; j++ )
            ( *system->modes[i] )[j] = { evec_3N[3 * j], evec_3N[3 * j + 1], evec_3N[3 * j + 2] };

    }
}

void Flip_Eigenmode(std::shared_ptr<Data::Spin_System> system,int idx_mode, int idx_img, int idx_chain)
{
    int nos = system->nos;

    for( int j = 0; j < nos; j++ )
    {
            ( *system->modes[idx_mode] )[j] = -( *system->modes[idx_mode] )[j] ;
    }
}

void Calculate_EigenmodesGD( std::shared_ptr<Data::Spin_System> system, int idx_img, int idx_chain, int GD_it)
{
    std::cout <<"nos: "<<system->nos<<std::endl;
    int nos = system->nos;

    std::cout <<"nos: "<<nos<<std::endl;

    //Check_Eigenmode_Parameters( system );
    std::cout <<"n_modes: "<<system->ema_parameters->n_modes<<std::endl;
    auto & n_modes = system->ema_parameters->n_modes;
    std::cout <<"n_modes: "<<n_modes<<std::endl;

    std::cout <<"spins: "<<(*system->spins)[0]<<std::endl;
    
    // vectorfield mode(nos, Vector3{1, 0, 0});
    vectorfield spins_initial = *system->spins;
    //std::cout <<"spins: "<<spins_initial<<std::endl;



    Log( Log_Level::Info, Log_Sender::EMA, fmt::format( "Started calculation of {} Eigenmodes ", n_modes ), idx_img,
         idx_chain );

    // Calculate the Eigenmodes
    vectorfield gradient( nos );
        // The gradient (unprojected)
    system->hamiltonian->Gradient( spins_initial, gradient );
    auto mask = system->geometry->mask_unpinned.data();
    auto g    = gradient.data();
    // Backend::par::apply(gradient.size(), [g, mask] SPIRIT_LAMBDA (int idx) {
    //     g[idx] = mask[idx]*g[idx];
    // });
     Vectormath::set_c_a( 1, gradient, gradient, system->geometry->mask_unpinned );

    VectorX eigenvalues;
    MatrixX eigenvectors;
    SpMatrixX tangent_basis = SpMatrixX( 3 * nos, 2 * nos );
    bool sparse = true;//system->ema_parameters->sparse;
    bool successful;
    // The Hessian (unprojected)
    SpMatrixX hessian( 3 * nos, 3 * nos );
    system->hamiltonian->Sparse_Hessian( spins_initial, hessian );
    // Get the eigenspectrum
    MatrixX prev_ev=MatrixX(2*nos,n_modes);
    for(int i=0;i< n_modes; i++ )
    {
        for( int j = 0; j < nos; j++ )
        {
            (prev_ev.col(i))[2*j]=((*system->modes2N[i])[j])[0];
            (prev_ev.col(i))[2*j+1]=((*system->modes2N[i] )[j])[1];
            //*image->modes2N[idx_mode] )[0].data()
            //std::cout <<(*system->modes2N[i] )[0].data()<< std::endl;
        }
    }
    SpMatrixX hessian_constrained = SpMatrixX( 2 * nos, 2 * nos );
    successful = Eigenmodes::computeLowEV(system->ema_parameters, spins_initial, gradient, hessian, n_modes, tangent_basis, hessian_constrained,eigenvalues, eigenvectors, prev_ev, GD_it);
    if( successful )
    {
        // get every mode and save it to system->modes
        for( int i = 0; i < n_modes; i++ )
        {
            // Extract the minimum mode (transform evec_lowest_2N back to 3N)
            VectorX evec_3N = tangent_basis * eigenvectors.col( i );

            // dynamically allocate the system->modes
            system->modes[i] = std::shared_ptr<vectorfield>( new vectorfield( nos, Vector3{ 1, 0, 0 } ) );

            // Set the modes
            for( int j = 0; j < nos; j++ )
                ( *system->modes[i] )[j] = { evec_3N[3 * j], evec_3N[3 * j + 1], evec_3N[3 * j + 2] };

            // dynamically allocate the system->modes
            system->modes2N[i] = std::shared_ptr<vectorfield>( new vectorfield( nos, Vector3{ 1, 0,0} ) );
            // Set the modes
            for( int j = 0; j < nos; j++ )
                ( *system->modes2N[i] )[j] = { eigenvectors.col(i)[2 * j], eigenvectors.col(i)[2 * j + 1],0};

            // get the eigenvalues
            system->eigenvalues[i] = eigenvalues( i );
        }

        Log( Log_Level::Info, Log_Sender::All, fmt::format( "Finished calculation of {} Eigenmodes ", n_modes ),
             idx_img, idx_chain );

        int ev_print = std::min( n_modes, 100 );
        Log( Log_Level::Info, Log_Sender::EMA,
             fmt::format( "Eigenvalues: {}", eigenvalues.head( ev_print ).transpose() ), idx_img, idx_chain );
    }
    else
    {
        //// TODO: What to do then?
        Log( Log_Level::Warning, Log_Sender::All, "Something went wrong in eigenmode calculation...", idx_img,
             idx_chain );
    }
}

} // namespace Eigenmodes
} // namespace Engine
