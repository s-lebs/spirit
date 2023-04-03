#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 10:07:54 2022

@author: lebs
"""

import os
import sys
import time
import numpy as np
#spirit_py_dir = os.path.abspath("/core/python")
#sys.path.insert(0, spirit_py_dir)
#print(sys.path)

from spirit_extras import import_spirit, memory_monitor
spirit_info = import_spirit.find_and_insert("~/Documents/Skyrmionen/spirit-var", stop_on_first_viable=True )[0]
print(spirit_info)

### Import Spirit modules
from spirit import state, configuration, simulation, geometry, io,hamiltonian, system,log
from spirit.parameters import ema




cfgfile = "input/input.cfg"
quiet = False    
#mem = memory_monitor.MemoryMonitor(dt=0.05)
#mem.start()

# dim=7
# J1=10#
# J2=-3.3333
# Ki=0.0001
# n=100

# dim=50
# J1=10
# J2=-3.3333
# Ki=0.0115
# n=100
# n2=10000

def ZeroModeVolume(dim=50,J1=10,J2=-3.3333,Ki=0.0115,amplitude=1):

    with state.State(cfgfile, quiet) as p_state:
        geometry.set_n_cells(p_state, [dim,dim,1])
        configuration.plus_z(p_state, idx_image=0)
        hamiltonian.set_anisotropy(p_state, Ki, [0,0,1], idx_image=0, idx_chain=-1)
        hamiltonian.set_exchange(p_state, 2, [J1,J2], idx_image=-1, idx_chain=-1)
        
        #io.image_read(p_state,"SingleSpin.ovf")
        io.image_read(p_state,"skyrmion_spcorrect333.ovf")
        
        #
        Spin=system.get_spin_directions(p_state)
        # Spin[0][0]=0
        # Spin[0][1]=1
        # Spin[0][2]=0
        SpinsO=Spin.copy()
        # print(SpinsO)
        norm=0
        prevNorm=-1
        spNorm=-2
        i=0
        Volume=0
        dVol=0
        SpinsPrev=SpinsO
        
        #io.image_write(p_state,"spin_rot_alt-2-0.ovf")
        
        ema.set_amplitude(p_state,amplitude)
        
        #start rotation by computing the eigenmodes once by hand, otherwise the last from mmf will be used:
        system.update_eigenmodes(p_state, idx_image=-1, idx_chain=-1)  
        
        while True:
            

            
            
            
            #compute eigenmodes and apply ZeroMode
            system.transfer_eigenmodes(p_state, idx_image=-1, idx_chain=-1)
            
            #ev1=system.get_eigenmode(p_state, 1, idx_image=-1, idx_chain=-1)
            
            #system.update_eigenmodes(p_state, idx_image=-1, idx_chain=-1) 
            
            #ev2=system.get_eigenmode(p_state, 1, idx_image=-1, idx_chain=-1)
            
            #print(ev1,ev2)
            
            #io.eigenmodes_write(p_state, "eigenvectors333-"+str(nbMMF)+"-"+str(nGD)+".ovf")
            eigenvalues=system.get_eigenvalues(p_state, idx_image=-1, idx_chain=-1)
            idx=0
            for s in eigenvalues:
                if abs(s)<=1e-6:
                    break
                idx=idx+1
            #print(idx)
            configuration.displace_eigenmode(p_state, idx, idx_image=-1,idx_chain=-1)
            
            
            
            #get back to the saddlepoint
            #start=time.time()
            #info=simulation.start(p_state, simulation.METHOD_MMF, simulation.SOLVER_VP, n_iterations=n2)
            #print(info.total_iterations,time.time()-start)
            
            #neccessary to compute the ZMV
            
            Spins=system.get_spin_directions(p_state) 
            dVol=np.linalg.norm(SpinsPrev-Spins)
            #print(SpinsO-Spins)
            
            SpinsPrev=Spins.copy()
            
           
            
            #neccessary for stopping the program
            
            spNorm=prevNorm
            prevNorm=norm
            norm=np.linalg.norm(SpinsO-Spins)
            #print([i,spNorm,prevNorm,norm],file=open("Volume_results_Spin.txt","a"))
            
            #io.image_write(p_state,"spin_rot_alt-2-"+str(i+1)+".ovf")
            
            Volume=Volume+dVol #add previous norm
            
            
            if spNorm > prevNorm and prevNorm < norm:
                break
            
            
                
                
            i=i+1
        Volume=Volume-np.linalg.norm(SpinsO-Spins)    
        
        print(amplitude,i,Volume,file=open("Volume_results_Spin.txt","a"))
        print(amplitude,i,Volume)

ZeroModeVolume(amplitude=1)
