"""
Eigenmode analysis (EMA)
-------------------------------------------------------------

This method, if needed, calculates modes (they can also be read in from a file)
and perturbs the spin system periodically in the direction of the eigenmode.
"""

import spirit.spiritlib as spiritlib
import ctypes

# Load Library
_spirit = spiritlib.load_spirit_library()

_EMA_Clear_Modes = _spirit.Parameters_EMA_Clear_Modes
_EMA_Clear_Modes.argtypes = [ctypes.c_void_p,
                             ctypes.c_int, ctypes.c_int]
_EMA_Clear_Modes.restype = None
def clear_modes(p_state, idx_image=-1, idx_chain=-1):
    """Clears the modes."""
    _EMA_Clear_Modes(ctypes.c_void_p(p_state),
                     ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

# ---------------------------------- Set ----------------------------------

_EMA_Set_N_Modes          = _spirit.Parameters_EMA_Set_N_Modes
_EMA_Set_N_Modes.argtypes = [ctypes.c_void_p, ctypes.c_int,
                            ctypes.c_int, ctypes.c_int]
_EMA_Set_N_Modes.restype  = None
def set_n_modes(p_state, n_modes, idx_image=-1, idx_chain=-1):
    """Set the number of modes to calculate or use."""
    _EMA_Set_N_Modes(ctypes.c_void_p(p_state), ctypes.c_int(n_modes),
                          ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

_EMA_Set_N_Mode_Follow          = _spirit.Parameters_EMA_Set_N_Mode_Follow
_EMA_Set_N_Mode_Follow.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                    ctypes.c_int, ctypes.c_int]
_EMA_Set_N_Mode_Follow.restype  = None
def set_n_mode_follow(p_state, n_mode, idx_image=-1, idx_chain=-1):
    """Set the index of the mode to use."""
    _EMA_Set_N_Mode_Follow(ctypes.c_void_p(p_state), ctypes.c_int(n_mode),
                          ctypes.c_int(idx_image), ctypes.c_int(idx_chain)) 

_EMA_Set_Sparse          = _spirit.Parameters_EMA_Set_Sparse
_EMA_Set_Sparse.argtypes = [ctypes.c_void_p, ctypes.c_bool,
                            ctypes.c_int, ctypes.c_int]
_EMA_Set_Sparse.restype  = None
def set_sparse(p_state, sparse, idx_image=-1, idx_chain=-1):
    """Set wether to use sparse matrices."""
    _EMA_Set_Sparse(ctypes.c_void_p(p_state), ctypes.c_bool(sparse),
                    ctypes.c_int(idx_image), ctypes.c_int(idx_chain))

_EMA_Set_Amplitude          = _spirit.Parameters_EMA_Set_Amplitude
_EMA_Set_Amplitude.argtypes = [ctypes.c_void_p, ctypes.c_float,
                                    ctypes.c_int, ctypes.c_int]
_EMA_Set_Amplitude.restype  = None
def set_amplitude(p_state, amplitude, idx_image=-1, idx_chain=-1):
    """Set the amplitude used to displace the eigemode."""
    _EMA_Set_Amplitude(ctypes.c_void_p(p_state), ctypes.c_float(amplitude),
                          ctypes.c_int(idx_image), ctypes.c_int(idx_chain))


# ---------------------------------- Get ----------------------------------

_EMA_Get_N_Modes          = _spirit.Parameters_EMA_Get_N_Modes
_EMA_Get_N_Modes.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_EMA_Get_N_Modes.restype  = ctypes.c_int
def get_n_modes(p_state, idx_image=-1, idx_chain=-1):
    """Returns the number of modes to calculate or use."""
    return int(_EMA_Get_N_Modes(p_state, ctypes.c_int(idx_image), ctypes.c_int(idx_chain)))

_EMA_Get_N_Mode_Follow          = _spirit.Parameters_EMA_Get_N_Mode_Follow
_EMA_Get_N_Mode_Follow.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_EMA_Get_N_Mode_Follow.restype  = ctypes.c_int
def get_n_mode_follow(p_state, idx_image=-1, idx_chain=-1):
    """Returns the index of the mode to use."""
    return int(_EMA_Get_N_Mode_Follow(p_state, ctypes.c_int(idx_image), ctypes.c_int(idx_chain)))


_EMA_Get_Sparse          = _spirit.Parameters_EMA_Get_Sparse
_EMA_Get_Sparse.argtypes = [ctypes.c_void_p, ctypes.c_bool,
                            ctypes.c_int, ctypes.c_int]
_EMA_Get_Sparse.restype  = ctypes.c_bool
def get_sparse(p_state, sparse, idx_image=-1, idx_chain=-1):
    """Set wether to use sparse matrices."""
    return bool(_EMA_Get_Sparse(ctypes.c_void_p(p_state),
                                ctypes.c_int(idx_image), ctypes.c_int(idx_chain)))

