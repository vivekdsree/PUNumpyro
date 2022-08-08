from jax import  numpy as jnp
import numpy as np

def load_file(path, speciesname, index, scaling_factor):
    """
    Load the data files corresponding to a specific species sort them and 
    return scaled version back to
    Inputs :  
        - path : 'dir' of the data folder w.r.t to cwd 
        - speciesname : Name of cell or chemokine type
        - indexlist : List of file indices to be used
        - scaling_factor : Scaling factor for each species
    Outputs : 
        - dat_species : Array of the scaled and sorted loaded data
    """
    dat_species = []
    for i in index:
        dat_species.extend(np.loadtxt(f"{path}/{speciesname}/{i}.dat"))

    dat_species = np.asarray(dat_species)
    dat_species = dat_species[jnp.argsort(dat_species[:, 0])]
    dat_species[:, 2] = dat_species[:, 2]*scaling_factor
    return jnp.array(dat_species)


def add_end_data(dat_species, index, end_time, homeostatic_value):
    """
    Assume Homeostasis by day 20 of the wound formation, add data at days 20 
    and 25 to drive all values down towards normal. Sample from a gaussian 
    near 1 scale and add to data
    Inputs :  
        - dat_species : Array of data for a species
        - index : List of file indices to be used
        - end_time : The day at which data is to be added
    Outputs : 
        - dat_species : Array of data for a species with added homeoestatic 
            value at end_time
    """
    # FIXME!!
    np.random.seed(10)
    ##

    length = len(index)
    vals = np.random.randn(length)*0.01+homeostatic_value
    days = np.ones(length)*end_time
    valarr = np.column_stack([days, days, vals])
    dat_species = jnp.append(dat_species, valarr, 0)
    return dat_species


if __name__ == '__main__':
    # Testing if the data load routines works
    '''File indices for each species data to be used'''
    rhokl = [1]
    macl = [1]
    neutl = [2]
    kcl = [1]
    tal = [2]
    tbl = [1]
    path = "../data"

    dat_mac = load_file(path, "Macro", macl, 30)
    dat_mac = add_end_data(dat_mac, macl, 20, 1.0)
    print(dat_mac.shape)
