#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys as _sys
import os as _os
_sys.path.insert(1, '..')

import libs
from interface.abm_interface import ParasTact
import numpy as _np
from sweep_paras_tactical_shocks import lifetime_func, Nm_shock_func
from abm_strategic.interface_distance import produce_M1_trajs_from_data as _produce_M1_trajs_from_data

main_dir = _os.path.abspath(__file__)
main_dir = _os.path.split(_os.path.dirname(main_dir))[0]

_result_dir = libs.paths.result_dir

zone = 'LIRR'

# Parameters directly controlled by tactical ABM
paras = ParasTact({})
paras['nsim'] = 10
paras['tmp_from_file'] = 1
paras['DAY'] = 86400.
paras['shock_f_lvl_min'] = 240
paras['shock_f_lvl_max'] = 340
paras['t_i'] = 8
paras['t_r'] = 0.5

paras['time_shock'] = 60.*2. # in minutes
paras['f_shocks'] = 0. # Number of shocks per day.
paras['f_shocks_iter'] = [0., 10., 20., 40., 80., 100., 150., 200., 250., 300., 400.]

paras['sig_V'] = 0.
paras['sig_V_iter'] = _np.arange(0., 0.26, 0.04)
paras['t_w'] = 150 
paras['t_w_iter'] = [150]

paras['paras_to_loop'] = ['f_shocks', 't_w']
assert len(paras['paras_to_loop'])==2

# Relationships between variables
paras.to_update['lifetime'] = (lifetime_func, ('time_shock', 't_w', 't_r', 't_i'))
paras.to_update['Nm_shock'] = (Nm_shock_func, ('f_shocks', 't_w', 't_r', 't_i', 'DAY', 'shock_f_lvl_min', 'shock_f_lvl_max'))

paras.update_priority = ['lifetime', 'Nm_shock']

paras.analyse_dependance()
#paras.initialize_paras()

#input_file=main_dir + '/trajectories/M1/trajs_' + zone + '_real_data.dat'
_n_strat = 10
input_files = []
for _i in range(_n_strat):
	#input_files.append(_os.path.join([_result_dir, 'trajectories', 'M1', 'trajs_Real_' + zone[:2] + '_v5.8_Strong_EXT' + zone + '_' + zone + '_2010-5-6+0_d2_cut240.0_directed_' + str(_i) +'.dat']))
	input_files.append(_result_dir + '/trajectories/M1/trajs_Real_' + zone[:2] + '_v5.8_Strong_EXT' + zone + '_' + zone + '_2010-5-6+0_d2_cut240.0_directed_' + str(_i) +'.dat')

# Used to copy all config files and avoid interactions between simultaneous codes.
temp_config_dir = _os.path.dirname(main_dir) + '/config_' + _os.path.split(__file__)[-1].split('.')[0]

# Paths from which these files are extracted
config_dir = main_dir + '/abm_tactical/config'
shock_file_zone = config_dir + '/shock_tmp_' + zone + '.dat'
bound_file_zone = config_dir + '/bound_latlon_' + zone + '.dat'
tmp_navpoints_file_zone = config_dir + '/temp_nvp_' + zone + '.dat'
config_file = config_dir + '/config.cfg'

paras.initialize_paras(config_file=config_file)

# Other parameters
force = True
starting_date=[2010, 5, 6, 0, 0, 0]
#n_iter=10

args = [zone, paras, input_files, config_file, shock_file_zone, bound_file_zone, tmp_navpoints_file_zone]

all_paras = {k:v for k,v in vars().items() if v not in args and k[:1]!='_' and k[-4:]!='func' and not k in ['ParasTact', 'main_dir'] and not k in [key for key in locals().keys()
       if isinstance(locals()[key], type(_sys)) and not key.startswith('__')]}

