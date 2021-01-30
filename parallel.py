#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 11:07:09 2018
TODO:
    - dump parameter space to file
    - check if filter/adapeter can be used for configure loggers Formatter
    (There is need for add nsim, process id can be added somwhere directily)
@author: ajkieloaho
"""

import os
import sys
import multiprocessing as mp
from threading import Thread
from multiprocessing import Process, Queue, Pool  # , cpu_count
#from psutil import cpu_count
from copy import deepcopy

from pandas import date_range

from tools.iotools import initialize_netcdf, write_ncf
from tools.iotools import jsonify
from pyAPES import Model

import time

import logging
import logging.handlers
import logging.config


def _result_writer(ncf):
    """
    Args:
        ncf: NetCDF4 file handle
    """
    
    logger = logging.getLogger()
    logger.info("Writer is ready!")
    
    while True:
        # results is tuple (Nsim, data)
        results = writing_queue.get()
        
        if results is None:
            ncf.close()
            logger.info("NetCDF4 file is closed. and Writer closes.")
            break

        #logger.info("Writing results of simulation {}".format(results[0]))
        write_ncf(nsim=results[0], results=results[1], ncf=ncf)
        logger.info("Results of simulation {} has been written".format(results[0]))
# logging to a single file from multiple processes
# https://docs.python.org/dev/howto/logging-cookbook.html#logging-to-a-single-file-from-multiple-processes


def _logger_listener():
    """
    """

    while True:
        record = logging_queue.get()

        if record is None:
            # print('logger done')
            break

        logger = logging.getLogger(record.name)
        logger.handle(record)


def _worker():
    """
    """

#    from pyAPES_utilities.spinup import soil_spinup

    # --- LOGGING ---
    qh = logging.handlers.QueueHandler(logging_queue)
    root = logging.getLogger()

    # !!! root level set should be in configuration dictionary!!!
    root.handlers = []
    root.setLevel(logging.INFO)
    root.addHandler(qh)

    # --- TASK QUEUE LISTENER ---
    while True:
        task = task_queue.get()

        if task is None:
            root.info('Worker done')
            break

        root.info("Creating simulation {}".format(task['nsim']))

        try:
#            soil_temperature = soil_spinup(
#                task['general'],
#                task['canopy'],
#                task['soil'],
#                task['forcing'],
#                moss_type,
#                0.01
#            )
#
#            task['soil']['heat_model']['initial_condition']['temperature'] = soil_temperature

            model = Model(
                dt=task['general']['dt'],
                canopy_para=task['canopy'],
                soil_para=task['soil'],
                forcing=task['forcing'],
                outputs=output_variables['variables'],
                nsim=task['nsim'],
            )

            result = model.run()
            root.info("Putting results of simulation {} to writing queue.".format(task['nsim']))
            writing_queue.put((task['nsim'], result))

        except:
            message = 'FAILED: simulation {}'.format(task['nsim'])
            root.info(message + '_' + sys.exc_info()[0])
        # can return something if everything went right


def driver(ncf_params,
           logging_configuration,
           N_workers):
    """
    Args:
        ncf_params (dict): netCDF4 parameters
        logging_configuration (dict): parallel logging configuration
        N_workers (int): number of worker processes
    """

    # --- PROCESSES ---
    running_time = time.time()

    workers = []
    for k in range(N_workers):
        workers.append(
            Process(
                target=_worker,
            )
        )

        task_queue.put(None)
        workers[k].start()

    # --- NETCDF4 ---
    ncf, _ = initialize_netcdf(
        variables=ncf_params['variables'],
        sim=ncf_params['Nsim'],
        soil_nodes=ncf_params['Nsoil_nodes'],
        canopy_nodes=ncf_params['Ncanopy_nodes'],
        planttypes=ncf_params['Nplant_types'],
        groundtypes=ncf_params['Nground_types'],
        time_index=ncf_params['time_index'],
        filepath=ncf_params['filepath'],
        filename=ncf_params['filename'])

    writing_thread = Thread(
        target=_result_writer,
        args=(ncf,)
    )

    writing_thread.start()

    # --- LOGGING ---
    logging.config.dictConfig(logging_configuration)

    logging_thread = Thread(
        target=_logger_listener,
    )

    logging_thread.start()

    # --- USER INFO ---

    logger = logging.getLogger()
    logger.info('Number of worker processes is {}, number of simulations: {}'.format(N_workers, Nsim))

    # --- CLOSE ---

    # join worker processes
    for w in workers:
        w.join()

    logger.info('Worker processes have joined.')
    logger.info('Running time %.2f seconds' % (time.time() - running_time))

    # end logging queue and join
    logging_queue.put_nowait(None)
    logging_thread.join()

    # end writing queue and join
    writing_queue.put_nowait(None)
    writing_thread.join()

    logger.info('Results are in path: ' + ncf_params['filepath'])

    return ncf_params['filepath']


def get_tasks(scenario,
              trajectories=None,
              optimal_trajectories=None,
              num_levels=None,
              save_samples=False,
             ):
    """ Creates parameters space for tasks
    
    Args:
        scenario (str): name of parameter scenario
        optimal_trajectories (int): number of optimal trajectories,
        num_levels (int): number of parameter levels in sampling
        save_samples (bool/str): save samples and parameter ranges
    """

    from sensitivity_tools import sensitivity_sampling
    from parameters.parametersets import get_parameters
    from parameters.sensitivity_ranges import get_problem_definition
    from parameters.parameter_tools import get_parameter_list
    from copy import deepcopy as copy
        
    if trajectories and optimal_trajectories and num_levels:
        problem_definition = get_problem_definition(scenario)
        default_parameters = problem_definition['default_parameters']
        parameter_ranges = problem_definition['ranges']
    
        start = default_parameters['general']['start_time'].split('-')
        start = ''.join(start)
        end = default_parameters['general']['end_time'].split('-')
        end = ''.join(end)
        dt = default_parameters['general']['dt']
        
        filename = scenario + '_' + start + '-' + end + '_trajectories_{}'.format(optimal_trajectories) + '_levels_{}'.format(num_levels)
        
        
        
        parameter_list = sensitivity_sampling(
            parameters=default_parameters,
            moss_ranges=parameter_ranges,
            trajectories=trajectories,
            optimal_trajectories=optimal_trajectories,
            num_levels=num_levels,
            save_samples=filename
        )
                
    else:
        parameter_list = get_parameter_list(scenario)
        
        start = parameter_list[0]['general']['start_time'].split('-')
        start = ''.join(start)
        end = parameter_list[0]['general']['end_time'].split('-')
        end = ''.join(start)
        dt = parameter_list[0]['general']['dt']
        
        filename = scenario + '_' + start + '-' + end
    
    
    
    filename = time.strftime('%Y%m%d_') + filename + '_pyAPES_results.nc'

    pyAPES_folder = os.getcwd()
    filepath = os.path.join(pyAPES_folder, results_directory, filename)

    freq = '{}S'.format(dt)
    
    #time_index = date_range(start, end, freq=freq, closed='left')
    
    # needed netCDF4 parameters are taken from parameters of first simulation
    
    
    time_index = parameter_list[0]['forcing'].index
    
    # ground layer types in parameters (only those which coverage is more than 0 m2)
    bl_para = parameter_list[0]['canopy']['forestfloor']['bottom_layer_types']
    bl_names = list(bl_para.keys())
    
    bl_types = []
    
    for blt in bl_names:
        if bl_para[blt]['coverage'] > 0:
            bl_types.append(blt)
    
    
    ncf = {
        'variables': output_variables['variables'],
        'Nsim': len(parameter_list),
        'Nsoil_nodes': len(parameter_list[0]['soil']['grid']['dz']),
        'Ncanopy_nodes': parameter_list[0]['canopy']['grid']['Nlayers'],
        'Nplant_types': len(parameter_list[0]['canopy']['planttypes']),
        'Nground_types': len(bl_types),
        'time_index': time_index,
        'filename': filename,
        'filepath': filepath,
    }

    return parameter_list, ncf


if __name__ == '__main__':
    import argparse
    from parameters.outputs import parallel_logging_configuration, output_variables, results_directory
    #mp.set_start_method('spawn')

    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', help='number of cpus to be used', type=int)
    parser.add_argument('--scenario', help='scenario name (Degero)', type=str)
    parser.add_argument('--opt_traj', help='number of optimal trajectories to be used', type=int)
    parser.add_argument('--traj', help='number of trajectrories (number of levels + 1)', type=int)
    parser.add_argument('--levels', help='number of levels to be used', type=int)
    args = parser.parse_args()

    # --- Queues ---
    manager = mp.Manager()
    logging_queue = Queue()
    writing_queue = Queue()
    task_queue = Queue()

    # --- TASKS ---
    scenario = args.scenario
    optimal_trajectories = args.opt_traj
    trajectories = args.traj
    num_levels = args.levels

    if optimal_trajectories and trajectories and num_levels: 
        tasks, ncf_params = get_tasks(
            scenario=scenario,
            trajectories=trajectories,
            optimal_trajectories=optimal_trajectories,
            num_levels=num_levels,
            save_samples=scenario
        )
        
        logfile_name = time.strftime('%Y%m%d_') + 'Morris_'+ scenario + '_' + 'levels_{}'.format(num_levels) + '_trajectories_{}'.format(optimal_trajectories) + '.log'
        
    else:
        tasks, ncf_params = get_tasks(
            scenario=scenario,
            trajectories=None,
            optimal_trajectories=None,
            num_levels=None,
            save_samples=scenario
        )
        
        logfile_name = time.strftime('%Y%m%d_') + 'OAT_'+ scenario + '_' + '.log'
        

    Nsim = len(tasks)

    for para in tasks:
        task_queue.put(deepcopy(para))

    # --- Number of workers ---
    Ncpu = args.cpu


    if Ncpu is None:
        #Ncpu = cpu_count(logical=False)
        Ncpu = 1

#   if Nsim > (Ncpu - 1):
#       N_workers = Ncpu - 1
#   else:
    N_workers = Ncpu - 1

    parallel_logging_configuration['handlers']['parallelAPES_file']['filename'] = logfile_name

    # --- DRIVER CALL ---
    outputfile = driver(
        ncf_params=ncf_params,
        logging_configuration=parallel_logging_configuration,
#        task_queue=task_queue,
#        logging_queue=logging_queue,
#        writing_queue=writing_queue,
        N_workers=N_workers)

    print(outputfile)