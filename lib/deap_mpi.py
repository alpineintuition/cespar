__author__ = "Florin Dzeladini"
__copyright__ = "Copyright 2023, Alpine Intuition SARL"
__license__ = "Apache-2.0 license"
__version__ = "1.0.0"
__email__ = "berat.denizdurduran@alpineintuition.ch"
__status__ = "Stable"

import sys
from deap import creator, base, tools, algorithms ,cma
from mpi4py import MPI
import time
import pickle
import numpy as np
comm = MPI.COMM_WORLD
NP = comm.Get_size() # new: gives number of ranks in comm
DEBUG = None

def getFromCheckpoint(x,id=None):
    file=open('checkpoints/checkpoint_{}.pkl'.format(x),'rb'); 
    if id is None:
        return pickle.load(file)['best_ind']
    else:
        return pickle.load(file)['best_ind'][id]

PAR_SPACE = (
    [0.0, -1.0, 0.0, 0.0, \
    -2.0, -90/15, -1.0, 0.0, \
    0.0, 0.0, 0.0, 0.0, 0.0, \
    0.0, 0.0, 0.0, 0.0, 0.0, \
    0.0, 0.0, \
    0.0, 0.0, \
    0.0, 0.0, 0.0, \
    0.0, 0.0, 0.0, 0.0, 0.0, \
    0.0, 0.0, 0.0, \
    0.0, \
    0.0, \
    0.0, 0.0, \
    0.0, 0.0, 0.0, \
    0.0, 0.0, 0.0, \
    0.0, 0.0, 0.0],
    [6.0, 3.0, 5.0, 3.0, \
    3.0, 20/15, 15/10, 3.0, \
    3.0, 3.0, 3.0, 3.0, 3.0, \
    3.0, 3.0, 3.0, 3.0, 3.0, \
    3.0, 3.0, \
    3.0, 3.0, \
    3.0, 3.0, 3.0, \
    3.0, 3.0, 3.0, 3.0, 3.0, \
    3.0, 3.0, 3.0, \
    3.0, \
    3.0, \
    3.0, 3.0, \
    2.0, 3.0, 3.0, \
    3.0, 3.0, 3.0, \
    3.0, 3.0, 3.0])


def init(debug=None,par_space=None):
    global DEBUG,NP,PAR_SPACE,FF_PAR_SPACE
    if(debug is not None):
        DEBUG = debug
    if(par_space is not None):
        PAR_SPACE = par_space


scale   = lambda x,a,b: a + (b-a)*(x/10)
unscale = lambda y,a,b: 10*(y-a)/(b-a)

scale_fb = lambda x: scale(np.array(x),np.array(PAR_SPACE[0]),np.array(PAR_SPACE[1]))
unscale_fb = lambda y: unscale(np.array(y),np.array(PAR_SPACE[0]),np.array(PAR_SPACE[1]))

get_test_fitness = lambda x : np.abs(np.array(x)).sum()

#      _ _           _             _               
#   __| (_)___ _ __ | |_ __ _  ___| |__   ___ _ __ 
#  / _` | / __| '_ \| __/ _` |/ __| '_ \ / _ \ '__|
# | (_| | \__ \ |_) | || (_| | (__| | | |  __/ |   
#  \__,_|_|___/ .__/ \__\__,_|\___|_| |_|\___|_|   
#             |_|                                  
available       = [True]*NP # Available workers
offsprings      = [None]*NP
fitnesses       = [None]*NP # Received fitnesses from worker
sim_durations   = [None]*NP
real_durations  = [None]*NP
distances       = [None]*NP
evaluated       = 0
def dispatcher(N, generator, best = None, beta = 0): # beta 0: fb, 1: ff
    global DEBUG, available, offsprings, fitnesses, sim_durations, real_durations, evaluated
    if(DEBUG is None):
        print("You should run deap_mpi.init(Boolean)")
        sys.exit(1)
    NP = len(offsprings)
    #######################################################################
    # We generate the population to be evaluated
    _offsprings = []
    while len(_offsprings) < NP:
        _offsprings += generator()
    #######################################################################
    # Disatch messages and wait for N replies (i.e. enough to generate a new population)
    # print("We start a dispatching and have {} already evaluated from last generation.".format(evaluated))
    # Scattering
    for i in range(1,NP):
        # 
        # If worker is available for work 
        #     We add the offsprings to the global array indexed by worker
        #---------------------------------------------------------
        #     This means that the worker has send back a fitness. 
        #     But this should also mean that this worker has his fitness beeing used. 
        #     So we should not set the worker availability to true when we receive from the worker.
        #     But when the actual individual from that worker has finished.
        if(available[i]): 
            if(len(offsprings) is not 0):
                offsprings[i] = _offsprings.pop()
                individual = [j for j in offsprings[i]]
                individual = scale_fb(np.array(individual))

                if DEBUG: print("[MASTER] sending invidiual to {}: ind={}".format(i,individual))
                if DEBUG: sys.stdout.flush()
                comm.send([individual, best, beta], dest=i, tag=i)
                available[i] = False
    # Gathering
    i = 0
    while(True):
        i = (i+1)%NP
        # If we have evaluated enough individual (i.e. got fitnesses from them) we quit the loop.
        #-------------------#
        # commenting the lines below will allow the dispatcher to send more work than 
        if(evaluated >= N):   
            break           

        # If worker sends back data (fitnesses)
        #     We add the fitness information to the global arrays indexed by worker
        # FB Optimization tag at i+NP FF Optimization tag at i+2*NP
        if(comm.Iprobe(source=i, tag=i+NP*(beta+1))):        
            data = comm.recv(source=i, tag=i+NP*(beta+1))
            fitnesses[i]      = data[0]
            sim_durations[i]  = data[1]
            real_durations[i] = data[2]
            distances[i]      = data[3]
            evaluated        += 1
            if DEBUG: print("[MASTER] {}/{} Evaluation done from worker {}".format(evaluated,N,i))
            if DEBUG: sys.stdout.flush()
        time.sleep(0.001)
    #######################################################################
    # We have enough individual to generate offsprings.
    _offsprings     = []
    _fitnesses      = []
    _real_durations = []
    _sim_durations  = []
    _distances      = []
    idx     = 0
    counter = 0
    # Loop over all offsprings array by worker.
    while (counter < N):
        fit = fitnesses[idx] 
        ind = offsprings[idx]
        rd  = real_durations[idx]
        sd  = sim_durations[idx]
        ds  = distances[idx]
        # If worker has finished working we add it to the offsprings to be used by the master process.
        if(fit and ind):
            available[idx]       = True  # We release the worker for work
            ############################## RESETS WORKER ARRAYS
            fitnesses[idx]       = None
            offsprings[idx]      = None
            real_durations[idx]  = None
            sim_durations[idx]   = None
            distances[idx]       = None
            ############################## ADD DATA TO OFFSPRING
            ind.fitness.values = (fit, sd, 0.0)
            _offsprings.append(ind)
            _fitnesses.append(fit)
            _real_durations.append(np.round(rd,3))
            _sim_durations.append(np.round(sd,3))
            _distances.append(ds)
            ####################################### INCREASE CTR
            evaluated -= 1
            counter+=1
        # WORKER ARRAYS INDEX

        idx+=1
    return _offsprings, _fitnesses, _sim_durations, _real_durations, _distances

def getGenerators(toolbox):
    return {
        'GA'    : lambda pop : algorithms.varAnd(pop, toolbox['GA'], cxpb=toolbox['GA'].cxpb(), mutpb=toolbox['GA'].mutpb()),
        'CMAES_FB' : lambda     : toolbox['CMAES_FB'].generator(),
        'CMAES_FF' : lambda     : toolbox['CMAES_FF'].generator(),
    }


def printGA():
    print("                                                                     ")
    print("                   ██████╗  █████╗                                   ")
    print("                  ██╔════╝ ██╔══██╗                                  ")
    print("                  ██║  ███╗███████║                                  ")
    print("                  ██║   ██║██╔══██║                 optimization v0.1")
    print("                  ╚██████╔╝██║  ██║                                  ")
    print("                   ╚═════╝ ╚═╝  ╚═╝                                  ")
    print("                       ")

def initGA(tournsize=3,indpb=0.05,cxpb=0.5,mutpb=0.1):
    toolbox = base.Toolbox()
    # toolbox.register("attr_double", getX)
    # toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_double, n=DIM)
    # toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    # toolbox.register("evaluate", env_evalOneMax)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=indpb)
    toolbox.register("select", tools.selTournament, tournsize=tournsize)
    toolbox.register("cxpb", lambda: cxpb)
    toolbox.register("mutpb", lambda: mutpb)
    return toolbox, 


def printCMAES():
    print("                                                                     ")
    print("       ██████╗███╗   ███╗ █████╗ ███████╗███████╗                    ")
    print("      ██╔════╝████╗ ████║██╔══██╗██╔════╝██╔════╝                    ")
    print("      ██║     ██╔████╔██║███████║█████╗  ███████╗   optimization v1.0")
    print("      ██║     ██║╚██╔╝██║██╔══██║██╔══╝  ╚════██║                    ")
    print("      ╚██████╗██║ ╚═╝ ██║██║  ██║███████╗███████║                    ")
    print("       ╚═════╝╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝                    ")


def initCMAES(best_ind=None,sigma=5.0,N=10,mu=None):
    if(best_ind is None):
        print("Error in initCMAES, best_ind must be set")
    toolbox = base.Toolbox()
    #Sigma = np.array([sigma]*len(best_ind))
    #Sigma[0:36] /= 10
    #strategy  = cma.Strategy(centroid=best_ind, sigma=Sigma , lambda_=N)
    if(mu):
        strategy  = cma.Strategy(centroid=best_ind, sigma=sigma , lambda_=N, mu=mu)
    else:
        strategy  = cma.Strategy(centroid=best_ind, sigma=sigma , lambda_=N)
    toolbox.register("generator", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)
    return toolbox, strategy

def printBig(OPTIMIZATION_TYPE):
    if("CMAES" in OPTIMIZATION_TYPE):
        printCMAES()
        if("CO_OPT" in OPTIMIZATION_TYPE):
            print("mode : ff,fb co-optimization ")
        else:
            print("mode : fb optimization ")
    if(OPTIMIZATION_TYPE == "GA"):
        printGA()


