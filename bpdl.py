#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 14:55:12 2022

@author: mariano
"""



######### IMPORT ESPRESSOMD
import espressomd
from espressomd import polymer
from espressomd import shapes
from espressomd.interactions import HarmonicBond
from espressomd.interactions import FeneBond
from espressomd.interactions import AngleCosine

from espressomd.observables import ParticlePositions
from espressomd.accumulators import Correlator


######### Importing other relevant python modules

import numpy as np
import scipy as scy
import os, sys
from itertools import product, combinations, combinations_with_replacement
import random
import pdb
import pandas as pd
import re 
import warnings
warnings.filterwarnings("ignore")


try:
    import cPickle as pickle
except:
    import pickle
import pprint


import time
import importlib


if False:
    os.chdir( '/usr/users/mariano/shupaper/MD_code' )

#### import simulation settings
import esprenv as env

sys.path += [ env.strStorEspr, env.strHome]
os.chdir( env.strHome )

import func0 as func




# toy model

# real region
# sys.argv = ['cosimo','-model','rnactcf5','-modelName','bpdl', '-procid', '0','-maxt','100000','-tit','3000','-tsam','lin','-param','deg2e2','-c','c0CH','-reg','rnactcf5','-dopt','dCTpd,P2less2,P2passCH']
# sys.argv = ['cosimo','-model','deg2','-modelName','bpdl', '-procid', '0','-maxt','100000','-tit','100','-tsam','lin','-param','deg2e2','-c','c0CH','-reg','chr2s1s1z1m1','-dopt','dCTp,dCHp3,dP2less2,P2passCH']
# zero RA concentration



# =============================================================================
# Initiate the model class and update model settings according to user specified inputs
# =============================================================================
sysargv = func.argvRead( sys.argv ) 
exec( 'from ' + sysargv['modelName'] + '_model' + ' import ' + sysargv['modelName'] + ' as modelClass' )
model = modelClass( sysargv )










############
model.startTime = time.time()


###========================================================================
# Import LJ features; Set temperature, thermostat, ...
###========================================================================
print('EspressoMD features active:')
print(espressomd.features())
required_features = ["LENNARD_JONES"]
espressomd.assert_features(required_features)



ssystem = espressomd.System( box_l= model.BOX_L
                           , periodicity=[True,True,True]
                           )


ssystem.seed = np.int_( round(  random.random() * 1e+8) )



# =============================================================================
# Set cell settings
# =============================================================================
ssystem.cell_system.skin = model.SKIN



# =============================================================================
# Set thermostats 
# =============================================================================
ssystem.thermostat.turn_off()

print('Langevin selected')
ssystem.thermostat.set_langevin( kT = model.KbT, gamma= model.GAMMA, seed=round(random.random() * 1e+8))
        

       
# =============================================================================
# Set time step
# =============================================================================
ssystem.time_step = model.TIME_STEP
  
print('Time step:', ssystem.time_step)
    



###========================================================================
# Import I/O settings
###========================================================================
from espressomd.io.writer import vtf
from espressomd.io.writer import h5md


###========================================================================
# Check some features of the system
###========================================================================
print( 'Simulation features utilized:')
print( ssystem.cell_system.get_state())







# =============================================================================
# Build System & Run simulation
# =============================================================================
print('\n###----------------------------------------###\nsimulation settings:','\nstr_sys:', 
      model.str_syst , 
      '| sys_param:', model.str_param_file1 , 
      '| str_tsam:', model.str_tsam , 
      '\n###----------------------------------------###\n' , 
      flush= model.str_flush )



# create storage folder 
os.system('mkdir -p ' + env.strStorEspr + '/' + model.str_syst)
os.system('mkdir -p ' + env.strStorEspr + '/' + model.str_syst + '/obs' )







# =============================================================================
# =============================================================================
# # Warmup
# # =============================================================================
# =============================================================================



model.filenameTraj = env.strStorEspr + '/' + model.str_syst + \
    '/traj_'+ '_'.join(filter(None, [ model.dyn2, model.procid, model.str_param_file2, model.str_tsam]))



stOutcome = 'Warm up failed!'




for iti in range( model.nite):
    print('Tentative warmup ',iti+1)
    ssystem.part.clear()
    ssystem.time = 0
    sysid = 0






    try:
        
        # =============================================================================
        # Add constraint        
        # =============================================================================
        if ('strCos' in list( vars( model).keys() )) and ( model.strCos != None):
        
            # remove previously created constraints
            [ ssystem.constraints.remove( cosi ) for cosi in ssystem.constraints ]    
        
            spsha = espressomd.shapes.Sphere( center=[model.b/2., model.b/2., model.b/2.], radius= model.b/2., direction=-1 )
            spcostyp = model.alltypes[-1]+1
            spcos = ssystem.constraints.add( shape= spsha, particle_type= spcostyp)
        
        
        
        
        ###========================================================================
        # Add polymer to the simulation box at random positions
        ###========================================================================
        strWmp = True
        if ('strBond' in list( vars( model).keys() )) and ( model.strBond != None):
            if strWmp:
                if model.strBond == 'fene':
                    bond = FeneBond(k=model.k_fene, d_r_max=model.d_r_max, r_0=model.r_0_fene)
                else:
                    bond = HarmonicBond(k=model.k_harm, r_0=model.r_0_harm)
        
            else:
                bond = HarmonicBond(k=model.k_harm, r_0=model.r_0_harm)    

            #     
            ssystem.bonded_inter.add(bond)
    
            if ('strBondPolAngle' in list( vars( model).keys() ) ) and model.strBondPolAngle == 'cos' :
                bondAngle = AngleCosine( bend= model.kbend , phi0= model.cosangle )
                ssystem.bonded_inter.add( bondAngle)
        
    
            ### generate polymer positions
            polymers = model.polf( model.beads_per_chain, model.bond_length, model.cosDic )
            
            if type( polymers) == bool:
                raise 
                
    
            ###########
            for pi, p in enumerate(polymers):
                poldf = model.bsmap[ model.bsmap['polyid']==pi]

                i = 0
                for rid, rrow in poldf.iterrows():
                    for moni in range(int(rrow['bsbatch'])):
    
                        costmp = ssystem.part.add( id=sysid, 
                                                 pos=p[i], 
                                                 type=int(rrow['type']))
                        if i > 0:
                            costmp = ssystem.part[ sysid].add_bond((bond, sysid - 1))
                            if (i in model.pstiffl[pi]) and (i < model.npolall[pi] -1):
                                costmp = ssystem.part[ sysid].add_bond((bondAngle, sysid - 1, sysid + 1))
    
                        sysid = sysid +1
                        i = i +1
                            
            #
            poly = [[ [i, ssystem.part[i].type] for i in range(np.size(ssystem.part[:].type)) \
                     if ssystem.part[i].type not in model.mol_type]]
                
                
            ### exclude non bounded interaction for adjacent beads on same polymer
            if ('DPD' in model.strTherm) and False:
                pass
            else:
                for p in poly:
                    for j, m in p[1:]:
                        ssystem.part[ j].add_exclusion( j-1)
    
    
            print('Polymer added')            
    

                          
        # =============================================================================
        # Add dimers 
        # =============================================================================
        if model.strDimBond is not False:
            ## 
            if model.strDimBond == 'fene':
                model.dimLen = model.r0fenedimer
                bondDim = FeneBond(k= model.kfenedimer, d_r_max=model.drmaxfenedimer, r_0=model.r0fenedimer)
            elif model.strDimBond == 'rigid':
                bondDim = RigidBond( r= model.dimLen )
            else:
                bondDim = HarmonicBond(k=model.kHarmDimer, r_0=model.r0harmdimer, 
                                       r_cut=model.rcharmdimer)
            
       
            # type of bonds that will be used
            ssystem.bonded_inter.add( bondDim)
            
            
            for dimi in range( model.ndimi[0]): # temp
            # for dimi in range( model.ndimi): # old
                booDim = False
                while booDim is False:
                    pp1 = np.random.random(3) * model.b
                    if ssystem.part[:].pos.size == 0 :
                        while np.linalg.norm( pp1 - model.b/2. ) > model.b/2. - model.bond_length :
                            pp1 = np.random.random(3) * model.b
                                
                        pp2 = np.copy( pp1 )
                        dire = np.random.randint(0,3)
                        pp2[ dire ] = pp2 [ dire ] + model.dimLen            
                        for dimi2 in range(10):
                            if ( np.linalg.norm( pp2 - model.b/2. ) > model.b/2. - model.bond_length )  :
                                pp2 = np.copy( pp1 )
                                dire = np.random.randint(0,3)
                                pp2[ dire ] = pp2 [ dire ] + model.dimLen
                            else : 
                                booDim = True
                                break            
                            
                    else : 
                        while np.linalg.norm( pp1 - model.b/2. ) > model.b/2. - model.bond_length or np.any( np.linalg.norm( pp1 - ssystem.part[:].pos, axis=1) < model.bond_length ):
                            pp1 = np.random.random(3) * model.b
                        
                        
                        
                        pp2 = np.copy( pp1 )
                        dire = np.random.randint(0,3)
                        pp2[ dire ] = pp2 [ dire ] + model.dimLen


                        
                        for dimi2 in range(10):
                            if ( np.linalg.norm( pp2 - model.b/2. ) > model.b/2. - model.bond_length ) or np.any( np.linalg.norm( pp2 - ssystem.part[:].pos, axis=1) < model.bond_length )  :
                                pp2 = np.copy( pp1 )
                                dire = np.random.randint(0,3)
                                pp2[ dire ] = pp2 [ dire ] + model.dimLen
                            else : 
                                booDim = True
                                break
                

                cos1 = ssystem.part.add(type= model.anistype[0], pos= pp1 )
                cos2 = ssystem.part.add(type= model.anistype[1], pos= pp2 )
                
                #
                costmp = ssystem.part[ cos1.id ].add_bond(( bondDim, cos2.id))
    
    
            print('Added dimers')            
            
            
            
            
            
                
        ###========================================================================
        # Add particles to the simulation box at random positions
        ###========================================================================
        for idx, typi in enumerate(model.mol_type):
            for i in range( model.molPart[ idx]):
                ppos = np.random.random(3) * model.b
                while ( np.linalg.norm( ppos - model.b/2. ) > model.b/2. -model.bond_length ) or np.any( np.linalg.norm( ppos - ssystem.part[:].pos, axis=1) < model.bond_length )  :
                    ppos = np.random.random(3) * model.b
                costmp = ssystem.part.add(type= model.mol_type[idx], pos= ppos)
                # mol_pos[ ii, :] = ppos
                # ii = ii +1

        #
        moly = []
        for idx, typi in enumerate(model.mol_type):
            moly += [[ [i, ssystem.part[i].type] for i in range(np.size(ssystem.part[:].type)) \
                     if ssystem.part[i].type == typi]]
        

        # add check of constraint
        if ('strCos' in list( vars( model).keys() )) and ( model.strCos != None):
            if np.all( np.linalg.norm( ssystem.part[:].pos - model.b/2., axis=1 ) <= model.b/2. - model.bond_length ):
                print('Constraint check fulfilled')
            else:
                print('Constraint check not fulfilled: aborting')
                raise


        print('Binders added')            






        # =============================================================================
        # Save original types
        # =============================================================================
        model.origTypes = ssystem.part[:].type    



    

        ###========================================================================
        # IO
        ###======================================================================== 
        fp, dft = func.confBegin( ssystem, model, espressomd.io.writer)
    
        obss_all = func.obsBegin( ssystem, model)



        
        
 

        ###========================================================================
        # Warm up
        ###========================================================================            
        for wmpl in model.wmpList:
            
            WARM_N_TIME = len( model.wmp_sam)
   

            # 
            if wmpl['step'] == 'eq':
                print("\rIntroduce hard sphere potentials (non interacting equilibration)")
                
            elif wmpl['step'] == 'pot':
                print("\rIntroduce interaction potentials with linear gradient")
                # Remove cap on force
                ssystem.force_cap = 0
                # act_min_dist = ssystem.analysis.min_dist() 



            for i, wmp_s in enumerate( list(model.wmp_sam)):
                print("\rWarmup integration %d/%d at time=%.0f of duration %.3f" % (i+1, WARM_N_TIME, ssystem.time, wmp_s * model.TIME_STEP), end='', flush=model.str_flush)
 

                    
                if wmpl['step'] == 'eq':
                    ###========================================================================
                    # Change potentials
                    ###========================================================================            
                    ### Add hard sphere unbounded interactions
                    for typi in model.allpairs:
                        ssystem.non_bonded_inter[ typi[0], typi[1] ].wca.set_params(
                            epsilon = model.ljh['eps'][ typi[0], typi[1], i], 
                            sigma   = model.ljh['sig'][ typi[0], typi[1], i]
                            )


        


                    # =============================================================================
                    # Add interactions with constraint        
                    # =============================================================================
                    if ('strCos' in list( vars( model).keys() )) and ( model.strCos != None):
                    
                        for typi in model.alltypes:
                            ssystem.non_bonded_inter[ typi, spcostyp ].wca.set_params(
                                epsilon = model.ljh['eps'][ typi, typi, i], 
                                sigma   = model.ljh['sig'][ typi, typi, i]
                                )

                                                        
                    # CAP on force
                    ssystem.force_cap = model.ljhcap[ i]
                    # act_min_dist = ssystem.analysis.min_dist()


                    # =============================================================================
                    # Zeroing velocities            
                    # =============================================================================
                    if model.strKillv :
                        espressomd.galilei.GalileiTransform.kill_particle_motion( ssystem )
        
                    # =============================================================================
                    # Zeroing forces
                    # =============================================================================
                    if model.strKillF:
                        espressomd.galilei.GalileiTransform.kill_particle_forces( ssystem )
    


 



                elif wmpl['step'] == 'pot':
                       
                    ### Add lj interactions between types with non bounded interactions
                    for typi in model.pairsb:
                        ssystem.non_bonded_inter[ typi[0], typi[1] ].lennard_jones.set_params(
                            epsilon = model.ljb['eps'][ typi[0], typi[1], i], 
                            sigma   = model.ljb['sig'][ typi[0], typi[1], i], 
                            cutoff  = model.ljb['cut'][ typi[0], typi[1], i]
                            , shift='auto')
            


                        
                ###========================================================================
                # Integrate Dynamics
                ###========================================================================            
                ssystem.integrator.run( wmp_s)
                

                


                
            
                ###========================================================================
                # Measure
                ###========================================================================  
                if model.wmpSaveObs:
                    obss_all = func.obsInterm( ssystem, model, obss_all, wmpl['step'])
                                    




            # =============================================================================
            # Save configuration and Measure at the end of Warmup Stage
            # =============================================================================
            fp, dft = func.confSave( ssystem, model, espressomd.io.writer, fp, dft, strVel = model.strVel, precision=model.savePrecision)
        
   





        # at the end of warmup no capping of force is in place
        ssystem.force_cap = 0
        stOutcome = '\nWarm up successful!'
        break
    
    

        
    except (Exception,RuntimeError) as e:
        print("\rError at run %d at time=%.0f " % (i+1, ssystem.time), end='', flush=model.str_flush)
        func.confEndWarmupError( model, espressomd.io.writer, fp )
        print( e)
                
        continue
    
    except Exception as e:
        func.confEndWarmupError( model, espressomd.io.writer, fp )
        print( e)
                
        continue
    
    
    
            
###        
print(stOutcome)
if stOutcome == 'Warm up failed!': 
    raise
print("\rWarmup ended at run %d at time=%.0f " % (i+1, ssystem.time), end='', flush= model.str_flush)















    

# =============================================================================
# ###========================================================================
# # Dynamics Starts
# ###========================================================================    
# =============================================================================
try:        
    

    # =============================================================================
    # Start Iteration over Dynamics Stages
    # =============================================================================
    for sidx, vt in enumerate( model.tsteps):
        print('\nStarting simulation phase:', vt['name'])


        ###========================================================================
        # Change potentials
        ###========================================================================            
        ### Add lj interactions between same specie
        for typi in model.pairsb:
            ssystem.non_bonded_inter[ typi[0], typi[1] ].lennard_jones.set_params(
                epsilon = vt['potentials']['ljb eps'][ typi[0], typi[1] ], 
                sigma   = vt['potentials']['ljb sig'][ typi[0], typi[1] ], 
                cutoff  = vt['potentials']['ljb cut'][ typi[0], typi[1] ]
                , shift='auto')



        print('Potentials modified')      
       
        








        dynprocl = []

        # =============================================================================
        # Loop extrusion
        # =============================================================================
        if model.strLope:

            
            lopePairs = np.empty((0,3), dtype=np.int64)
            loppy = [0,1]
            lopedtMod = 0
            dynprocl += [0]
            #
            type1 = [ np.isin( ssystem.part[:].type, model.idanis), 
                      np.isin( ssystem.part[:].type, model.idanis)
                      ] 
            type2 = [ ssystem.part[:].type == model.anistype[0], ssystem.part[:].type == model.anistype[1]]
            id1 = [ssystem.part[:].id [ type1[0] ], ssystem.part[:].id [ type1[1] ]]
            id2 = [ssystem.part[:].id [ type2[0] ], ssystem.part[:].id [ type2[1] ]]
            
            # where to stall?
            if re.match( '.*Art.*|.*Real.*', model.modelods_paramsheet ):
                model.idd1not = func.defLEtypes2( ssystem, model, strmode= model.lopeInsulMode )
            else:
                model.idd1not = func.defLEtypes2( ssystem, model, rseed= model.seedCtcfStall, coheStallSitesFract = model.coheStallSitesFract )
            
            #      
            if model.strLopeBond == 'fene':
                bondLope = FeneBond(k= model.kfeneLope, d_r_max= model.drmaxFenelope, r_0= model.r0feneLope)
                bondLopeM = HarmonicBond(k= model.kharmLope, r_0= model.r0harmLope, r_cut=model.rcHarmLope)
            elif model.strLopeBond == 'harm':
                bondLope = HarmonicBond(k= model.kharmLope, r_0= model.r0harmLope, r_cut=model.rcHarmLope)
                bondLopeM = HarmonicBond(k= model.kharmLopeM, r_0= model.r0harmLope, r_cut=model.rcHarmLopeM)
                
            ssystem.bonded_inter.add( bondLope)
            ssystem.bonded_inter.add( bondLopeM)


            
            #
            dimask = np.isin( ssystem.part[:].type  , model.anistype)
            sysanis = pd.DataFrame({
                'id' : ssystem.part[:].id[ dimask ],
                'type' : ssystem.part[:].type[ dimask ]
                    })
            
            sysanis['dimPairs'] = np.int_( np.arange(1.1, sysanis.shape[0]/2. + 1.1,.5) )[:sysanis.shape[0]]         
            

            # ids of CT molecules
            model.idct = ssystem.part[:].id[ np.isin( ssystem.part[:].type, model.lopeStallMol ) ]
            model.stallAnywayIdLope = ssystem.part[:].id[ np.isin( ssystem.part[:].type, model.stallAnywayTypeLope ) ].tolist()
            
            # ids of get-away sites
            model.detachAlwaysIdLope = ssystem.part[:].id[ np.isin( ssystem.part[:].type, model.detachAlwaysType ) ].tolist()


            # type dependent lope probabilities
            lopeBirthProbSysDF = pd.DataFrame( data={
                'type' : ssystem.part[:].type[np.isin( ssystem.part[:].type, model.pol_type)] 
                })
            model.lopeBirthProbSys = lopeBirthProbSysDF.merge( model.lopeBirthProbDf,
                              how= 'left', on='type').lopeBirthProb.values

            model.lopeBirthProbSysBase = lopeBirthProbSysDF.merge( model.lopeBirthProbDfBase,
                              how= 'left', on='type').lopeBirthProb.values
    
            ##
            model.idCHloadP2proxy = ssystem.part[:].id[ np.isin( ssystem.part[:].type, model.lopeHighProbBirthType ) ]
            model.idP2proxyLE = ssystem.part[:].id[ np.isin( ssystem.part[:].type, model.P2proxyLEtype ) ]
    
    
        # =============================================================================
        #             Reel - Lope interactions
        # =============================================================================

        if (len(model.strReel) > 0) and ('reelIdxNotLope' in dir(model)) :
            reelLopeBool1 = True
        else:
            reelLopeBool1 = False

        if (len(model.strReel) > 0) and ('boundReelType' in dir(model)) :
            reelLopeBool2 = True
        else:
            reelLopeBool2 = False



        # =============================================================================
        #             Reel
        # =============================================================================

        if len(model.strReel) > 0:
            
            #
            if ('P2passCH' in dir( model)) and model.P2passCH:
                P2passCH = model.P2passCH
            else:
                P2passCH = False                
            
            dynprocl += [1]
            bondReel = []
            bondReelM = []
            id2reel = []
            reelPairs = []
                
            for idxreel, strReeli in enumerate(model.strReel):
                #
                if strReeli:
                    reelPairs += [np.empty((0,4), dtype=np.int64)]
                    reelly = [0,1]
                    reeldtMod = 0

                    #
                    type2reel = [ np.isin( ssystem.part[:].type, model.molreeltyp[idxreel]), 
                              np.isin( ssystem.part[:].type, model.molreeltyp[idxreel])
                              ] 

    
                    id2reel += [[ ssystem.part[:].id [ type2reel[0] ], ssystem.part[:].id [ type2reel[1] ] ]]
                    

                    
                    #      
                    if model.strReelBond[idxreel] == 'fene':
                        bondReel += [ FeneBond(k= model.kfeneReel[idxreel], d_r_max= model.drmaxFeneReel[idxreel], r_0= model.r0feneReel[idxreel]) ]
                        bondReelM += [ HarmonicBond(k= model.kharmReel[idxreel], r_0= model.r0harmReel[idxreel],
                                                   r_cut= model.rcHarmReel[idxreel]) ]
                    elif model.strReelBond[idxreel] == 'harm':
                        bondReel += [ HarmonicBond(k= model.kharmReel[idxreel], r_0= model.r0harmReel[idxreel],
                                                   r_cut= model.rcHarmReel[idxreel]) ]
                        bondReelM += [ HarmonicBond(k= model.kharmReel[idxreel], r_0= model.r0harmReel[idxreel],
                                                   r_cut= model.rcHarmReel[idxreel]) ]
                        
                        
                    ssystem.bonded_inter.add( bondReel[idxreel] )
                    ssystem.bonded_inter.add( bondReelM[idxreel] )
                        
                        
                        




        if model.strSpec:
            dynprocl += [2]
            specPairs = np.empty((0,2), dtype=np.int64)
           
            specNHSpring = ssystem.part[:].id[  np.isin( ssystem.part[:].type, model.bsspec_type) ]
            specNHFBSpring = ssystem.part[:].id[  np.isin( ssystem.part[:].type, model.molspec_type) ]
            
            specNHSpring = np.int_(np.concatenate(( specNHSpring[:,None], np.zeros((len(specNHSpring),1))), axis=1 ))
            specNHFBSpring = np.int_(np.concatenate(( specNHFBSpring[:,None], np.zeros((len(specNHFBSpring),1))), axis=1))


            if model.strSpecBond == 'fene':
                bondSpec = FeneBond(k= model.kfeneSpec, d_r_max= model.drmaxFeneSpec, r_0= model.r0feneSpec)
            elif model.strSpecBond == 'harm':
                bondSpec = HarmonicBond(k= model.kHarmSpec, r_0= model.r0HarmSpec
                                        , r_cut=model.rcHarmSpec )

            ssystem.bonded_inter.add( bondSpec )    





        # obs start settings        
        lopdfall = pd.DataFrame([])
        reeldfall = pd.DataFrame([])
        specdfall = pd.DataFrame([])
        model.startDynTime = time.time()



        ###========================================================================
        # Start Integration Iterations
        ###========================================================================            
        sampling_v = vt['sampling_v']
        sampling_iterations = np.size(sampling_v)


        for i, smpl in enumerate(list(np.int_(sampling_v))):
            # =============================================================================
            # Save configuration
            # =============================================================================
            fp, dft = func.confSave( ssystem, model, espressomd.io.writer, fp, dft, strVel = model.strVel, precision=model.savePrecision)
        
    
               
            # =============================================================================
            # Run integration
            # =============================================================================
            print("\rrun %d of %d at time=%.0f of duration %d" % (i, sampling_iterations, ssystem.time, smpl), end='')
    
            
            # =============================================================================
            #     Loop Extrusion
            # =============================================================================
            if model.strLope:

                
        
                if (model.lopedt < smpl) or (smpl + lopedtMod > model.lopedt):
                    lope_smlp = [model.lopedt - lopedtMod] + [model.lopedt] * np.int_( np.floor( (smpl - ( model.lopedt - lopedtMod) ) / model.lopedt ) ) 
                    lopedtMod = (smpl - ( model.lopedt - lopedtMod) ) % model.lopedt

                   
                    for lopei in lope_smlp:
                        
                        # =============================================================================
                        # Integrate first                    
                        # =============================================================================
                        ssystem.integrator.run(lopei)
                        


                        random.shuffle( dynprocl )    
                        for dynprocli in dynprocl:

                            if dynprocli == 0:
                                ## check whether to dissociate some LEF
                                lopeDeathRand = np.random.random( lopePairs.shape[0])
                                    
                                lopeIdxDead = np.where( ( lopeDeathRand < model.lopeDeathProb ) | \
                                    ( np.isin( lopePairs[:,0], model.detachAlwaysIdLope ) ) )[0]
            
            
                                    
                                # select the entire dimer
                                dimPairs = sysanis[ np.isin( sysanis.id, lopePairs[ lopeIdxDead, 1 ]) ].dimPairs.tolist()
                                lopeIdxDead = np.isin( lopePairs[:,1], sysanis[ np.isin( sysanis.dimPairs, dimPairs )].id )
    
                                for lopair in lopePairs[ lopeIdxDead, : ].tolist():
                                    ssystem.part[ lopair[0] ].delete_bond(( bondLope, lopair[1] ))

                                    if model.strLopeBond == 'fene':
                                        ssystem.part[ lopair[0] ].delete_exclusion( lopair[1]  )
                                    elif model.strLopeBond == 'harm':
                                        pass                                  
    
                                    # remove spring bond on near bead - NEW
                                    ssystem.part[ lopair[0]+lopair[2] ].delete_bond(( bondLopeM, lopair[1] ))
    
                                    
                                    
                                    # give back particle type (hence non-bonded interactions)
                                    ssystem.part[ lopair[1] ].type = model.origTypes[ lopair[1] ]
                                    




                                ## check if any LEF is moving forward
                                random.shuffle(loppy)
                                for idside in loppy: 
        
                                    if model.specCheck == 'distance':
                                        ## check if any CT is close to the stalling site - NEW
                                        dmdf = pd.DataFrame( scy.spatial.distance_matrix( 
                                            ssystem.part[ : ].pos[ model.idd1not[idside] ], 
                                            ssystem.part[ : ].pos[ model.idct ]
                                            ))
                                        dmsubdf = dmdf[ dmdf <= model.lopeStall_cutoff ].dropna(axis=0, how='all').dropna(axis=1, how='all')
                                        dmsub = np.array( dmsubdf )
                                        if dmsub.size == 0: continue
    
                                        lopePairsMask = np.where( (lopePairs[:, 2] == model.dmov[ idside]) & ~lopeIdxDead )[0]
        
        
                                        boolMove = ~np.isin( lopePairs[ lopePairsMask, 0] + lopePairs[ lopePairsMask, 2], 
                                                    np.array(model.idd1not[idside])[ dmsubdf.index ].tolist() + model.stallAnywayIdLope
                                                    )    
        
                                    elif model.specCheck == 'bond':
                                        # check if any CT is bound to Cs   
                                        lopePairsMask = np.where( (lopePairs[:, 2] == model.dmov[ idside]) & ~lopeIdxDead )[0]
                                        
        
                                        bondl = specPairs[ np.isin( specPairs[:,0], model.idd1not[idside]), 0 ]
    
                                        reelPoints = []
                                        for idxreel2 in range(len(reelPairs)):
                                            reelPoints += reelPairs[ idxreel2][:,0].tolist()
                                            

                                        boolMove = ~np.isin( lopePairs[ lopePairsMask, 0] + lopePairs[ lopePairsMask, 2], 
                                                    bondl.tolist() + model.stallAnywayIdLope + reelPoints
                                                    )    

        
    
                                    if boolMove.sum() > 0:
                                        ## check if any is close enough to move
                                        idxcos = np.where( boolMove )[0]
                                        iddimove = lopePairs[ lopePairsMask[idxcos], 0 ] + lopePairs[ lopePairsMask[idxcos], 2]
                                        
                                        dl = np.linalg.norm( 
                                            ssystem.part[ : ].pos[ iddimove ] - ssystem.part[ : ].pos[ lopePairs[ lopePairsMask[idxcos], 1]] ,
                                            axis=1 )
                                        dlid = np.where( dl <= model.lope_MoveCutoff)[0]
                                        
                
                                        closeTypPairsMove = list( zip( lopePairs[ lopePairsMask[idxcos[ dlid]], 0] + \
                                                                      lopePairs[ lopePairsMask[idxcos[ dlid]], 2] , 
                                                                      lopePairs[ lopePairsMask[idxcos[ dlid]], 1] ,
                                                                      [model.dmov[ idside] ] * len(dlid)
                                                                      )
                                                                 )
                                        bondedTypPairs = lopePairs[ lopePairsMask[idxcos[ dlid]], :].tolist()
                
                                        for lopair in bondedTypPairs:
                                            ssystem.part[ lopair[0] ].delete_bond(( bondLope, lopair[1] ))
            
                                            if model.strLopeBond == 'fene':
                                                ssystem.part[ lopair[0] ].delete_exclusion( lopair[1]  )
                                            elif model.strLopeBond == 'harm':
                                                pass        
            
                                            # remove spring bond on near bead - NEW
                                            ssystem.part[ lopair[0]+lopair[2] ].delete_bond(( bondLopeM, lopair[1] ))
                                            
            
                
                                        # for lopair in closeTypPairsMove[ lopeIdxMove, : ].tolist(): # move all that can and that have passed the random threshold
                                        for lopair in closeTypPairsMove: #  move all that can
                                            ssystem.part[ lopair[0] ].add_bond(( bondLope, lopair[1] ))

                                            if model.strLopeBond == 'fene':
                                                ssystem.part[ lopair[0] ].add_exclusion( lopair[1]  )
                                            elif model.strLopeBond == 'harm':
                                                pass                                         
    
                                            # add spring bond on near bead - NEW
                                            costmp = ssystem.part[ lopair[0]+lopair[2] ].add_bond(( bondLopeM, lopair[1] ))
    
                                            
                    
                                        # stopping particles that just moved ahead
                                        if len( closeTypPairsMove ) > 0:
                                            ssystem.part[ np.array( closeTypPairsMove)[ :, 0 ] ].v = np.zeros(( len( closeTypPairsMove),3))
                                            ssystem.part[ np.array( closeTypPairsMove)[ :, 1 ] ].v = np.zeros(( len( closeTypPairsMove),3))
                                            ssystem.part[ np.array( closeTypPairsMove)[ :, 0 ] + np.array( closeTypPairsMove)[ :, 2 ] ].v = np.zeros(( len( closeTypPairsMove),3))
                                            ssystem.part[ np.array( bondedTypPairs)[ :, 0 ] ].v = np.zeros(( len( bondedTypPairs),3))
            
            

                                        # move LEF
                                        lopePairs[ lopePairsMask[idxcos[ dlid]], 0] = lopePairs[ lopePairsMask[idxcos[ dlid]], 0] + \
                                            lopePairs[ lopePairsMask[idxcos[ dlid]], 2]
        
        
        
                
                                ## check if new LEF arise      
                                if False:
                                    reelPoints = []
                                    for idxreel2 in range(len(reelPairs)):
                                        reelPoints += reelPairs[ idxreel2][:,0].tolist() + (reelPairs[ idxreel2][:,0] +1).tolist() + (reelPairs[ idxreel2][:,0] -1).tolist()

                                else:
                                    ##                  
                                    idd2 = np.array( list( set(id2[idside]) - set( lopePairs[:,1]) ) )
                                    dmdf = pd.DataFrame( scy.spatial.distance_matrix( 
                                        ssystem.part[ : ].pos[ model.idCHloadP2proxy ], 
                                        ssystem.part[ : ].pos[ model.idP2proxyLE ]
                                        ))
                                    dmsubdf = dmdf[ dmdf <= model.lopePeproxy_cutoff ].dropna(axis=0, how='all').dropna(axis=1, how='all')
                                    dmsub = np.array( dmsubdf )
                                    if dmsub.size == 0: 
                                        reelPoints = []
                                    else:                                            
                                        reelPoints = np.array(model.idCHloadP2proxy)[ dmsubdf.index ].tolist()


                                    
                                ##
                                random.shuffle(loppy)
                                newselected = np.empty((0,3), dtype=np.int64)
                                for idside in loppy:
                                    # remove the pairs that are already LEF...
                                    if reelLopeBool1:
                                        idd1 = np.array( list( set(id1[idside]) - set( lopePairs[:,0]) - set( reelPairs[model.reelIdxNotLope[0]][:,0])) )
                                    else:
                                        idd1 = np.array( list( set(id1[idside]) - set( lopePairs[:,0]) ) )
                                        
                                    idd2 = np.array( list( set(id2[idside]) - set( lopePairs[:,1]) ) )
                                
                                    # it can happen that there's no new particle to attach or any binding site left
                                    if len(idd1) == 0 or len(idd2) == 0: continue
                                
                                    # 
                                    pos1 = ssystem.part[ idd1.tolist() ].pos
                                    pos2 = ssystem.part[ idd2.tolist() ].pos

                                    
                                    dmdf = pd.DataFrame( scy.spatial.distance_matrix( pos1, pos2))
                                    dmsubdf = dmdf[ dmdf <= model.lope_cutoff ].dropna(axis=0, how='all').dropna(axis=1, how='all')
                                    dmsub = np.array( dmsubdf )
                                    if dmsub.size == 0: continue
                                    
                                    closeTypPairsNew = \
                                        np.array( list( set( zip( idd1[ dmsubdf.index], idd2[ dmsubdf.columns[ np.nanargmin( dmsub, 1)] ] ) ) & \
                                                       set( zip( idd1[ dmsubdf.index[ np.nanargmin( dmsub, 0)] ], idd2[ dmsubdf.columns ] ) ) )
                                                 )
                
                
                                    lopeBirthRand = np.random.random( len(closeTypPairsNew))
                                    lopeIdxNew = np.where( lopeBirthRand < 
                                                          np.where( 
                                                              np.isin( closeTypPairsNew[:,0], reelPoints)
                                                              , model.lopeBirthProbSys[ closeTypPairsNew[:,0]], model.lopeBirthProbSysBase[ closeTypPairsNew[:,0]]) 
                                                          )[0]
                                    
                                    #
                                    newselected = np.concatenate( ( newselected, 
                                                                    np.int_( np.concatenate( ( 
                                                                        closeTypPairsNew[ lopeIdxNew, : ] ,
                                                                        np.ones( (lopeIdxNew.size, 1)) * model.dmov[ idside] ), axis=1 ) )
                                                                    ), axis=0 ) # add new bonds
        
    
     
                                 
                                ### lope analysis
                                lpdf = pd.DataFrame( newselected)
                                lpdf.sort_values(by=1, inplace=True)
                                lpdf.columns = ['poltype', 'anistype', 'strand']
                                
        
                                
                                lloop = lpdf.merge( sysanis, how= 'left', left_on = 'anistype', right_on = 'id' )
                                # this filters only pairs belonging to the same dimer
                                llooptmp = lloop.groupby('dimPairs').agg( len).reset_index()[['dimPairs','id']]
                                llooptmp = llooptmp[ llooptmp ['id'] == 2 ]

                                lloop2 = lloop.merge( llooptmp['dimPairs'], how='right', on='dimPairs' )
                                # this filters only for dimers binding locally, not binding to distal sites
                                if lloop2.shape[0] > 0 :
                                    lloop3 = lloop2.pivot_table( columns = 'type', index='dimPairs', values='poltype')
                                    lloop3['loopDim'] = lloop3[ model.anistype[0]] - lloop3[ model.anistype[1]]
                                    
                                    ### this filters for local loops (loopDim=1,2), but also well directioned (loopDim!=-1,-2)
                                    loopid1 = lloop3[ np.isin( lloop3.loopDim, model.minLoopSize ) ].reset_index()[ model.anistype[0] ].tolist() + \
                                                lloop3[ np.isin( lloop3.loopDim, model.minLoopSize ) ].reset_index()[ model.anistype[1] ].tolist()
                                    newselected = newselected[ np.isin( newselected[:,0], loopid1), : ]                             
                                
                                else:
                                    newselected = np.empty((0,3), dtype=np.int64)
    
    



                                    
                                ## first remove particles killed at the beginning
                                # remove dead dimer
                                lopePairs = lopePairs[ ~lopeIdxDead, :]
                                
                                
                                ## add bonds
                                if ( newselected.ndim > 1) and ( newselected.size > 0): 
                                    

                                    
                                    for lopair in newselected.tolist():
                                        costmp = ssystem.part[ lopair[0] ].add_bond(( bondLope, lopair[1] ))

                                        if model.strLopeBond == 'fene':
                                            ssystem.part[ lopair[0] ].add_exclusion( lopair[1]  )
                                        elif model.strLopeBond == 'harm':
                                            pass                                    
                                        
                                        # add spring bond on near bead
                                        costmp = ssystem.part[ lopair[0] +lopair[2] ].add_bond(( bondLopeM, lopair[1] ))
                                        
                                        
                                        # change particle type
                                        ssystem.part[ lopair[1] ].type = model.origTypes[ lopair[1] ] +1 # model.inertType
                                        
            
                                    # stopping particles that associate
                                    ssystem.part[ newselected[ :, 0 ] ].v = np.zeros((newselected.shape[0],3))        
                                    ssystem.part[ newselected[ :, 1 ] ].v = np.zeros((newselected.shape[0],3))        
                                    ssystem.part[ newselected[ :, 0 ] + newselected[ :, 2 ] ].v = np.zeros((newselected.shape[0],3))        
                
                                    # add new born LEF
                                    lopePairs = np.concatenate( ( lopePairs, 
                                                                 newselected
                                                                    ), axis=0 ) # add new bonds
                
    
    

    
                                ###========================================================================
                                # Measure lope props
                                ###========================================================================      
                                lopdf = pd.DataFrame( lopePairs, columns=['bondid1','bondid2','direction'] )
        
    
                                
        

                            elif dynprocli == 1:
    
                                # =============================================================================
                                # Polymerase reeling
                                # =============================================================================
    
                                for idxreel, strReeli in enumerate(model.strReel):

                                    ## check whether to dissociate some REF
                                    if reelPairs[idxreel].size > 0:
                                        random.shuffle(reelly)
                                        reelpairdeath = np.zeros( reelPairs[idxreel].shape[0], dtype=bool)
                                        for idside in reelly: 
                                            reelpairdeath = reelpairdeath | ((reelPairs[idxreel][:, 2] == model.dmov[ idside]) & ( np.isin( reelPairs[idxreel][:, 0], model.geneend[idxreel][idside])))
                                            
                                        reelIdxDead = ( np.random.random( reelpairdeath.shape[0]) < model.reelDeathProb[idxreel] ) & reelpairdeath
                                        
                                    else:
                                        reelIdxDead = np.zeros( reelPairs[idxreel].shape[0], dtype=bool)
                                        
                                        
                                            
                
                                    for reelpair in reelPairs[idxreel][ reelIdxDead, : ].tolist():

                                        ssystem.part[ reelpair[0] ].delete_bond(( bondReel[idxreel], reelpair[1] ))

                                        if model.strReelBond[idxreel] == 'fene':
                                            ssystem.part[ reelpair[0] ].delete_exclusion( reelpair[1]  )
                                        elif model.strReelBond[idxreel] == 'harm': 
                                            pass
                                        
                                        # remove spring bond on near bead - NEW
                                        ssystem.part[ reelpair[0]+reelpair[2] ].delete_bond(( bondReelM[idxreel], reelpair[1] ))

                                        
                                        # give back particle type (hence non-bonded interactions)
                                        ssystem.part[ reelpair[1] ].type = model.origTypes[ reelpair[1] ]
                                        
                                        
                                    if reelLopeBool2 and reelPairs[idxreel][ reelIdxDead, 0 ].size > 0:
                                        # =============================================================================
                                        #             Change detached bs category back to the original one
                                        # =============================================================================
                                        ssystem.part[ reelPairs[idxreel][ reelIdxDead, 0 ] ].type = model.origTypes[ reelPairs[idxreel][ reelIdxDead, 0 ] ] # model.inertType
                                        ssystem.part[ reelPairs[idxreel][ reelIdxDead, 0 ] ].v = np.zeros((reelPairs[idxreel][ reelIdxDead, : ].shape[0],3))                                                                     
                                
                                
                                    

            

            
            
                    
                                    ## check if any REF is moving forward
                                    random.shuffle(reelly)
                                    for idside in reelly: 
                                        for idxreelpos in model.reelpos[ idxreel]:
                                            reelPairsMask2 = np.where( (reelPairs[idxreel][:, 2] == model.dmov[ idside]) & ~reelIdxDead)[0]
                                            reelPairsMask = np.where( (reelPairs[idxreel][:, 2] == model.dmov[ idside]) & \
                                                                     (reelPairs[idxreel][:, 3] == idxreelpos) & \
                                                                         ~reelIdxDead )[0]
                                            
                                            if (reelPairs[idxreel].size > 0) and (reelPairsMask.size > 0):
                                                # stalling-dependence on orientation
                                                
                                                if P2passCH == 'P2notpass':
                                                    idxcos = np.where( ~np.isin( reelPairs[idxreel][ reelPairsMask, 0] + reelPairs[idxreel][ reelPairsMask, 2], 
                                                                                model.idd1notReel[idxreel][idside] + reelPairs[idxreel][ reelPairsMask2, 0].tolist() + lopePairs[:,0].tolist() ) )[0]
                                                elif P2passCH == 'both':
                                                    idxcos = np.where( ~np.isin( reelPairs[idxreel][ reelPairsMask, 0] + reelPairs[idxreel][ reelPairsMask, 2], 
                                                                                model.idd1notReel[idxreel][idside] + reelPairs[idxreel][ reelPairsMask2, 0].tolist()  ) )[0]
                                                else:
                                                    idxcos = np.where( ~np.isin( reelPairs[idxreel][ reelPairsMask, 0] + reelPairs[idxreel][ reelPairsMask, 2], 
                                                                                model.idd1notReel[idxreel][idside] + reelPairs[idxreel][ reelPairsMask2, 0].tolist()  ) )[0]
                                                    
                                                iddimove = reelPairs[idxreel][ reelPairsMask[idxcos], 0 ] + reelPairs[idxreel][ reelPairsMask[idxcos], 2]
                                                dl = np.linalg.norm( 
                                                    ssystem.part[ : ].pos[ iddimove ] - ssystem.part[ : ].pos[ reelPairs[idxreel][ reelPairsMask[idxcos], 1]] ,
                                                    axis=1 )
                                                
                                                
                                                dlid = np.where( dl <= model.reel_MoveCutoff[idxreel])[0]


                                                
                                                if dlid.size > 0:
                                                    
                                                    if idxreelpos == 0:  
                                        
                                                        reelMoveRand = np.random.random( len(reelPairsMask[idxcos[ dlid]]))
                                                        reelIdxMove = np.where( reelMoveRand < model.reelMoveProbStart )[0]
                                                        
                                                        reelPairsMask2 = reelPairsMask[idxcos[ dlid]][reelIdxMove]

                                                        
                                                    else:
                                                        reelPairsMask2 = reelPairsMask[idxcos[ dlid]]
                                                        
            
                                                    closeTypPairsMove = np.array(list( zip( reelPairs[idxreel][ reelPairsMask2, 0] + \
                                                                                  reelPairs[idxreel][ reelPairsMask2, 2] , 
                                                                                  reelPairs[idxreel][ reelPairsMask2, 1] ,
                                                                                  [model.dmov[ idside] ] * len(reelPairsMask2) )
                                                                             ))
                                                    bondedTypPairs = reelPairs[idxreel][ reelPairsMask2, :3]
                            
                            
                                                    for reelpair in bondedTypPairs.tolist(): # [reelIdxMove]
                                                        ssystem.part[ reelpair[0] ].delete_bond(( bondReel[idxreel], reelpair[1] ))
                                                        # ssystem.part[ reelpair[0] ].delete_exclusion( reelpair[1]  )
                                                        if model.strReelBond[idxreel] == 'fene':
                                                            ssystem.part[ reelpair[0] ].delete_exclusion( reelpair[1]  )
                                                        elif model.strReelBond[idxreel] == 'harm': 
                                                            pass          
                                                        
                                                        
                                                        # remove spring bond on near bead - NEW
                                                        ssystem.part[ reelpair[0]+reelpair[2] ].delete_bond(( bondReelM[idxreel], reelpair[1] ))


                                                        
                                                        
                                                    if reelLopeBool2 and bondedTypPairs.size > 0:
                                                        # =============================================================================
                                                        #             Change detached bs category back to the original one
                                                        # =============================================================================
                                                        ssystem.part[ bondedTypPairs[ :, 0 ] ].type = model.origTypes[ bondedTypPairs[ :, 0 ] ] # model.inertType
                                                        ssystem.part[ bondedTypPairs[ :, 0 ] ].v = np.zeros((bondedTypPairs.shape[0],3))                            
                        
                        
                        
                                                    # for reelpair in closeTypPairsMove[ lopeIdxMove, : ].tolist(): # move all that can and that have passed the random threshold
                                                    for reelpair in closeTypPairsMove.tolist(): #  move all that can
                                                        ssystem.part[ reelpair[0] ].add_bond(( bondReel[idxreel], reelpair[1] ))

                                                        if model.strReelBond[idxreel] == 'fene':
                                                            ssystem.part[ reelpair[0] ].add_exclusion( reelpair[1]  )
                                                        elif model.strReelBond[idxreel] == 'harm': 
                                                            pass                                            
                                                        
                                                        # add spring bond on near bead
                                                        costmp = ssystem.part[ reelpair[0] +reelpair[2] ].add_bond(( bondReelM[idxreel], reelpair[1] ))
                                
                                                    # stopping particles that just moved ahead
                                                    if len( closeTypPairsMove ) > 0:
                                                        ssystem.part[ closeTypPairsMove[ :, 0 ] ].v = np.zeros(( len( closeTypPairsMove),3))
                                                        ssystem.part[ closeTypPairsMove[ :, 1 ] ].v = np.zeros(( len( closeTypPairsMove),3))
                                                        ssystem.part[ bondedTypPairs[ :, 0 ] ].v = np.zeros(( len( bondedTypPairs),3))
                                                        
                                                        ssystem.part[ closeTypPairsMove[ :, 0 ] + closeTypPairsMove[ :, 2 ] ].v = np.zeros(( len( closeTypPairsMove),3))
                                                        
                                                        # move REF
                                                        reelPairs[idxreel][ reelPairsMask2, 0] = reelPairs[idxreel][ reelPairsMask2, 0] + \
                                                            reelPairs[idxreel][ reelPairsMask2, 2]
                                                            
                                                            
                    
                                                    if (len( closeTypPairsMove ) > 0) and (idxreelpos == 0):
                                                        # from start move to normal move
                                                        reelPairs[idxreel][ reelPairsMask2, 3] = 1                
            
                    
            
                                    ## check if new REF arise                    
                                    random.shuffle(reelly)
                                    newselected = np.empty((0,4), dtype=np.int64)
                                    for idside in reelly:
                                        # remove the pairs that are already REF...
                                        if reelLopeBool1:
                                            idd1 = np.array( list( 
                                                set(model.id1reel[idxreel][idside]) -set( reelPairs[idxreel][:,0]) -set(newselected[:,0])- set( lopePairs[:,0]) )
                                                )
                                        else:
                                            idd1 = np.array( list( set(model.id1reel[idxreel][idside]) - set( reelPairs[idxreel][:,0]) -set(newselected[:,0])) )
                                        
                                        idd2 = np.array( list( set(id2reel[idxreel][idside]) - set( reelPairs[idxreel][:,1]) -set(newselected[:,1])) )
                                    
                                        # it can happen that there's no new particle to attach or any binding site left
                                        if len(idd1) == 0 or len(idd2) == 0: continue
                                    
                                        # 
                                        pos1 = ssystem.part[ idd1.tolist() ].pos
                                        pos2 = ssystem.part[ idd2.tolist() ].pos

                                        #
                                        dmdf = pd.DataFrame( scy.spatial.distance_matrix( pos1, pos2))
                                        dmsubdf = dmdf[ dmdf <= model.reel_cutoff[idxreel] ].dropna(axis=0, how='all').dropna(axis=1, how='all')
                                        dmsub = np.array( dmsubdf )
                                        
                                        # print('search new RF:', dmdf.values, model.reel_cutoff[idxreel])
                                        
                                        if dmsub.size == 0: continue
                                        
                                        closeTypPairsNew = \
                                            np.array( list( set( zip( idd1[ dmsubdf.index], idd2[ dmsubdf.columns[ np.nanargmin( dmsub, 1)] ] ) ) & \
                                                           set( zip( idd1[ dmsubdf.index[ np.nanargmin( dmsub, 0)] ], idd2[ dmsubdf.columns ] ) ) )
                                                     )
                    
                    
                                        reelBirthRand = np.random.random( len(closeTypPairsNew))
                                        reelIdxNew = np.where( reelBirthRand < model.reelBirthProb[idxreel] )[0]#.tolist()
                                        
                                        #
                                        newselected = np.concatenate( ( newselected, 
                                                                        np.int_( np.concatenate( ( 
                                                                            closeTypPairsNew[ reelIdxNew, : ] ,
                                                                            np.ones( (reelIdxNew.size, 1)) * model.dmov[ idside] ,
                                                                            np.ones( (reelIdxNew.size, 1)) * model.reelpos[ idxreel][0]
                                                                            ## np.zeros( (reelIdxNew.size, 1)) # OLDW
                                                                            ), axis=1 ) )
                                                                        ), axis=0 ) # add new bonds
            
        
            
                                    
                                    
                                    # first remove particles killed at the beginning
                                    reelPairs[idxreel] = reelPairs[idxreel][ ~reelIdxDead, :]
                                    
                                    
                                    
                                    #
                                    # add bonds
                                    if ( newselected.ndim > 1) and ( newselected.size > 0): 
                                        for reelpair in newselected.tolist():
                                            costmp = ssystem.part[ reelpair[0] ].add_bond(( bondReel[idxreel], reelpair[1] ))
                                            # ssystem.part[ reelpair[0] ].add_exclusion( reelpair[1]  )
                                            if model.strReelBond[idxreel] == 'fene':
                                                ssystem.part[ reelpair[0] ].add_exclusion( reelpair[1]  )
                                            elif model.strReelBond[idxreel] == 'harm': 
                                                pass                                            
                                            # change particle type
                                            ssystem.part[ reelpair[1] ].type = model.origTypes[ reelpair[1] ] +1 # model.inertType
                                            
                                            
                                            # add spring bond on near bead
                                            costmp = ssystem.part[ reelpair[0] +reelpair[2] ].add_bond(( bondReelM[idxreel], reelpair[1] ))

                                            
                                            
                                        if reelLopeBool2:
                                            # =============================================================================
                                            #             Change Attached bs category to a new one
                                            # =============================================================================
                                            ssystem.part[ newselected[ :, 0 ] ].type = model.boundReelType # model.inertType
                                            ssystem.part[ newselected[ :, 0 ] ].v = np.zeros((newselected.shape[0],3))
                                            
                                                                                            
                                            
                                                                                            
                
                                        # stopping particles that associate
                                        ssystem.part[ newselected[ :, 0 ] ].v = np.zeros((newselected.shape[0],3))        
                                        ssystem.part[ newselected[ :, 1 ] ].v = np.zeros((newselected.shape[0],3))        
                    
                                        ssystem.part[ newselected[ :, 0 ] + newselected[ :, 2 ] ].v = np.zeros((newselected.shape[0],3))        
                    
                    
                                        # add new born LEF
                                        reelPairs[idxreel] = np.concatenate( ( reelPairs[idxreel], 
                                                                     newselected
                                                                        ), axis=0 ) # add new bonds
                   
                    
                    
    
                                    ###========================================================================
                                    # Measure reel props
                                    ###========================================================================      
                                    # reeling
                                    if idxreel == 0:
                                        reeldf = pd.DataFrame( reelPairs[idxreel], columns=['bondid1','bondid2','direction','reelpos'] )
                                    else:
                                        reeldf = reeldf.append(
                                            pd.DataFrame( reelPairs[idxreel], columns=['bondid1','bondid2','direction','reelpos'] )
                                            )

    
    

                            elif dynprocli == 2:

    
                                # =============================================================================
                                # Specific interactions
                                # =============================================================================
    
                                if model.strSpec:
                                    ## check whether to dissociate some HFB
                                    specDeathRand = np.random.random( specPairs.shape[0])
                                    specIdxDead = (specDeathRand < model.specDeathProb )
                
                
                                    for specpair in specPairs[ specIdxDead, : ].tolist():
                                        ssystem.part[ specpair[0] ].delete_bond(( bondSpec, specpair[1] ))
                                        
                                        if model.strSpecBond == 'fene':
                                            ssystem.part[ specpair[0] ].delete_exclusion( specpair[1]  )
                                        elif model.strSpecBond == 'harm':
                                            pass
                                        
                
                                    
        
        
        
                                    ## check if new HFB arise                    
                                    newselected = np.empty((0,2), dtype=np.int64)
                                    
                                    
                                    # exclude HFB that have reached the biggest multiplicity allowed
                                    idd1 = specNHSpring[specNHSpring[:, 1] < model.specMaxHSpring, 0]
                                    idd2 = specNHFBSpring[specNHFBSpring[:, 1] < model.specMaxHFBSpring, 0]
                                
                                    # it can happen that there's no new particle to attach or any binding site left
                                    if len(idd1) == 0 or len(idd2) == 0: pass
                                    else:
                                    
                                        # 
                                        pos1 = ssystem.part[ idd1 ].pos
                                        pos2 = ssystem.part[ idd2.tolist() ].pos
                                        #
                                        dmdf = pd.DataFrame( scy.spatial.distance_matrix( pos1, pos2))
                                        dmsubdf = dmdf[ dmdf <= model.spec_cutoff ].dropna(axis=0, how='all').dropna(axis=1, how='all')
                                        dmsub = np.array( dmsubdf )
                                        #
                                        if dmsub.size > 0: 
                                            closeTypPairsNew = \
                                                np.array( list( set( zip( idd1[ dmsubdf.index], idd2[ dmsubdf.columns[ np.nanargmin( dmsub, 1)] ] ) ) & \
                                                               set( zip( idd1[ dmsubdf.index[ np.nanargmin( dmsub, 0)] ], idd2[ dmsubdf.columns ] ) ) )
                                                         )
                        
                                            closeTypPairsNew = np.array( list(
                                                set( tuple( map( tuple, closeTypPairsNew) ) ) - set( tuple( map( tuple, specPairs) ) )
                                                ) )
                                            
                        
                                            if len(closeTypPairsNew) > 0: 
                                                specBirthRand = np.random.random( len(closeTypPairsNew))
                                                specIdxNew = np.where( specBirthRand < model.specBirthProb )[0]#.tolist()
                                                
                                                closeTypPairsNew = closeTypPairsNew[ specIdxNew, : ]
                                                #
                                                specNHSpring[ np.isin( specNHSpring[:,0], closeTypPairsNew[:,0] ), 1] += 1
                                                specNHFBSpring[ np.isin( specNHFBSpring[:,0], closeTypPairsNew[:,1] ), 1] +=1
                                               
                                                #
                                                newselected = closeTypPairsNew
        
        
            
                                    
                                    
                                    # first remove particles killed at the beginning
                                    specNHSpring[ np.isin( specNHSpring[:,0], specPairs[ specIdxDead, 0] ), 1] -= 1
                                    specNHFBSpring[ np.isin( specNHFBSpring[:,0], specPairs[ specIdxDead, 1] ), 1] -=1
                                    specPairs = specPairs[ ~specIdxDead, :]
                                    
                                    
                                    
                                    #
                                    # add bonds
                                    if ( newselected.ndim > 1) and ( newselected.size > 0): 
                                        for specpair in newselected.tolist():
                                            costmp = ssystem.part[ specpair[0] ].add_bond(( bondSpec, specpair[1] ))
                                            
                                            if model.strSpecBond == 'fene':
                                                ssystem.part[ specpair[0] ].add_exclusion( specpair[1]  )
                                            elif model.strSpecBond == 'harm':
                                                pass                                        
                                            
                                            # change particle type
                                            ssystem.part[ specpair[1] ].type = model.origTypes[ specpair[1] ] +1 # model.inertType
                                            
                
                                        # stopping particles that associate
                                        ssystem.part[ newselected[ :, 0 ] ].v = np.zeros((newselected.shape[0],3))        
                                        ssystem.part[ newselected[ :, 1 ] ].v = np.zeros((newselected.shape[0],3))        
                    
                                        # add new born LEF
                                        specPairs = np.concatenate( ( specPairs, 
                                                                     newselected
                                                                        ), axis=0 ) # add new bonds
    
                    
                                    # back to original type for those with no bonds
                                    specNobond = specNHFBSpring[specNHFBSpring[:, 1] == 0, 0]
                                    if specNobond.size > 0:
                                        ssystem.part[ specNobond ].type = model.origTypes[ specNobond ]
    
    
    

                                    ###========================================================================
                                    # Measure specific interactions
                                    ###========================================================================      
                                    specdf = pd.DataFrame( specPairs, columns=['bondid1','bondid2'] )
  


        
    
    
    
    
    


                    

                        ###========================================================================
                        # Measure extrusion props
                        ###========================================================================      
                        if lopdf.shape[0] == 0:
                            lopdf = pd.DataFrame(data={
                                'bondid1': [np.nan],
                                'bondid2': [np.nan],
                                'direction': [np.nan],
                                'time': ssystem.time})
                        else: lopdf['time'] = ssystem.time
                        
                        
                        lopdf['conf'] = model.procid
                        
                        lopdfall = lopdfall.append(lopdf)    
        
        
        
                        ###========================================================================
                        # Measure reeling props
                        ###========================================================================      
                        try:
                            if reeldf.shape[0] == 0:
                                reeldf = pd.DataFrame(data={
                                    'bondid1': [np.nan],
                                    'bondid2': [np.nan],
                                    'direction': [np.nan],
                                    # 'bondMode': [''],
                                    'reelpos': [np.nan],
                                    'time': ssystem.time})
                            else: reeldf['time'] = ssystem.time

                            reeldf['conf'] = model.procid
                            
                            reeldfall = reeldfall.append(reeldf)              

                        except:
                            pass
                        
                        
                        ###========================================================================
                        # Measure specific bonds
                        ###========================================================================        
                        if model.strSpec:
                            if specdf.shape[0] == 0:
                                specdf = pd.DataFrame(data={
                                    'bondid1': [np.nan],
                                    'bondid2': [np.nan],
                                    'time': ssystem.time})
                            else: specdf['time'] = ssystem.time

                            specdf['conf'] = model.procid
                            
                            specdfall = specdfall.append(specdf)         
    
    
    
                else:
                    ssystem.integrator.run(smpl)
                    lopedtMod = lopedtMod + smpl


                # save lope activity
                lopdfall = func.obsLopeSave2(lopdfall, model, env, 'Lope') 
                reeldfall = func.obsLopeSave2(reeldfall, model, env, 'Reel') 
                specdfall = func.obsLopeSave2(specdfall, model, env, 'Spec') 

                
            else:
                ssystem.integrator.run(smpl)                
                
                


    
    
    
            ###========================================================================
            # Measure
            ###========================================================================                 
            obss_all = func.obsInterm( ssystem, model, obss_all, vt['name'])
            
            
        

    
    print('\n---> Simulation done')
    
    
    
    # =============================================================================
    #         Finalize the observables
    # =============================================================================
    if lopdfall.shape[0] > 0:
        model.startDynTime = 0
        lopdfall = func.obsLopeSave2(lopdfall, model, env, 'Lope', When='end') 

    if reeldfall.shape[0] > 0:
        model.startDynTime = 0
        reeldfall = func.obsLopeSave2(reeldfall, model, env, 'Reel', When='end')         

    if specdfall.shape[0] > 0:
        model.startDynTime = 0
        specdfall = func.obsLopeSave2(specdfall, model, env, 'Spec', When='end')         

            

    
    
        
    

except Exception as e:
    print('*** Exception encountered:')
    print( e)
    # func.PrintException()
    print('\nSimulation failed! Saving final snapshot and exit')

    ###========================================================================
    # Measure and save what you can
    ###========================================================================                 
    obss_all = func.obsInterm( ssystem, model, obss_all, vt['name'], exept = True)


    







# =============================================================================
# =============================================================================
# # Saving simulation    
# =============================================================================
# =============================================================================


# =============================================================================
# Save Measured data
# =============================================================================
# conf ID
obss_all['conf'] = model.procid
obss_all['param']= model.str_param_file2

# time
endTime = time.time()
obss_all['cputime'] = (endTime - model.startTime)/ model.nMPI
#
func.obsEnd(obss_all, model, env)  
        

# =============================================================================
# Save configuration
# =============================================================================
func.confEnd( ssystem, model, espressomd.io.writer, fp, dft, precision=model.savePrecision )       

        

