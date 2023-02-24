#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 17:27:44 2021

@author: mariano
"""

import random
import numpy as np
import scipy as scy
from scipy.spatial import distance_matrix
import scipy.stats
import scipy.signal as scygnal
import pandas as pd
import re
import os, sys
import importlib
import itertools
import pdb
import time 


import linecache


#### import simulation settings
import esprenv 
import esprenv as env
from esprenv import *

if hasattr(__builtins__, '__IPYTHON__'):
    importlib.reload( esprenv )
    from esprenv import *



    
    

# set some warnings
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
pd.options.mode.chained_assignment = None  # default='warn'


def argvRead( Argv):
    # check if user edits some params
    argl = Argv[1:]
    argvtup = [ (idx,cosi)  for idx, cosi in enumerate(argl) if re.search( '-', cosi)]
    
    diddi = {}
    for argi in argvtup:
        if argi[1] == '-mint':
            diddi.update({'mint': np.int_(argl[ argi[0]+1] )})
        elif argi[1] == '-maxt':
            diddi.update({'maxt': np.int_(argl[ argi[0]+1] )})
        elif argi[1] == '-tit':
            diddi.update({'tit': np.int_(argl[ argi[0]+1] )})
        elif argi[1] == '-procid':
            diddi.update({'procid': argl[ argi[0]+1] })
        elif argi[1] == '-sysname':
            diddi.update({'str_sys': argl[ argi[0]+1] })
        elif argi[1] == '-tsam':
            diddi.update({'str_tsam': argl[ argi[0]+1] })
        elif argi[1] == '-model':
            diddi.update({'str_syst': argl[ argi[0]+1] })            
        elif argi[1] == '-param':
            diddi.update({'str_param': argl[ argi[0]+1] })            
        elif argi[1] == '-system':
            diddi.update({'strSetFile': argl[ argi[0]+1] })            
        elif argi[1] == '-mode':
            diddi.update({'str_mode': argl[ argi[0]+1] })            
        elif argi[1] == '-nMPI':
            diddi.update({'nMPI': np.int_(argl[ argi[0]+1] ) })   
            print('MPI run. no. of nodes:' , diddi['nMPI'])
        elif argi[1] == '-maxt2':
            diddi.update({'maxt2': np.int_(argl[ argi[0]+1] )})
        elif argi[1] == '-dyn1':
            diddi.update({'dyn1': argl[ argi[0]+1] })
        elif argi[1] == '-dyn2':
            diddi.update({'dyn2': argl[ argi[0]+1] })
        elif argi[1] == '-modelName':
            diddi.update({'modelName': argl[ argi[0]+1] })    
        elif argi[1] == '-SAopt':
            diddi.update({'SAopt': argl[ argi[0]+1] })     
        elif argi[1] == '-scaleback':   
            diddi.update({'scaleback': True })                 
        elif argi[1] == '-svfile':
            diddi.update({'svfile': argl[ argi[0]+1].split(',') }) 
        elif argi[1] == '-therm':
            # diddi.update({'strTherm': [argl[ argi[0]+1]] })   
            diddi.update({'strTherm': argl[ argi[0]+1].split(',') })   
        elif argi[1] == '-c':
            diddi.update({'strC': argl[ argi[0]+1] })
        elif argi[1] == '-dopt':
            diddi.update({'otheropts': argl[ argi[0]+1] })
        elif argi[1] == '-reg':
            diddi.update({'strReg': argl[ argi[0]+1] })
        elif argi[1] == '-bscal':
            diddi.update({'bscal': np.float32(argl[ argi[0]+1] ) })   
        else:
            print('option',argi[1],'not recognized')

    return diddi





def argvRead2( Argv, localKeys):
    # check if user edits some params
    argl = Argv[1:]
    argvtup = [ (idx, re.sub('-','', cosi) )  for idx, cosi in enumerate(argl) if re.search( '-', cosi)]
    
    diddi = {}
    for argi in argvtup:
        diddi.update({ argi[1]: argl[ argi[0]+1] })
        if argi[1] not in localKeys:
            print( 'updated variable ', argi[1])
        else:
            print( 'created variable ', argi[1], 'no previously assigned')

    return diddi












# =============================================================================
# Check if user edits some params
# =============================================================================
locals().update( argvRead( sys.argv ) )







# =============================================================================
# Block 2 of functions
# =============================================================================

def polGenEMD( polNumb, beads_per_chain, bond_length, start_positions ):
    
    polymers = polymer.positions(n_polymers= polNumb,
                                 beads_per_chain=beads_per_chain,
                                 bond_length=bond_length, seed=round(random.random() * 1e+8) ,
                                 min_distance = bond_length, 
                                 start_positions = start_positions ,
                                 respect_constraints = True
                                 )    
    
    


def polGen( beads_per_chain, bond_length, cosDic ):

    startPos = cosDic['startPos']
    b = cosDic['boxl']
    polNumb = 1

    poll = np.ones(( polNumb, beads_per_chain, 3 ) ) * startPos # np.array([b/2.,b/2.,b/2.])
    ii = 1
    dpos = np.array([0, 0, bond_length])
    while ii < beads_per_chain:
        curPos = poll[0,ii-1,:]
        for iii in range( 6 * 5) :
            if ( np.linalg.norm( curPos + dpos - b/2. , axis=0 ) > b/2. -bond_length ) or ( np.any( np.linalg.norm( curPos + dpos - poll, axis=2) < bond_length ) ): 
                dang = np.random.random(2) * 2 * np.pi
                dpos = np.array([
                    np.sin( dang[0]) * np.cos( dang[1] ) ,
                    np.sin( dang[0]) * np.sin( dang[1] ) ,
                    np.cos( dang[0] )
                    ]) * bond_length
            else:
                poll[0,ii,:] = curPos + dpos
                ii = ii +1
                break
        
        if iii == 6 * 5 -1:
            # pdb.set_trace()
            ii = ii -1
            poll[0,ii,:] = np.ones((3)) * startPos
            # print( ii+1,' -> ', ii)
    
    # Check Sphere Constraint
    cond1 = np.all( np.linalg.norm( poll - b/2., axis=2) <= b/2. -bond_length )
    # Check polymer Constraint
    cond2 = np.allclose( np.linalg.norm( poll[0,1:,:] - poll[0,:-1,:], axis=1), bond_length)
    # Check polymer-polymer minima distance is respected
    dm = scy.spatial.distance_matrix( poll[0,:], poll[0,:])
    dm [ np.diag_indices( dm.shape[0]) ] = bond_length
    cond3 = not np.any( dm < bond_length )
    
    
    if cond1 and cond2 and cond3:
        print('Polymer generation successful!')
        return poll
    else :
        print('Error generating polymer: returning False')
        return False






def polGen2( bpcv, bond_length, cosDic ):

    b = cosDic['boxl']

    if type( bpcv) is not list:
        bpcv = [ bpcv]

    chkStaPos = (('startPosi' in cosDic.keys()) and (cosDic['startPosi'].shape[0] == len(bpcv)))
    if chkStaPos:
        pollnp = cosDic['startPosi']

    polly = []
    itertimes = 6 * 5 # coordination number * attempts
    for poli, bpc in enumerate(bpcv):

        ii = 1
        while ii < bpc:
            
            if ii == 1 and poli == 0:
                if not chkStaPos:
                    curPos = np.random.random((1,3)) * b
                    while np.linalg.norm( curPos  - b/2. , axis=1 ) > b/2. -bond_length : 
                        curPos = np.random.random((1,3)) * b
                    
                    poll = np.copy(curPos)
                    pollnp = np.copy(poll)    
                else:
                    poll = cosDic['startPosi'][ None, 0, :]
                    curPos = cosDic['startPosi'][ None, 0, :]
                
            elif ii == 1:
                if not chkStaPos:
                    curPos = np.random.random((1,3)) * b
                    while ( np.linalg.norm( curPos  - b/2. , axis=1 ) > b/2. -bond_length ) or ( np.any( np.linalg.norm( curPos - pollnp, axis=2) < bond_length ) ): 
                        curPos = np.random.random((1,3)) * b
                    
                    poll = np.copy(curPos)
                    pollnp = np.concatenate( (pollnp, poll), 0)
                else:
                    poll = cosDic['startPosi'][ None, poli, :]                    
                    curPos = cosDic['startPosi'][ None, poli, :]                    
                
            else:
                curPos = poll[ None,ii-1,:]
                
            # random direction
            dangThe = np.random.random() * 2 * np.pi
            dangPh = np.random.random() * np.pi
            dpos = np.array([
                np.sin( dangThe) * np.cos( dangPh ) ,
                np.sin( dangThe) * np.sin( dangPh ) ,
                np.cos( dangThe )
                ]) * bond_length
                
            
            for iii in range( itertimes) :
                if ( np.linalg.norm( curPos + dpos - b/2. , axis=1 ) > b/2. -bond_length ) or ( np.any( np.linalg.norm( curPos + dpos - pollnp, axis=1) < bond_length ) ): 
                    dangThe = np.random.random() * 2 * np.pi
                    dangPh = np.random.random() * np.pi
                    dpos = np.array([
                        np.sin( dangThe) * np.cos( dangPh ) ,
                        np.sin( dangThe) * np.sin( dangPh ) ,
                        np.cos( dangThe )
                        ]) * bond_length
                else:
                    pollnp = np.concatenate( ( pollnp, curPos + dpos), 0)
                    poll = np.concatenate( ( poll, curPos + dpos), 0)
                    ii = ii +1
                    break
            
            if iii == itertimes -1:
                # pdb.set_trace()
                ii = ii -1
                pollnp = np.delete( pollnp, -1, 0)
                poll = np.delete( poll, -1, 0)
                # print( ii+1,' -> ', ii)
        
        # Check Sphere Constraint
        cond1 = np.all( np.linalg.norm( poll - b/2., axis=1) <= b/2. -bond_length )
        # Check polymer Constraint
        cond2 = np.allclose( np.linalg.norm( poll[ 1:,:] - poll[ :-1,:], axis=1), bond_length)
        # Check polymer-polymer minimal distance is respected
        dm = scy.spatial.distance_matrix( poll, poll)
        dm [ np.diag_indices( dm.shape[0]) ] = bond_length
        cond3 = not np.any( dm < bond_length )
    
        polly += [poll]
    
    
        if cond1 and cond2 and cond3:
            print('Polymer',poli+1,'/',len(bpcv),'generation successful!')
        else :
            print('Error generating polymer: returning False')
            return False

    return polly








def calcUnits( molC, N, chrend, chrstart, bond_length, eta=.05)   :
    # pdb.set_trace()
    # =============================================================================
    # Unit of measure
    # =============================================================================
    try:
        ll = chrend - chrstart
        res = ll  / N
    except:
        ll = [chrend[i] - chrstart[i] for i in range(len(chrend))]
        res = ll[0] / N[0]
        N = np.array( N ).sum()
        ll = np.array( ll ).sum()
    
    # 
    bioD = {'genome length': 
                {'hg19': 3.1 * 2 * 10**9  , 
                'mm9': 6.5 * 10**9 } , 
            'nucleus radius': # micron
                {'HeLa 1': (3 * 690 / (4 * np.pi) )**(1/3.)  , 
                'HeLa 2':  (3 * 374 / (4 * np.pi) )**(1/3.)  ,
                'Fibroblast':  (3 * 500 / (4 * np.pi) )**(1/3.)  ,
                'mESC Nicodemi': 3.5 / 2 ,  # this to be checked
                'mouse L cell': (3 * 435 / (4 * np.pi) )**(1/3.) } 
            }
    
    G = bioD['genome length']['hg19'] 
    R = bioD['nucleus radius']['Fibroblast'] * 10**(-6)
    
    # box size
    b = 2 * R * ( ll / G ) **(1/3.)  # meters
    
    # linearly depends on the estimation of nucleus radius!!!
    sig = ( res / ll ) ** ( 1/ 3.) * b  # meters
    # sig = ( res / G ) ** ( 1/ 3.) * R  # meters
    dunit = sig / bond_length
    
    r = b / 2
    
    # simul box size
    bsimsig = np.int_( b * 2 / dunit )
    rsimsig = np.int_( r * 2 / dunit )
    bsim = b * 2
    rsim = r * 2
    
    # time units
    nu = 10**(-2) # Pascal * sec [.1 P]
    KbT = 4.11 * 10 ** (-21) # T= 298K [Joule]
    gamm = 3 * np.pi * nu * sig
    D = KbT / gamm
    
    # LJ time 
    mkdalt = 1000 * 1.66033 * 10**-27 
    tLJ = sig * np.sqrt(  50 * mkdalt / KbT)
    
    
    # standard MD
    # eta = .025 # .1 ?
    tau = eta * ( 6 * np.pi * sig**3 / KbT)
    
    # Browninan time scale
    tB = 3 * np.pi * sig ** 3 * nu / KbT # sig ** 2 / D [sec]
    # dramatically depends on the correct estimation of sigma!! 
    # 3 * np.pi * (3*10**-8) ** 3 * nu / KbT # sig ** 2 / D [sec]
    
    nts = 10**7 # number of simulation time steps
    ts = 10**-2 # simulation time step
    tT = nts * tB * ts # mapped time duration of simulation [sec]
    
    
    # map binder concentration
    Na = 6.022 * 10**23
    molPart = []
    # pdb.set_trace()
    for CnA in molC:
        # CnA = 10 # n mol / liter = 10**-9 mol / 10**-3 cubic meters
        P = int(np.round( CnA * 10**-9 * Na * ( 4/ 3. * np.pi * rsim**3 * 10**3 ) ))
        Ptot = int(np.round( CnA * 10**-9 * Na * ( 4/ 3. * np.pi * R**3 * 10**3 ) ))
    
        # given number of particles map concentration
        c = ( 1100 * 3 / ( 4 * np.pi) ) * ( rsim )**(-3) / Na # mol / m
        c2 = c * 10 ** (9) / 10** (3) # nano mol / liter    
    
        molPart += [P]
    
    # add num of pol beads as last element
    # molPart += [N]
    
    return molPart, bsimsig, tB, sig, tau, tLJ














def readClasses( model ):

    # model.resBestFN = '/SA/resltBest_'+.regg+'_N'+str(N)+'_M'+str(M)+'.csv'
    
    ##
    try:
        chrL = list(zip( model.chrname, model.chrstart, model.chrend))
    except:
        chrL = [ (model.chrname, model.chrstart, model.chrend) ]
        
    ##
    polid, npol, taillen = 0, [], []
    binning, bsmap, bestpd = pd.DataFrame([]), pd.DataFrame([]), pd.Series([])
    for chrid, (chrname, chrstart, chrend) in enumerate(chrL):
        # save/read best corr matrix
        # ttcpd = pd.read_csv( model.resBestFN[chrid]
        #                        , sep = ',', index_col=False )
        
        # model.N = model.ttcpd.N[0]
        # model.M = model.ttcpd.M[0]
        # model.R = np.copy(model.M)
        # model.NR = model.N * model.R
        # pdb.set_trace()
        # select one ttc
        # model.besttmp = model.ttcpd.iloc[ model.bestIndex[chrid] ]
        besttmp = model.besttmp[chrid]
        besttmp['polyid'] = polid
        bestpd = bestpd.append( besttmp)
        
        # ttcpd2 = ttcpd[ [ str(ni) for ni in range(0, model.NR) ] ]
        # (ttcpd2 != 0).sum(1).min(), (ttcpd2 != 0).sum(1).max()
        
        ttcsel = besttmp[ [ str(ni) for ni in range(0, model.NR) ] ].values.flatten()
        
        # Entropy 2
        # uni2 = np.unique( ttcsel.tolist()+list(range(0, model.M)), return_counts=True)
        # ncou2 = uni2[1]-1
        # nint = ncou2[1:].sum()

        binntmp = pd.DataFrame( data={
            'cumpos': list(range(1,model.NR + 1)) ,
            'type' : ttcsel + model.M-1 ,
            'part' : np.arange(chrstart, chrend, (chrend-chrstart)/model.NR)[: model.NR] ,
            'chromStart' : np.arange(chrstart, chrend, (chrend-chrstart)/model.NR)[: model.NR]
            })
        
        taildf = pd.DataFrame(
            np.zeros( ( 1,  binntmp.shape[1])) * np.nan ,
            columns = binntmp.columns
            )
        
        taildf['type'] = 2 * model.M -1
        taildf['part'] = - 2 * (polid+1) 
        binntmp = binntmp.append( taildf)

        taildf['part'] = - 2 * (polid+1) +1
        binntmp = taildf.append( binntmp )
        
        
        # map of bs        
        binntmp['Index'] = binntmp.reset_index().index
        binntmp['dclass']= binntmp.shift(-1)[ 'type' ] - binntmp[ 'type' ]
        bsmaptmp = binntmp[ binntmp['dclass'] != 0]        
        bsmaptmp['bsbatch'] = bsmaptmp['Index'] - bsmaptmp.shift(+1)['Index']
        
        
        # estimate tails of polymer
        taillentmp = np.int_( np.round( model.NR ** (1/2) ) )
        
        npoltmp = model.NR
        
        #
        bsmaptmp['bsbatch'].iloc[0] = taillentmp
        bsmaptmp['bsbatch'].iloc[-1] = taillentmp


        ###
        binntmp['polyid'] = polid
        binning = binning.append( binntmp)
        
        ###
        bsmaptmp['polyid'] = polid
        bsmaptmp['bsbatch'] = bsmaptmp['bsbatch'].astype(int)
        bsmaptmp['monocumsum'] = bsmaptmp['bsbatch'].cumsum()
        bsmap = bsmap.append( bsmaptmp)


        
        ###        
        polid = polid + 1
        taillen += [taillentmp]
        npol += [npoltmp]



    
    return npol, taillen, binning, bsmap, bestpd









def str2list( stri):
    # print( stri[1:-1])
    # pdb.set_trace()
    coll = [ int(ele) for ele in stri[0].split() ]
    return coll









def genArtificial( model, strpart= 'natural' ):
    ## process .bed artificial sysstem file
    header = ['chrom', 'chromStart', 'chromEnd', 'name', 'annot', 'strand', 'score' ]
    chrArt = pd.read_excel( model.artifSys[0] , engine="odf", sheet_name= model.artifSys[1], index_col=False, usecols=header)
    chrArt.columns = header[:len(chrArt.columns)]
    chromArt_map = pd.read_excel( env.strHome + '/' + model.modelods + '_model.ods', engine="odf", sheet_name= model.modelods_modelsheet, index_col=False)
    
    chromArt2 = chrArt.merge( chromArt_map, how='left', left_on='name', right_on='name' )
   
    chromArt2['dchrbp'] = chromArt2['chromEnd'] - chromArt2['chromStart']

    
    parampd = pd.read_excel( model.artifSys[0] , engine="odf", sheet_name= 'param' , index_col=False)
    model.bpres = parampd[parampd.ssystem == model.artifSys[1] ].bpres.values[0]
    taillentmp = parampd[parampd.ssystem == model.artifSys[1] ].taillength.values[0]

    chromArt2['chromStart'] = chromArt2['chromStart'] * model.bpres
    chromArt2['chromEnd'] = chromArt2['chromEnd'] * model.bpres
    chrNamAnnot = chromArt2[['name','annot']].drop_duplicates()
    chrNamAnnot['nameAnnot'] = chrNamAnnot['name'] + '+' + chrNamAnnot['annot']
    chrNamAnnot['class'] = chrNamAnnot.index

    chromArt2 = chromArt2.merge( chrNamAnnot, on=['name','annot'], how='left')


    ##
    chrnames = chromArt2.chrom.unique()
    chrL = []
    for chrnamei in chrnames:
        chrL += [[
            chrnamei ,
            chromArt2[ chromArt2.chrom == chrnamei ].chromStart.iloc[0] ,
            chromArt2[ chromArt2.chrom == chrnamei ].chromEnd.iloc[-1]
            ]]
    

    ##
    polid, npol, taillen = 0, [], []
    binning, bsmap = pd.DataFrame([]), pd.DataFrame([])
    for chrname, chrstart, chrend in chrL:
        print( 'Polymer: ', chrname, chrstart, chrend)

        # this selects the region 
        chrsub = chromArt2[ (chromArt2['chrom']== chrname) & ( chromArt2['chromStart'] >= chrstart) & ( chromArt2['chromEnd'] <= chrend) ]
        # chrsub['chromCenter'] = ((chrsub['chromEnd']+chrsub['chromStart'])/2.).astype(int)

        
        
        # create partition
        if strpart == 'natural':
            part = np.arange( chrsub['chromStart'].values[0], chrsub['chromEnd'].values[-1], model.bpres )
        elif strpart == 'hicpart':
            # map into the hic matrix
            part = np.int_( np.arange( 
                np.round( chrsub['chromStart'].values[0] / model.bpres ) * model.bpres , 
                np.round( chrsub['chromEnd'].values[-1] / model.bpres ) * model.bpres +1, 
                model.bpres
                ) )
        elif strpart == 'given':
            part = partgiven
            
        # pdb.set_trace()
        partdf = pd.DataFrame( part, columns=['part'])
        # partdf['partCenter'] = ((partdf + partdf.shift(-1))/2.)
        partdf.dropna(how='any', axis=0, inplace=True)
        # partdf.partCenter = partdf.partCenter.astype(int)
        # partdf['partCenter'] = ((partdf['chromEnd']+partdf['chromStart'])/2.).astype(int)
        
        # merge as of
        # pdb.set_trace()
        part2 = pd.merge_asof( chrsub[['chromStart','name', 'annot', 'strand', 'class' , model.modelname]], partdf, 
                              left_on = 'chromStart', right_on ='part' ,  
                              tolerance= model.bpres, direction='nearest' )
        
        part2['partn'] = (part2['part'] / model.bpres) # .astype( np.int32)
        part2[ 'class' ] = part2[ 'class' ].astype('int')
        
        # remove classes that are contiguous and identical (this can happen because we remove intervals below threshold bpmin )
        part2['dclass']= part2.shift(-1)[ 'class' ] - part2[ 'class' ]
        part4 = part2[ part2['dclass'] != 0]        
        
        
        # calculate how many beads are in each interval
        part4['dpn'] = part4.shift(-1)['partn'] - part4['partn']
        part4['dpn2'] = part4['dpn']
        
        # deal with last interval
        part4['dpn2'].iloc[-1] = np.round( (chrsub['chromEnd'].values[-1] - part4['chromStart'].values[-1]) / model.bpres )
        part4['dpn'].iloc[-1] = part4['dpn2'].iloc[-1]
        
        # assign 1 bead to all intervals smaller than bpres
        part4.replace({'dpn2':{0:1}}, inplace=True)
        part4['dpn2'] = part4['dpn2'].astype('int')
        
        
        # calculate how many beads are there
        part4['cumpos'] = part4['dpn2'].cumsum() # .shift(+1, fill_value=0) # part4.partn2.values - part4 ['partn2'].iloc[0]
        bslen = part4['dpn2'].sum()
        # estimate tails of polymer
        # taillentmp = np.int_( np.round( bslen ** (1/2) ) )
        npoltmp = 2 * taillentmp + bslen
        
        
        # tails
        taildf = pd.DataFrame(
            np.zeros( ( 1,  part4.shape[1])) * np.nan ,
            columns = part4.columns
            )
        
        taildf[ model.modelname ] = int( model.tailtyp )
        taildf['dpn2'] = taillentmp
        taildf['dpn'] = taillentmp
        
        part4 = taildf.append( part4.append( taildf))
        
        
        
        
        
        
        # create binning from partition
        binntmp = partdf.merge( part4, how= 'left', on='part')
        binntmp[ 'class' ] = binntmp[ 'class' ].fillna( method='ffill')
        binntmp['type'] = binntmp[ model.modelname ]

        if binntmp.dpn2.sum() != binntmp.shape[0]:
            print('This is not good! binning.dpn2.sum() != binning.shape[0]')
            raise 
            
        taildf = pd.DataFrame(
            np.zeros( ( 1,  binntmp.shape[1])) * np.nan ,
            columns = binntmp.columns
            )
        # pdb.set_trace()
        taildf['type'] = int( model.tailtyp )
        taildf['part'] = - 2 * (polid+1) 
        binntmp = binntmp.append( taildf)

        taildf['part'] = - 2 * (polid+1) +1
        binntmp = taildf.append( binntmp )
            
            
        
        
        # map of bs
        bsmaptmp = part4[[ model.modelname ,'dpn2','chromStart']]
        bsmaptmp.columns = ['type','bsbatch','chromStart']
        
        
        
        ###
        binntmp['polyid'] = polid
        binning = binning.append( binntmp)
        
        ###
        bsmaptmp['polyid'] = polid
        bsmaptmp['monocumsum'] = bsmaptmp['bsbatch'].cumsum()
        bsmap = bsmap.append( bsmaptmp)

        ###        
        polid = polid + 1
        taillen += [taillentmp]
        npol += [npoltmp]
        
        
        ### genome
        genome = [ int(bsmap.chromStart.min())]
        genome += [ int(bsmap.chromStart.max() + model.bpres)]
        genome += [ model.bpres]
    
    
    
    return npol, taillen, binning, bsmap, genome







def genReal( model, strpart= 'natural' ):
    ## process .bed artificial sysstem file
    header = ['chrom', 'chromStart', 'chromEnd', 'name', 'annot', 'strand', 'score', 'chromStartBp', 'chromEndBp','chrMiddl_peak' ]
    chrArt = pd.read_excel( model.artifSys[0] , engine="odf", sheet_name= model.artifSys[1], index_col=False, usecols=header)
    chrArt.columns = header[:len(chrArt.columns)]
    chromArt_map = pd.read_excel( env.strHome + '/' + model.modelods + '_model.ods', engine="odf", sheet_name= model.modelods_modelsheet, index_col=False)
    
    chromArt2 = chrArt.merge( chromArt_map, how='left', left_on='name', right_on='name' )
   
    chromArt2['dchrbp'] = chromArt2['chromEnd'] - chromArt2['chromStart']
    
    
    parampd = pd.read_excel( model.artifSys[0] , engine="odf", sheet_name= 'param' , index_col=False)
    model.bpres = parampd[parampd.ssystem == model.artifSys[1] ].bpres.values[0]
    taillentmp = parampd[parampd.ssystem == model.artifSys[1] ].taillength.values[0]
            
    chromArt2['chromStart'] = chromArt2['chromStart'] * model.bpres
    chromArt2['chromEnd'] = chromArt2['chromEnd'] * model.bpres
    chrNamAnnot = chromArt2[['name','annot']].drop_duplicates()
    chrNamAnnot['nameAnnot'] = chrNamAnnot['name'] + '+' + chrNamAnnot['annot']
    chrNamAnnot['class'] = chrNamAnnot.index

    chromArt2 = chromArt2.merge( chrNamAnnot, on=['name','annot'], how='left')


    ##
    chrnames = chromArt2.chrom.unique()
    chrL = []
    for chrnamei in chrnames:
        chrL += [[
            chrnamei ,
            chromArt2[ chromArt2.chrom == chrnamei ].chromStart.iloc[0] ,
            chromArt2[ chromArt2.chrom == chrnamei ].chromEnd.iloc[-1]
            ]]
    

    ##
    polid, npol, taillen = 0, [], []
    binning, bsmap, binning2 = pd.DataFrame([]), pd.DataFrame([]), pd.DataFrame([])
    for chrname, chrstart, chrend in chrL:
        print( 'Polymer: ', chrname, chrstart, chrend)

        # this selects the region 
        chrsub = chromArt2[ (chromArt2['chrom']== chrname) & ( chromArt2['chromStart'] >= chrstart) & ( chromArt2['chromEnd'] <= chrend) ]
        # chrsub['chromCenter'] = ((chrsub['chromEnd']+chrsub['chromStart'])/2.).astype(int)

        
        
        # create partition
        if strpart == 'natural':
            part = np.arange( chrsub['chromStart'].values[0], chrsub['chromEnd'].values[-1], model.bpres )
        elif strpart == 'hicpart':
            # map into the hic matrix
            part = np.int_( np.arange( 
                np.round( chrsub['chromStart'].values[0] / model.bpres ) * model.bpres , 
                np.round( chrsub['chromEnd'].values[-1] / model.bpres ) * model.bpres +1, 
                model.bpres
                ) )
        elif strpart == 'given':
            part = partgiven
            
        # pdb.set_trace()
        partdf = pd.DataFrame( data={
            'part' : part})
        # partdf['partCenter'] = ((partdf + partdf.shift(-1))/2.)
        partdf.dropna(how='any', axis=0, inplace=True)
        # partdf.partCenter = partdf.partCenter.astype(int)
        # partdf['partCenter'] = ((partdf['chromEnd']+partdf['chromStart'])/2.).astype(int)
        
        # remove first and last batch of bins
        # because artificially placed inert for simulatory reasons
        binrmvfirst, binrmvlast = chrsub['dchrbp'].iloc[0], chrsub['dchrbp'].iloc[-1]
        
        # merge as of
        # pdb.set_trace()
        part2 = pd.merge_asof( chrsub[['chromStart','name', 'annot', 'strand', 'class', 'chromStartBp','chromEndBp','chrMiddl_peak', model.modelname]], 
                              partdf, 
                              left_on = 'chromStart', right_on ='part' ,  
                              tolerance= model.bpres, direction='nearest' )
        
        part2['partn'] = (part2['part'] / model.bpres) # .astype( np.int32)
        part2[ 'class' ] = part2[ 'class' ].astype('int')
        
        # remove classes that are contiguous and identical (this can happen because we remove intervals below threshold bpmin )
        # part2['dclass']= part2[ 'class' ] - part2[ 'class' ].shift(1)
        # part4 = part2[ part2['dclass'] != 0]        
        part4 = part2.copy()
        
        # calculate how many beads are in each interval
        part4['dpn'] = part4.shift(-1)['partn'] - part4['partn']
        part4['dpn2'] = part4['dpn']
        
        # deal with last interval
        part4['dpn2'].iloc[-1] = np.round( (chrsub['chromEnd'].values[-1] - part4['chromStart'].values[-1]) / model.bpres )
        part4['dpn'].iloc[-1] = part4['dpn2'].iloc[-1]
        
        # assign 1 bead to all intervals smaller than bpres
        part4.replace({'dpn2':{0:1}}, inplace=True)
        part4['dpn2'] = part4['dpn2'].astype('int')
        
        
        # calculate how many beads are there
        part4['cumpos'] = part4['dpn2'].cumsum() # .shift(+1, fill_value=0) # part4.partn2.values - part4 ['partn2'].iloc[0]
        bslen = part4['dpn2'].sum()
        # estimate tails of polymer
        # taillentmp = np.int_( np.round( bslen ** (1/2) ) )
        npoltmp = 2 * taillentmp + bslen
        
        
        # tails
        taildf = pd.DataFrame(
            np.zeros( ( 1,  part4.shape[1])) * np.nan ,
            columns = part4.columns
            )
        
        taildf[ model.modelname ] = int( model.tailtyp )
        taildf['dpn2'] = taillentmp
        taildf['dpn'] = taillentmp
        
        part4 = taildf.append( part4.append( taildf))
        
        
        
        
        
        # =============================================================================
        #         # create binning from partition
        # =============================================================================
        binntmp = partdf.merge( part4, how= 'left', on='part')

        
        binntmp[ 'Dbp' ] = 0
        binntmp[ 'Dbp' ] = binntmp[ 'Dbp' ].where( ~np.isnan(binntmp[ 'chromStartBp' ]), model.bpres)
        binntmp[ 'Dbp2' ] = (binntmp['chromStartBp'].fillna( method='ffill') - binntmp['chromStartBp'].fillna( method='ffill').shift(1)).fillna(value=0)
        binntmp[ 'Dbp2' ] = binntmp[ 'Dbp2' ].where( binntmp[ 'Dbp2' ]==0, model.bpres)
        
        # pdb.set_trace()
        binntmp[ 'chromStartBp2' ] = ((binntmp[ 'Dbp' ] + binntmp[ 'Dbp2' ]).cumsum() + binntmp[ 'chromStartBp' ].iloc[0]).astype(int)
        # binntmp[ 'chromStartBp2' ] = ((binntmp[ 'Dbp' ] + binntmp[ 'Dbp2' ]).cumsum() + binntmp[ 'chromStartBp' ].iloc[0] - model.bpres).astype(int)

        binntmp[ 'class' ] = binntmp[ 'class' ].fillna( method='ffill')
        binntmp['type'] = binntmp[ model.modelname ]

        if binntmp.dpn2.sum() != binntmp.shape[0]:
            print('This is not good! binning.dpn2.sum() != binning.shape[0]')
            raise 
            
        taildf = pd.DataFrame(
            np.zeros( ( 1,  binntmp.shape[1])) * np.nan ,
            columns = binntmp.columns
            )
        # pdb.set_trace()
        taildf['type'] = int( model.tailtyp )
        taildf['part'] = - 2 * (polid+1) 
        binntmp = binntmp.append( taildf)

        taildf['part'] = - 2 * (polid+1) +1
        binntmp = taildf.append( binntmp )
            
            
        
        
        # map of bs
        bsmaptmp = part4[[ model.modelname ,'dpn2','chromStart']]
        bsmaptmp.columns = ['type','bsbatch','chromStart']
        
        
        
        ### binning
        binntmp['polyid'] = polid
        binning = binning.append( binntmp)
        
        ### binning for comparison
        binning2 = binning2.append(
            binntmp.iloc[binrmvfirst+1:-binrmvlast-1] )
        
        
        ### bs map
        bsmaptmp['polyid'] = polid
        bsmaptmp['monocumsum'] = bsmaptmp['bsbatch'].cumsum()
        bsmap = bsmap.append( bsmaptmp)

        ###        
        polid = polid + 1
        taillen += [taillentmp]
        npol += [npoltmp]
        
        
        ### genome
        genome = [ chrname]
        genome += [ int(binning.chromStartBp2.min())]
        genome += [ int(binning.chromStartBp2.max() + model.bpres)]
        genome += [ model.bpres]
    
    
    
    return npol, taillen, binning, bsmap, genome, binning2




















def defLEtypes( model, strmode = 'sameAsBind_randOrient_stalltails', rseed=42, coheStallSitesFract=None ):

    if strmode == 'sameAsBind_randOrient_stalltails':
        
        # same as binding types
        model.bsmap['monocumsum2'] = model.bsmap['bsbatch'].cumsum().shift(fill_value=0)
        bsmapstall = model.bsmap[ np.isin( model.bsmap.type, model.idlopestall )]

        # remove a fraction of sites because not real insulation sites    
        if coheStallSitesFract is not None:
            bsmapstall = bsmapstall.sample( frac=coheStallSitesFract, random_state=rseed)

        # add random ctcf orientation    
        np.random.seed( rseed)
        bsmapstall['stallOrientation'] = np.random.randint(0,2,bsmapstall.shape[0])
    
        idd1not = []
        for stallid in range(0,2):
            bstmp = bsmapstall[ bsmapstall['stallOrientation']== stallid]
    
            idd1not += [ np.concatenate(
                [ np.arange( int(rrow.monocumsum2), int(rrow.monocumsum)) for idx, rrow in bstmp.iterrows()]
                ).tolist()    ]
            

        # add stalling on tails
        for stallid in range(0,2):
                idd1not[ stallid] += list(range(0,model.taillen[0]))
                idd1not[ stallid] += list(range( model.npol[0]-model.taillen[0], model.npol[0] ))
                
    
    return idd1not





def defLEtypes2( ssystem, model, strmode = 'sameAsBind_randOrient_stalltails', rseed=42, coheStallSitesFract=None, strSys='espresso' ):

    if strmode == 'sameAsBind_randOrient_stalltails':
        random.seed( rseed)
        if strSys =='espresso':
            idstall = ssystem.part[:].id[ np.isin( ssystem.part[:].type, model.idlopestall )]
        elif strSys == 'df':
            idstall = ssystem.Index[ np.isin( ssystem.type, model.idlopestall )].values
            
        # remove a fraction of sites because not real insulation sites    
        idstall = np.array( random.sample( idstall.tolist(), int(idstall.size * coheStallSitesFract) ) )

        # add random ctcf orientation    
        np.random.seed( rseed)
        idstallbool = np.random.randint(0,2,idstall.size)
        idd1not = [
            list(idstall[ idstallbool==0]),
            list(idstall[ idstallbool==1])
            ]
        
        
        # add stalling on tails
        for stallid in range(0,2):
            # idd1not[ stallid] += list(range(0,model.taillen[0]))
            # idd1not[ stallid] += list(range( model.npol[0]-model.taillen[0], model.npol[0] ))
            if strSys =='espresso':
                idd1not[ stallid] += list( ssystem.part[:].id[ np.isin( ssystem.part[:].type, model.stallAnywayTypeLope)] )
            elif strSys == 'df':
                idd1not[ stallid] += list( ssystem.Index[ np.isin( ssystem.type, model.stallAnywayTypeLope)] )
            
                
            
    elif strmode in ['artOrient_stalltails','artOrient_stalltails_invertOrient']:
        if strSys =='espresso':
            idlopestallnotail = list(set(model.idlopestall)-set([model.tailtyp]))
            idstall = ssystem.part[:].id[ np.isin( ssystem.part[:].type, idlopestallnotail )]
        elif strSys == 'df':
            idlopestallnotail = list(set(model.idlopestall)-set([model.tailtyp]))
            idstall = ssystem.Index[ np.isin( ssystem.type, idlopestallnotail )].values
                    
        # add ctcf from model
        strandpd = model.binnffill[ 
            ( np.isin( model.binnffill.strand, ['+','-']) ) & 
            ( np.isin( model.binnffill.type, idlopestallnotail ) )
            ][['Index','strand']]

        if strmode in ['artOrient_stalltails']:
            idd1not = [
                list(idstall[ strandpd.strand=='+']),
                list(idstall[ strandpd.strand=='-'])
                ]        
        elif strmode in ['artOrient_stalltails_invertOrient']:
            idd1not = [
                list(idstall[ strandpd.strand=='-']),
                list(idstall[ strandpd.strand=='+'])
                ]        

        
        # add stalling on tails
        for stallid in range(0,2):
            # idd1not[ stallid] += list(range(0,model.taillen[0]))
            # idd1not[ stallid] += list(range( model.npol[0]-model.taillen[0], model.npol[0] ))
            if strSys =='espresso':
                idd1not[ stallid] += list( ssystem.part[:].id[ np.isin( ssystem.part[:].type, model.stallAnywayTypeLope)] )
            elif strSys == 'df':
                idd1not[ stallid] += list( ssystem.Index[ np.isin( ssystem.type, model.stallAnywayTypeLope)] )
                        
    
    return idd1not




def defReeltypes( model, strmode = 'sameAsBind_randOrient_stalleverywhere', rseed=42 ):

    if strmode == 'sameAsBind_randOrient_stalleverywhere':
        
        # same as binding types
        model.bsmap['monocumsum2'] = model.bsmap['bsbatch'].cumsum().shift(fill_value=0)
        bsmapstall = model.bsmap[ np.isin( model.bsmap.type, model.idreelstall )]
    
        # add random ctcf orientation    
        np.random.seed( rseed)
        bsmapstall['stallOrientation'] = np.random.randint(0,2,bsmapstall.shape[0])
    
        idd1not = []
        for stallid in range(0,2):
            bstmp = bsmapstall[ bsmapstall['stallOrientation']== stallid]
    
            idd1not += [ np.concatenate(
                [ np.arange( int(rrow.monocumsum2), int(rrow.monocumsum)) for idx, rrow in bstmp.iterrows()]
                ).tolist()    ]
            
        # add stalling on tails
        for stallid in range(0,2):
            if stallid == 0:
                idd1not[ stallid] += list(range(0,model.taillen[0]))
            elif stallid == 1:
                idd1not[ stallid] += list(range( model.npol[0]-model.taillen[0], model.npol[0] ))
                
    
    return idd1not














def addTailCons( bpres, chrname, chrstart, chrend, bpmin = 90 ):
    N = 10
    ctail = ['m']

    binding, npol = genBrackley2( bpres, chrname, chrstart, chrend, bpmin = 90 )
    binding += [ list( range(0, N )) + list( range( npol-1 , npol-1 + N )) ]
    npol = npol + 2 * N
    
    return binding, npol, ctail














def readH5( strFile):
    
    # Check if all good    
    import h5py
    h5file = h5py.File( strFile, 'r')
    positions = h5file['particles/atoms/position/value']   
        
        





def readVTF():
    
    # =============================================================================
    #     import vtk
    #     reader = vtk.vtkUnstructuredGridReader(  'trajectory_'+str_sys+'.vtf')
    # =============================================================================

    pass



def writeCustom( ssystem):
    ppos = ssystem.part[:].pos
    typv = ssystem.part[:].type    
    
    df = pd.DataFrame(np.concatenate( (ppos, typv[:,None]), 1 ) )
    df['time'] = ssystem.time
    df['obs'] = 'pos'
    
    df.columns = ['x','y','z','type','time','obs']

    return df



def writeV( ssystem):
    pv = ssystem.part[:].v
    typv = ssystem.part[:].type    
    
    df = pd.DataFrame(np.concatenate( (pv, typv[:,None]), 1 ) )
    df['time'] = ssystem.time
    df['obs'] = 'v'
    
    df.columns = ['x','y','z','type','time','obs']

    return df



def readCustom( strfile):
    
    ssystem  = pd.read_csv( strFile, compression='gzip')
    
    return ssystem





def warmup( wmpDic):
        
    SIG = wmpDic['SIG']
    LJCUT = wmpDic['LJCUT']
    epsCapMax = wmpDic['epsCapMax']
    epsCapMin = wmpDic['epsCapMin']
    nnn = wmpDic['nnn']
    www = wmpDic['www']
    ttt = wmpDic['ttt']
    Ehf = wmpDic['Ehf']
    Ehi = wmpDic['Ehi']
    Enbf = wmpDic['Enbf']
    Enbi = wmpDic['Enbi']
    alltypes = wmpDic['alltypes']
    allpairs = wmpDic['allpairs']
    pairsb = wmpDic['pairsb']

    if 'pairglj' in wmpDic.keys():
        pairglj = wmpDic['pairglj']


    #
    Deps = epsCapMax - epsCapMin    
    lll = .1 # starting value
    mmm = .005 # starting value
    
    WARM_N_TIME = np.int_( (epsCapMax - epsCapMin) / epsCapMax / lll )
    WARM_STEPS = np.int_( epsCapMax * lll / mmm)
    
    # lll <= Deps / epsCapMax / nnn
    # WARM_STEPS >= www
    while lll > Deps / epsCapMax / nnn or WARM_STEPS < www or mmm > Deps / ttt:
        lll = lll / 1.5
        WARM_STEPS = np.int_( epsCapMax * lll / mmm)
        if lll > Deps / epsCapMax / nnn or WARM_STEPS < www or mmm > Deps / ttt:
            mmm = mmm / 2.
            WARM_STEPS = np.int_( epsCapMax * lll / mmm)
    
    #
    WARM_N_TIME = np.int_( (epsCapMax - epsCapMin) / epsCapMax / lll )
    ljhcap = np.arange(0,WARM_N_TIME,1) * mmm * WARM_STEPS + epsCapMin 
    #
    wmp_sam = np.int_( np.ones( WARM_N_TIME ) * WARM_STEPS )
    #
    wpT = WARM_N_TIME * WARM_STEPS
    
    
    # =============================================================================
    # Define warmup potentials
    # =============================================================================
    ##
# =============================================================================
#     ljh = {
#         'eps': np.zeros(( len(alltypes), len(alltypes), WARM_N_TIME)) ,
#         'sig': np.zeros(( len(alltypes), len(alltypes), WARM_N_TIME)) ,
#         'cut': np.zeros(( len(alltypes), len(alltypes), WARM_N_TIME)) ,
#         }
# =============================================================================
    ljh = {
        'eps': np.zeros(( alltypes[-1]+1, alltypes[-1]+1, WARM_N_TIME)) ,
        'sig': np.zeros(( alltypes[-1]+1, alltypes[-1]+1, WARM_N_TIME)) ,
        'cut': np.zeros(( alltypes[-1]+1, alltypes[-1]+1, WARM_N_TIME)) ,
        }

    # pdb.set_trace()
    
    for tyi in allpairs:
        ljh['eps'][ tyi[0], tyi[1], : ] = np.arange( Ehi[ tyi[0], tyi[1]] , Ehf[ tyi[0], tyi[1]], ( Ehf[ tyi[0], tyi[1]] - Ehi[ tyi[0], tyi[1]] ) / ( WARM_N_TIME ) ) [:WARM_N_TIME]
        ljh['sig'][ tyi[0], tyi[1], : ] = np.ones((WARM_N_TIME)) * SIG # .8
        ljh['cut'][ tyi[0], tyi[1], : ] = 2**(1/6.) * ljh['sig'][ tyi[0], tyi[1], : ]  
    
    if 'pairglj' in wmpDic.keys():
        for tyi in pairglj:
            ljh['eps'][ tyi[0], tyi[1], : ] = np.arange( Ehi[ tyi[0], tyi[1]] , Ehf[ tyi[0], tyi[1]], ( Ehf[ tyi[0], tyi[1]] - Ehi[ tyi[0], tyi[1]] ) / ( WARM_N_TIME ) ) [:WARM_N_TIME]
            ljh['sig'][ tyi[0], tyi[1], : ] = np.ones((WARM_N_TIME)) * SIG # .8
            ljh['cut'][ tyi[0], tyi[1], : ] = 3**(1/4.) * ljh['sig'][ tyi[0], tyi[1], : ]  
    
    
    ##
# =============================================================================
#     ljb = {
#         'eps': np.zeros(( len(alltypes), len(alltypes), WARM_N_TIME)) ,
#         'sig': np.zeros(( len(alltypes), len(alltypes), WARM_N_TIME)) ,
#         'cut': np.zeros(( len(alltypes), len(alltypes), WARM_N_TIME)) ,
#         }
# =============================================================================
    ljb = {
        'eps': np.zeros(( alltypes[-1]+1, alltypes[-1]+1, WARM_N_TIME)) ,
        'sig': np.zeros(( alltypes[-1]+1, alltypes[-1]+1, WARM_N_TIME)) ,
        'cut': np.zeros(( alltypes[-1]+1, alltypes[-1]+1, WARM_N_TIME)) ,
        }
    for tyi in pairsb:
        ljb['eps'][ tyi[0], tyi[1], : ] = np.arange( Enbi[ tyi[0], tyi[1]] , Enbf[ tyi[0], tyi[1]], ( Enbf[ tyi[0], tyi[1]] - Enbi[ tyi[0], tyi[1]]) / ( WARM_N_TIME)) [:WARM_N_TIME]
        ljb['sig'][ tyi[0], tyi[1], : ] = np.ones((WARM_N_TIME)) * SIG # np.arange( .8 , 1.5 * 2**(-1/6.), ( 1.5 * 2**(-1/6.) - .8) / ( WARM_N_TIME))  
        ljb['cut'][ tyi[0], tyi[1], : ] = np.arange( ljh['cut'][ tyi[0], tyi[1], -1 ] , LJCUT , ( LJCUT - ljh['cut'][ tyi[0], tyi[1], -1 ] ) / ( WARM_N_TIME))  [:WARM_N_TIME]
    
    
    
    
        
    return ljh, ljb, wmp_sam, ljhcap









def readConf( loadDict):

    try:
        print('reading config', loadDict['filename']+ '.gz' )
        df = pd.read_csv( loadDict['filename'] + '.gz', compression='gzip', index_col=0)
        # df = pd.read_csv( strStorage + '/umg/traj'+str(runi)+'_'+str_sys+'.gz', compression='gzip', index_col=0)
        # dftimesize = df.time.unique().size
        if loadDict['timeSample'][0] == -1:
            tme = np.sort( df.time.unique())[-1]
            df = df.set_index('time').loc[ tme ].reset_index()
            
        elif type( loadDict['timeSample'] ) is str:
            df = df.set_index('simulation stage').loc[ loadDict['timeSample'] ].reset_index()
                
            tme = np.sort( df.time.unique())[-1]
            df = df.set_index('time').loc[ tme ].reset_index()            
            
        else:
            df = df.set_index('time').loc[ loadDict['timeSample']].reset_index()
        
        
        df['index'] = df.index
        df.type = df.type.apply( int)
        
    except:
        print('config not found. Skipping...')
        raise
        
 
    
    
    return df # , dftimesize



def readConf2( loadDict):

    try:
        print('reading config', loadDict['filename']+ '.gz' )
        df = pd.read_csv( loadDict['filename'] + '.gz', compression='gzip', index_col=0)
        # df = pd.read_csv( strStorage + '/umg/traj'+str(runi)+'_'+str_sys+'.gz', compression='gzip', index_col=0)
        # dftimesize = df.time.unique().size
        dftimes = df.time.unique()

        if loadDict['timeSample'][0] == -1:
            tme = np.sort( df.time.unique())[-1]
            df = df.set_index('time').loc[ tme ].reset_index()
            
        elif type( loadDict['timeSample'] ) is str:
            df = df.set_index('simulation stage').loc[ loadDict['timeSample'] ].reset_index()
                
            tme = np.sort( df.time.unique())[-1]
            df = df.set_index('time').loc[ tme ].reset_index()            
            
        else:
            df = df.set_index('time').loc[ loadDict['timeSample']].reset_index()
        
        
        df['index'] = df.index
        df.type = df.type.apply( int)
        
    except:
        print('config not found. Skipping...')
        raise
    
    
    return df, dftimes








def logSamp( samDict):
    
    ## logarithmic extension
    mint = samDict['mint']
    maxt = samDict['maxt']
    tit = samDict['tit']

    if ('maxt2' in samDict.keys()) and (samDict['maxt2'] is not False):
        maxt2 = samDict['maxt2']
    
        Dn = np.ceil( np.log( (maxt2 - mint) / mint) / np.log( maxt / mint ) * tit) - tit

        # tp = np.int_( mint * (maxt/mint) ** ( np.arange(tit, tit +Dn +1)/tit) ) + mint
        tp = np.int_( mint * (maxt/mint) ** ( np.arange(0, tit +Dn +1)/tit) ) + mint

    else:
        tp = np.int_( mint * (maxt/mint) ** ( np.arange(0, tit +1)/tit) ) + mint
        
    
    return tp    
    
    






















# =============================================================================
# Save configuration
# =============================================================================


def confBegin( ssystem, model, espiowriter):
    if model.str_write=='vmd':
        fp = open( model.filenameTraj +'.vtf', mode='w+t')

        # write structure block as header
        espiowriter.vtf.writevsf( ssystem, fp)
        
        dft = 0
             

    elif model.str_write == 'h5md':
        os.system('rm ' + model.filenameTraj +'.h5')
        fp = espiowriter.h5md.H5md(filename= model.filenameTraj + '.h5'
                       , write_pos=True
                       , write_vel=True
                       , write_species = True
                       )
        fp.write()
        dft = 0
        
    elif model.str_write == 'custom':
        fp = open( model.filenameTraj +'.vtf', mode='w+t')

        # write structure block as header
        espiowriter.vtf.writevsf( ssystem, fp)

        # save csv
        dft = writeCustom( ssystem)




    return fp, dft





def confSave( ssystem, model, espiowriter, fp, dft, strVel = False, precision=None):
    for wri in range(7):
        try:
            if model.str_write=='vmd':
                espiowriter.vtf.writevcf( ssystem, fp)
                
            elif model.str_write == 'h5md':
                espiowriter.h5.write()            
                
            elif model.str_write == 'custom':
                presTime = time.time()
                if (presTime - model.startTime > 5 * 60 * 60) and \
                    ( (model.str_tsam == 'log') or \
                     (( 'strFlagSaveInterm' in dir(model)) and (model.strFlagSaveInterm is True))) : 
                    try: 
                        espiowriter.vtf.writevcf( ssystem, fp)
                        fp.close()
                    except:
                        fp = open( model.filenameTraj +'.vtf', mode='a')
                        espiowriter.vtf.writevcf( ssystem, fp)
                        fp.close()

                    # save csv
                    dftmp = writeCustom( ssystem)                
                    dft = dft.append( dftmp )     
                    
                    # save velocities?
                    if strVel :
                        dfv = writeV( ssystem)
                        dft = dft.append( dfv )   
                    else:
                        print(' -> Saving configuration without velocities')
                        
                    # save only at a lower precision?
                    if precision is not None:
                        dft[['x','y','z']] = np.around( dft[['x','y','z']].values, precision )
                        
                        
                    dft.to_csv( model.filenameTraj +'.gz', compression='gzip')      

                else:
                    espiowriter.vtf.writevcf( ssystem, fp)
                    fp.flush()
                    # append csv
                    dftmp = writeCustom( ssystem)                
                    dft = dft.append( dftmp )                     
                    
                    # save velocities?
                    if strVel :
                        dfv = writeV( ssystem)
                        dft = dft.append( dfv )     
                        
                    # save only at a lower precision?
                    if precision is not None:
                        dft[['x','y','z']] = np.around( dft[['x','y','z']].values, precision )
                        
                                                


            elif model.str_write == 'pdb':
                # espiowriter = universe
                u = espiowriter
                u.atoms.write("system.pdb")

                
            elif model.str_write == 'gromacs':
                eos, u, W = espiowriter
                u.load_new(eos.trajectory)  # load the frame to the MDA universe
                W.write_next_timestep(u.trajectory.ts)  # append it to the trajectory
                                            
                    
            break
        
        except:
            wrimin = random.randrange(10,30)
            time.sleep( 60 * wrimin)
            print('Writing attempt n.',wri,'failed. Retrying in ',wrimin,'minutes...')
     
        if wri == 7:
            print('All writing attempts failed. Save what you can and get out of here.')
            raise
        

    return fp, dft




def confEnd( ssystem, model, espiowriter, fp, dft, precision=None ):
    for wri in range(7):
        try:
            if model.str_write=='vmd':
                espiowriter.vtf.writevcf( ssystem, fp)
                fp.close()
                
            elif model.str_write == 'h5md':
                espiowriter.h5.write() 
                espiowriter.h5.close()            
                
            elif model.str_write == 'custom':
                print('Saving configuration with velocities')
                try: 
                    espiowriter.vtf.writevcf( ssystem, fp)
                    fp.close()
                except:
                    fp = open( model.filenameTraj +'.vtf', mode='a')
                    espiowriter.vtf.writevcf( ssystem, fp)
                    fp.close()

                # save csv
                dftmp = writeCustom( ssystem)                
                dfv = writeV( ssystem)
        
                dft = dft.append( dftmp )                     
                dft = dft.append( dfv )                     

                # save only at a lower precision?
                if precision is not None:
                    dft[['x','y','z']] = np.around( dft[['x','y','z']].values, precision )
                        
                dft.to_csv( model.filenameTraj +'.gz', compression='gzip')      


            elif model.str_write == 'pdb':
                # espiowriter = universe
                u = espiowriter
                u.atoms.write("system.pdb")

                
            elif model.str_write == 'gromacs':
                eos, u, W = espiowriter
                u.load_new(eos.trajectory)  # load the frame to the MDA universe
                W.write_next_timestep(u.trajectory.ts)  # append it to the trajectory
                    
    
            break
        
        except:
            wrimin = random.randrange(10,30)
            time.sleep( 60 * wrimin)
            print('Writing attempt n.',wri,'failed. Retrying in ',wrimin,'minutes...')
     
        if wri == 7:
            print('Final config I/O failed. Final configuration lost')    





def confEndWarmupError( model, espiowriter, fp ):
    if model.str_write=='vmd':
        fp.close()
        
    elif model.str_write == 'h5md':
        espiowriter.h5.close()            
        os.system('rm ' + strStorage + '/traj'+ procid +'_'+str_sys+'.h5')         
        
    elif model.str_write == 'custom':
        fp.close()     




def saveVTK( ssystem, fn):
    # write to VTK
    ssystem.part.writevtk("part_type_0_1.vtk", types=[0, 1])
    ssystem.part.writevtk("part_type_2.vtk", types=[2])
    ssystem.part.writevtk("part_all.vtk")    





###========================================================================
# Measure
###========================================================================  

def obsSave( ssystem, model, dynName, exept=False):
    
    if exept :
        obss = pd.DataFrame(
            data = {
                model.obs_cols[0]: [ssystem.time] ,
                model.obs_cols[1]: [  np.nan ] ,
                model.obs_cols[2]: [  np.nan ] ,
                }
            )
    
    else:
        energies = ssystem.analysis.energy()

        obss = pd.DataFrame(
            data = {
                model.obs_cols[0]: [ssystem.time] ,
                model.obs_cols[1]: [energies['kinetic'] / (1.5 * np.sum( model.Npart ))] ,
                model.obs_cols[2]: [energies['total']] ,
                }
            )

    for idtyp, tyi in enumerate(model.typev):
        try:
            rg2 = ssystem.analysis.gyration_tensor( p_type= tyi)['Rg^2']    
            
            if rg2 == 0: obss[ model.obs_cols[idtyp + 3]] = np.nan
            else: obss[ model.obs_cols[idtyp + 3]] = rg2

        except:
            obss[ model.obs_cols[idtyp + 3]] = np.nan
            
            

    obss['simulation stage'] = dynName

    return obss







def obsBegin( ssystem, model):
        
    return obsSave( ssystem, model, 'start')

    

def obsInterm( ssystem, model, obss_all, dynName, exept = False):
    
    obss_all = obss_all.append( obsSave( ssystem, model, dynName, exept ) )

    return obss_all




def obsEnd(obss_all, model, env):
    # pickle
    if model.str_saveMode == 'pickle':
        with open(env.strStorEspr + '/' + model.str_syst + '/obs/pickle_' + model.str_syst, 'wb') as handle:
            pickle.dump( [rg2, typev, stime]
                        , handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    # pandas table        
    elif model.str_saveMode == 'pandas':
        try:
            data_df_old = pd.read_csv( env.strStorEspr + '/' + model.str_syst + '/obs/obs_' + model.str_syst +'_'+model.str_param +'.csv', index_col=None)
            data_df_old.append( obss_all).to_csv( env.strStorEspr + '/' + model.str_syst + '/obs/obs_' + model.str_syst +'_'+model.str_param + '.csv', index=False)
            print('Warning: file appended to an existing one.')
        except:
            obss_all.to_csv( env.strStorEspr + '/' + model.str_syst + '/obs/obs_' + model.str_syst +'_'+model.str_param + '.csv', index=False)      

    # pandas table        
    elif model.str_saveMode == 'pandas+w':
        obss_all.to_csv( env.strStorEspr + '/' + model.str_syst + '/obs/obs_' + model.str_syst +'_'+model.str_param + '.csv', index=False)      






## new obs measure
def obsLopeSave(obss_all, model, env):
    
    # conf ID
    obss_all['conf'] = model.procid
    obss_all['param']= model.str_param
    

    
    # pickle
    if model.str_saveMode == 'pickle':
        with open(env.strHome + '/pickle_' + model.str_syst+'_'+ model.strC, 'wb') as handle:
            pickle.dump( [rg2, typev, stime]
                        , handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    # pandas table        
    elif model.str_saveMode == 'pandas':
        try:
            data_df_old = pd.read_csv( env.strHome + '/obsLope_' + model.str_syst+'_'+ model.strC +'.csv', index_col=None)
            data_df_old.append( obss_all).to_csv( env.strHome + '/obsLope_' + model.str_syst+'_'+ model.strC + '.csv', index=False)
            print('Warning: file appended to an existing one.')
        except:
            obss_all.to_csv( env.strHome + '/obsLope_' + model.str_syst+'_'+ model.strC + '.csv', index=False)      



def obsLopeSave2(obss_all, model, env, strKind, When='interm'):
    
    # conf ID
    # obss_all['conf'] = model.procid
    # obss_all['param']= model.str_param

    presTime = time.time()
    if (When == 'end') or ( (presTime - model.startTime > 5 * 60 * 60) and \
                    ( (model.str_tsam == 'log') or \
                     (( 'strFlagSaveInterm' in dir(model)) and (model.strFlagSaveInterm is True))) ): 
        print(' -> Saving '+strKind+' obs')

    
        # pickle
        if model.str_saveMode == 'pickle':
            with open(env.strStorEspr + '/' + model.str_syst + '/obs/pickle_' + model.str_syst+'_'+ model.strC, 'wb') as handle:
                pickle.dump( [rg2, typev, stime]
                            , handle, protocol=pickle.HIGHEST_PROTOCOL)
                
        # pandas table        
        elif model.str_saveMode == 'pandas':
            try:
                data_df_old = pd.read_csv( env.strStorEspr + '/' + model.str_syst + '/obs/obs'+strKind+'_' + model.str_syst+'_'+ model.strC +'_'+model.str_param+'_c' + model.procid +'.csv.gz', index_col=None, compression='gzip')
                data_df_old.append( obss_all).to_csv( env.strStorEspr + '/' + model.str_syst + '/obs/obs'+strKind+'_' + model.str_syst+'_'+ model.strC +'_'+model.str_param+ '_c' + model.procid +'.csv.gz', index=False, compression='gzip')
                print('Warning: file appended to an existing one.')
            except:
                obss_all.to_csv( env.strStorEspr + '/' + model.str_syst + '/obs/obs'+strKind+'_' + model.str_syst +'_'+ model.strC +'_'+model.str_param+ '_c' + model.procid +'.csv.gz', index=False, compression='gzip')      

        # pandas table        
        elif (model.str_saveMode == 'pandas+w') and (When=='start'):
            obss_all.to_csv( env.strStorEspr + '/' + model.str_syst + '/obs/obs'+strKind+'_' + model.str_syst +'_'+ model.strC +'_'+model.str_param+ '_c' + model.procid +'.csv.gz', index=False, compression='gzip')      



        obss_all = pd.DataFrame([])    
    else:
        pass
        
    return obss_all












# =============================================================================
# conf I/O
# =============================================================================










#####

def bin2bead( df ):
    tmpl = list( df.type.astype(int).values )
    tmpl.sort()
    return tmpl


def bin2type( df, binning, typv):
    df['Index'] = df.index
    df = df.set_index('type').loc[ typv]
    df = df.sort_values( 'Index').reset_index()    
    df['Index'] = df.index
    # 
    binning['Index'] = binning.reset_index().index
    
    #
    ppos02 = df.merge( binning, on='Index', how='inner')
    beatypl = ppos02[['type','part']].groupby('part').apply( bin2bead).tolist()
    beatypl.sort()
    return list( beatypl  for beatypl,_ in itertools.groupby(beatypl ))



def bin2type2( binning):
    partBeads = binning[['type','part','polyid']].groupby(['polyid','part'], sort=False).apply( bin2bead)
    
    beatypl = partBeads.tolist()
    beatypl.sort()
    
    return list( beatypl  for beatypl,_ in itertools.groupby(beatypl )), partBeads



def getUniqLL( LL):
    return list( beatypl  for beatypl,_ in itertools.groupby( LL ))



def getUniqLL2( LL):
    return [list(x) for x in set(tuple(x) for x in LL)]










def selectSimulTime( strTimel, tpveq):
    # check time condition
    if 'condition' in re.split('\+', strTimel[0]):
        timev = df.time.unique()
        tpv = np.argwhere( np.isclose( timev, te, atol=tatol ))
        if 'last' in re.split('\+', strTimel[0]):
            tpv = tpv[-1]

    else:
        tpv = tpveq 
        
    return tpv






# =============================================================================
# Check functions
# =============================================================================



def check_lopPairs( lopePairs, sysanis, model):

    lpdf = pd.DataFrame( lopePairs)
    lpdf.columns = ['poltype', 'anistype', 'strand']
    lpdf.sort_values(by='anistype', inplace=True)
    
    boolLopairs = False
    
    lloop = lpdf.merge( sysanis, how= 'left', left_on = 'anistype', right_on = 'id' )
    # this filters only pairs belonging to the same dimer
    llooptmp = lloop.groupby('dimPairs').agg( len).reset_index()[['dimPairs','id']]
    llooptmp = llooptmp[ llooptmp ['id'] == 2 ]
    # llooptmp.columns = ['dimPairs','id']
    lloop2 = lloop.merge( llooptmp['dimPairs'], how='right', on='dimPairs' )
    # this filters only for dimers binding locally, not binding to distal sites
    if lloop2.shape[0] > 0 :
        lloop3 = lloop2.pivot_table( columns = 'type', index='dimPairs', values='poltype')
        lloop3['loopDim'] = lloop3[ model.anistype[0]] - lloop3[ model.anistype[1]]
        
        ### this filters for local loops (loopDim=1,2), but also well directioned (loopDim!=-1,-2)
        loopid1 = lloop3[ np.isin( lloop3.loopDim, model.minLoopSize ) ].reset_index()[ model.anistype[0] ].tolist() + \
                    lloop3[ np.isin( lloop3.loopDim, model.minLoopSize ) ].reset_index()[ model.anistype[1] ].tolist()
        lopePairs = lopePairs[ np.isin( lopePairs[:,0], loopid1), : ]      
    
    
        boolLopairs = (lloop3.loopDim < 0).sum() > 0
    
    
    return boolLopairs



def reportLoops( lopePairs, sysanis, model):

    lpdf = pd.DataFrame( lopePairs)
    lpdf.columns = ['poltype', 'anistype', 'strand']
    lpdf.sort_values(by='anistype', inplace=True)
    
    boolLopairs = False
    
    lloop = lpdf.merge( sysanis, how= 'left', left_on = 'anistype', right_on = 'id' )
    # this filters only pairs belonging to the same dimer
    llooptmp = lloop.groupby('dimPairs').agg( len).reset_index()[['dimPairs','id']]
    llooptmp = llooptmp[ llooptmp ['id'] == 2 ]
    # llooptmp.columns = ['dimPairs','id']
    lloop2 = lloop.merge( llooptmp['dimPairs'], how='right', on='dimPairs' )
    # this filters only for dimers binding locally, not binding to distal sites
    if lloop2.shape[0] > 0 :
        lloop3 = lloop2.pivot_table( columns = 'type', index='dimPairs', values='poltype')
        lloop3['loopDim'] = lloop3[ model.anistype[0]] - lloop3[ model.anistype[1]]
        
        ### this filters for local loops (loopDim=1,2), but also well directioned (loopDim!=-1,-2)
        loopid1 = lloop3[ np.isin( lloop3.loopDim, model.minLoopSize ) ].reset_index()[ model.anistype[0] ].tolist() + \
                    lloop3[ np.isin( lloop3.loopDim, model.minLoopSize ) ].reset_index()[ model.anistype[1] ].tolist()
        lopePairs = lopePairs[ np.isin( lopePairs[:,0], loopid1), : ]      
    
    
        #boolLopairs = (lloop3.loopDim < 0).sum() > 0
    
    else:
        lloop3 = np.empty((0,3), dtype=np.int64)
    
    return lloop3
















def PrintException():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print( 'EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj) )











