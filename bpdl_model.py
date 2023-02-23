#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 11:29:45 2022

@author: mariano
"""




import os
import sys
from itertools import product, combinations, combinations_with_replacement
import numpy as np
import pandas as pd
import importlib
import re
import pdb 

#### import simulation settings
import esprenv as env
import func0 as func

    





class bpdl():
    
    def __init__( self, *initial_data, **kwargs):
        
                
        
        
        # =============================================================================
        # Check for user provided inputs
        # =============================================================================
        # Default params
        self.mint, self.maxt, self.tit, self.maxt2 = 100, 10000, 100, False

        #
        self.str_flush = False


        #
        self.nMPI = 1
        self.dyn1 = ''
        self.dyn2 = 's1'
        
        

        
        # =============================================================================
        #         Variate options string
        # =============================================================================
        self.otheropts = 'd0' 
        
        
                
        
        
        
        # =============================================================================
        # Update settings from user input
        # =============================================================================
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])


    
    


        # =============================================================================
        #     Set up parameters 
        # =============================================================================
        self.tailtyp = 23   
        
        self.mol_type = [0,10]
            
        self.specific_type = [10,11]
        self.reel_type = [1]
        self.spring_type = []
        self.anistype = [6,8]
        self.anistypebond = [7,9]
        self.pol_change = [22]
        
        
        self.npoly = 1
        self.str_param_file1 = self.str_param
        self.str_param_file2 = self.str_param            
        
        



        # =============================================================================
        #         Set up polymers and types
        # =============================================================================
        self.modelods = 'degron'
        self.modelods_modelsheet = 'HMMmodel'
        self.modelods_energysheet = 'paramtypeReal'
        self.modelods_paramsheet = 'paramReal'
        self.modelods_concsheet = 'concentrations'
        self.modelname = 'sbsle9'

        self.artifSys = [ env.strHome + '/' +'degron_sys.ods', self.strReg ]
        self.npol, self.taillen, self.binning, self.bsmap, self.genome, self.binning2 = func.genReal( self )

        self.chrname, self.chrstart, self.chrend, self.bpres = self.genome
        
        self.chrend = self.chrend + 2 * self.taillen[0] * self.bpres ### temp




        ######## types
        self.pol_type = list( self.bsmap.sort_values('type').type.unique()  )
        if 'pol_change' in locals():
            self.pol_type += self.pol_change
        
        self.ptail = [ self.tailtyp ]
        self.ctail = [ ]           
            

        
        
        ######## types
        self.alltypes = self.mol_type + self.reel_type + self.spring_type + self.specific_type +\
            self.anistype + self.anistypebond + self.pol_type
        self.alltypes = list( set( self.alltypes) )
        self.alltypes.sort()
        
        self.nbs = len( self.pol_type )
        
        
        self.npolall = self.npol


            
           


        
        # =============================================================================
        # Set up nomenclature    
        # =============================================================================
        self.str_syst = self.str_syst + '_' + self.strReg








        
        
        
        # =============================================================================
        # #### Potentials parameters
        # =============================================================================
        self.paramlist = self.str_param
        
        print('--> Energies and others: in file', env.strHome + '/model'+self.modelods+'.ods' ,
              ', sheet ', self.modelods_energysheet, 'and', self.modelods_paramsheet, 
              ', model type:', self.str_param)
        self.Energies = pd.read_excel( env.strHome + '/'+self.modelods  +'_model.ods', engine="odf", sheet_name=self.modelods_energysheet)
        try: self.Energies.drop('types', 1, inplace=True)
        except: pass        
        
        self.Model = pd.read_excel( env.strHome + '/'+self.modelods  +'_model.ods', engine="odf", sheet_name=self.modelods_paramsheet)
        
        # get model params
        self.GAMMA, self.KbT, self.TIME_STEP, self.SKIN, self.SIG, self.LJCUT, self.STIFF, self.KBEND = self.Model.set_index('model').loc[ self.str_param ][['gamma','kbt','timestep','skin','sigma','ljcut','stiff','kbend']].values

        






        # =============================================================================
        # # Build Affinity matrices
        # =============================================================================
        # LJ eps - end of warmup
        Enbfdf = self.Energies.set_index(['type','stage','model','par','inter']).loc[('lj','dyn1',self.str_param,'eps','unbn')]
        
        self.Enbf = np.zeros(( self.alltypes[-1]+1, self.alltypes[-1]+1))
        
        self.Enbf[ Enbfdf['I'].astype(int), Enbfdf['j'].astype(int)] = Enbfdf['val']
        self.Enbf[ Enbfdf['j'].astype(int), Enbfdf['I'].astype(int)] = Enbfdf['val']



        # hard sphere eps - end of warmup
        Ehfdf = self.Energies.set_index(['type','stage','model','par','inter']).loc[('wca','dyn1',self.str_param,'eps','unbn')]
        
        self.Ehf = np.zeros(( self.alltypes[-1]+1, self.alltypes[-1]+1))
        
        if (Ehfdf.I == 'all').sum() > 0:
            self.Ehf[ :, :] = Ehfdf.val.values
        else :
            self.Ehf[ Ehfdf['I'].astype(int), Ehfdf['j'].astype(int)] = Ehfdf['val']
            self.Ehf[ Ehfdf['j'].astype(int), Ehfdf['I'].astype(int)] = Ehfdf['val']
            
            
        # hard sphere eps - start of warmup
        Ehidf = self.Energies.set_index(['type','stage','model','par','inter']).loc[('wca','wmp',self.str_param,'eps','unbn')]
        

        self.Ehi = np.zeros(( self.alltypes[-1]+1, self.alltypes[-1]+1))
        
        if (Ehidf.I == 'all').sum() > 0:
            self.Ehi[ :, :] = Ehidf.val.values
        else :
            self.Ehi[ Ehidf['I'].astype(int), Ehidf['j'].astype(int)] = Ehidf['val']
            self.Ehi[ Ehidf['j'].astype(int), Ehidf['I'].astype(int)] = Ehidf['val']            


        # LJ bind eps - start of warmup
        Enbidf = self.Energies.set_index(['type','stage','model','par','inter']).loc[('wca','dyn1',self.str_param,'eps','unbn')]
        

        self.Enbi = np.zeros(( self.alltypes[-1]+1, self.alltypes[-1]+1))
                    
        if (Enbidf.I == 'all').sum() > 0:
            self.Enbi[ :, :] = Enbidf.val.values
        else :
            self.Enbi[ Enbidf['I'].astype(int), Enbidf['j'].astype(int)] = Enbidf['val']
            self.Enbi[ Enbidf['j'].astype(int), Enbidf['I'].astype(int)] = Enbidf['val']


    


        
        

        
        

        
        ###========================================================================
        # Polymer stiffness
        ###========================================================================
        self.strBond = 'fene' # fene # harm
        self.k_fene = 30
        self.d_r_max = .8 # .8, 1., .75
        self.r_0_fene = 1.6 # 1.6, 1.75
        self.k_harm = 1.
        self.r_0_harm = 1.
        self.bond_length = 1.6 # 1.6, 1.75
        
        # 
        self.perslen = 3000 # bp
        if self.STIFF == 'all': # all polymer has stiffness
            self.strBondPolAngle = 'cos'
            self.pstiffl = [ list(range( npoli)) for npoli in self.npolall]
            self.kbend =  self.KbT / 2  *  ( self.perslen/ self.bpres )
            self.cosangle = 0
        elif self.STIFF == 'all+k': # all polymer has stiffness
            self.strBondPolAngle = 'cos'
            self.pstiffl = [ list(range( npoli)) for npoli in self.npolall]
            self.kbend = self.KBEND
            self.cosangle = 0            
        elif self.STIFF == 'no':
            self.strBondPolAngle = None
            self.pstiffl = [[]]
        else :
            self.strBondPolAngle = None
            self.pstiffl = [[]]        
        
        
        
        
        # =============================================================================
        #         ######## particles and box
        # =============================================================================
        strContour = 'dense' # rg2 , dense
        ### calc units
        if strContour == 'dense':
            # n mol / liter = 10**-9 mol / 10**-3 cubic meters

            print('--> Concentrations: in file', env.strHome + '/model'+self.modelods+'.ods' ,
                  ', sheet ', self.modelods_concsheet, 'and', self.modelods_paramsheet, 
                  ', model type:', self.str_param)
            self.Concs = pd.read_excel( env.strHome + '/'+self.modelods  +'_model.ods', engine="odf", sheet_name=self.modelods_concsheet)
                        

            self.strCmodel = re.sub( 'e+[0-9]+', '', self.str_param) + self.strC
            Concs = self.Concs.set_index(['model']).loc[(self.strCmodel)]

            concsolo = Concs[Concs['kind'] == 'solo']
            concdimer = Concs[Concs['kind'] == 'dimer']
                            
            self.molC = []
            for tipi in  self.mol_type:
                conc = concsolo[concsolo.type == tipi].nmol_liter.tolist()
                if len(conc) >0:
                    self.molC += conc
                else:
                    self.molC += [0]
        
            self.molCdim = []
            for tipi in  self.anistype:
                conc = list(concdimer[concdimer.type == tipi].nmol_liter.values * 2)
                if len(conc) >0:
                    self.molCdim += conc
                else:
                    self.molCdim += [0]


            self.molAll = self.molC + self.molCdim                
                
                


                  
            
            ###
            print('--> no. of particles and box size: dense environment. Concentration given:', self.molC, 'n mol / liter')
            self.molNpartAll, self.b, self.tB, self.sig, self.tau, self.tLJ = func.calcUnits( self.molAll, self.npol, [self.chrend], [self.chrstart], self.bond_length, eta=.025)
            
            if self.otheropts == 'd5':
                bfact = 2
                self.b = bfact * self.b
            elif self.otheropts == 'd6':
                bfact = 2
                self.b = bfact * self.b
                self.molNpartAll = [ bfact**3 * numi for numi in self.molNpartAll]
            
            
            self.molPart = self.molNpartAll[:len(self.molC)]
            self.ndimi = self.molNpartAll[len(self.molC):]

            ##### 
            for idimi, dimi in enumerate(self.ndimi):
                self.ndimi[idimi] = np.int_( np.round( dimi / 2) )
            self.Npart = np.array( self.molPart + self.ndimi + self.npolall )

        
        
        elif strContour == 'rg2':
            self.molPart = [ 200, 600]
            self.ndimi = 50
            print('--> no. of particles and box size: dilute solution [box = rg2]. No. of binders given:',self.moln)
            self.Npart = np.array( self.molPart + [2*self.ndimi] + self.npolall )
            self.Radii = np.array( [  self.SIG ,  self.SIG , self.SIG , self.SIG  ] )
            self.Density = np.array([.003,.003,.003, .01])
            self.b = np.int_( 2 * (  ( self.Npart * self.Radii ** 3 ).sum() / self.Density.sum() )**(1/3) )



        

        


        
        
        ###========================================================================
        # Types Settings
        ###========================================================================
        ### types
        self.typev = self.alltypes
        
        ### set observables
        self.obs_cols =  ['time','instantaneous_temperature','etotal'] + [ 'rg2 type ' + str(int(typint)) for typint in self.typev ]
        
        # pairls with non zero energy
        self.eel = [ list( ee) for ee in np.where( np.triu( self.Enbf ) >0) ]

        # interacting pairs
        self.pairsb = list( zip( self.eel[0], self.eel[1] ))
        
        # all pairs (for warmup)
        self.allpairs =  list( combinations_with_replacement( self.typev, 2) )
        
        # hard sphere interactions
        self.pairsHard = list( set( self.allpairs ) - set( self.pairsb ) )  
        
        # poly pairs
        self.polpairs = list( combinations_with_replacement( self.pol_type, 2) )
        
        
        
        

        
        
        # =============================================================================
        # Loop extrusion settings
        # =============================================================================
        
        if len(self.anistype) > 0:
            self.strLope = True
        else:
            self.strLope = False        
        

        if self.strLope:
            self.lope_cutoff= self.bond_length 
            
            ####
            self.lope_MoveCutoff= self.SIG * 1.2 
            self.lopedt = 200 
            self.lopeDeathProb = .00006
                
            ####
            self.strLopeBond = 'harm' # fene # harm 
            # harm
            self.r0harmLope = self.bond_length 
            self.kharmLope = 8
            self.rcHarmLope = self.bond_length * 3.5
            self.kharmLopeM = 5
            self.rcHarmLopeM = self.bond_length * 5
            # fene
            self.r0feneLope = self.bond_length 
            self.kfeneLope = 30
            self.drmaxFenelope = .8
            
            self.idlopestall = [19] # stall on insulators 
            self.detachAlwaysType = [self.tailtyp]
            self.idanis = list( set(self.pol_type) - set(self.idlopestall ) - set(self.detachAlwaysType)) # attach eveywhere
            
            self.dmov = [1,-1]
            self.stallAnywayTypeLope = []
            
            ####
            self.P2proxyLEtype = [0,1]
            self.lopePeproxy_cutoff = self.bond_length * 1.5
            ###
            
            ### 
            self.lopeHighProbBirthType = [15,21]
                

            
            if re.match( '.*dCHp3.*', self.otheropts):   
                self.lopeBirthProbType = np.ones( len(self.pol_type)) * .1 # [.9,.1,0,.1,0]
                self.lopeBirthProbType[ np.where( np.isin( np.array(self.pol_type), self.tailtyp))[0]] = 0

                self.lopeBirthProbTypeBase = np.ones( len(self.pol_type)) * .1 # [.9,.1,0,.1,0]
                self.lopeBirthProbTypeBase [ np.where( np.isin( np.array(self.pol_type), self.tailtyp))[0]] = 0

            elif re.match( '.*dCHp5.*', self.otheropts):   
                self.lopeBirthProbType = np.ones( len(self.pol_type)) * .1 # [.9,.1,0,.1,0]
                self.lopeBirthProbType[ np.where( np.isin( np.array(self.pol_type), self.lopeHighProbBirthType))[0]] = .9
                self.lopeBirthProbType[ np.where( np.isin( np.array(self.pol_type), self.tailtyp))[0]] = 0
                
                self.lopeBirthProbTypeBase = np.ones( len(self.pol_type)) * .1 # [.9,.1,0,.1,0]
                self.lopeBirthProbTypeBase [ np.where( np.isin( np.array(self.pol_type), self.tailtyp))[0]] = 0
                



            self.lopeBirthProbDf = pd.DataFrame( data={
                'type' : self.pol_type ,
                'lopeBirthProb': self.lopeBirthProbType
                })  
            
            self.lopeBirthProbDfBase = pd.DataFrame( data={
                'type' : self.pol_type ,
                'lopeBirthProb': self.lopeBirthProbTypeBase
                })                  
        
            # allow also the dimer to bind on the same promoter (this is good when we model the promoter as one bead only)
            self.minLoopSize = [0] 
        
            # types of the molecule that stalls lope
            self.lopeStallMol = [ 10,11 ]
            self.lopeStall_cutoff = self.bond_length * 2
            self.specCheck = 'bond' # bond, distance                    
            
                


            # =============================================================================
            #         Generate LE types
            # =============================================================================
            self.lopeInsulMode = 'artOrient_stalltails_invertOrient'
                  

            
        
        

        # =============================================================================
        # Reel extrusion settings
        # =============================================================================
        try:
            
            if re.match( '.*P2passCH.*', self.otheropts):   
                self.P2passCH = 'P2pass'
            elif re.match( '.*P2bothpassCH.*', self.otheropts):   
                self.P2passCH = 'both'            
            elif re.match( '.*P2notpassCH.*', self.otheropts):   
                self.P2passCH = 'P2notpass'            
            
            Ereel = self.Energies.set_index(['model','inter']).loc[(self.paramlist,'reel')]
                
            Ereelp = Ereel.reset_index()[['I','j','par','val']]
            nreel = Ereel.shape[0]
            
            self.strReel = [True] * nreel
            self.reeldt = [ 300] * nreel
            
            self.strReelBond = [ 'harm' ] * nreel # fene # harm 
            self.r0harmReel = [ self.bond_length ] * nreel 
            self.kharmReel = [ 8 ] * nreel
            self.rcHarmReel = [self.bond_length * 3] * nreel
            
            self.r0feneReel = [ self.bond_length ] * nreel 
            self.kfeneReel = [ 30] * nreel
            self.drmaxFeneReel = [ .8] * nreel
            
            self.dmov = [1,-1]
        
            self.reelseed = [ 42 ] * nreel
            
            
            self.polreeltyp = [[]] * nreel
            self.molreeltyp = [[]] * nreel
            mapreel = {}
            for idree, reerows in  Ereelp.iterrows():
                if np.isnan( reerows.val ):
                    self.molreeltyp[ idree] = [ reerows.I-1 ] # type to attach to
                    self.polreeltyp[ idree] = [ reerows.j ] # type of attaching binder 
    
                mapreel.update({(reerows.I,reerows.j) : idree})
                Ereelp.drop( idree, axis=0, inplace=True)
                
            if Ereelp.shape[0] > 0:
                for key, val in mapreel.items():
                    for parid, reerows in Ereelp.set_index(['I','j']).loc[(key)].iterrows():
                        exec( 'self.' + reerows.par + '['+ str( val )+'] = ' + str(reerows.val))

        

            
            ###
            self.reel_MoveCutoff= [self.SIG * 1.05  ] * nreel 
                
            self.reel_cutoff= [self.bond_length * 1.5] * nreel 

            self.reelBirthProb = [ .7] * nreel 
            self.reelDeathProb = [ .1] * nreel 

            self.reelpos = [[0,1]] * nreel
            self.reelMoveProbStart = .4      
            



            ## Enhancer preferential loader?
            # if promoter is bound by rnap, cohesin cannot bind (but it can for other beads like enhancers)
            self.reelIdxNotLope = [0]
   

        except:
            self.strReel = []








        # =============================================================================
        # Potentials-Bonds types
        # =============================================================================
        
        self.binnffill = self.binning.fillna( method='ffill')
        genomeindex = [-1] + list(range( self.taillen[0], self.taillen[0] + self.binnffill[1:-1].shape[0] )) + [-2] 
        
        self.binnffill['Index'] = genomeindex            
        
        
        ## define start, end and reel types
        reelclass = ['ActivePromoter','Enhancer','PoisedPromoter']
        start = [['1_Active_Promoter']]
        reel = [['GENE','INTRON','EXON','CTCF+G','TES','ENHANCER+G']]
        end = [['TES']]
        
        self.idd1notReel, self.geneend, self.id1reel = [], [], []
        
        for idxreel, strReeli in enumerate(self.strReel):
            
            bnfll = self.binnffill
    
            ###            
            bnfllp = bnfll[bnfll.strand == '+']
            bnflln = bnfll[bnfll.strand == '-']
    
    
            ###
            tail1 = list(range( 0, self.taillen[0]))
            tail2 = list(range( bnfll.Index.iloc[-2]+1, bnfll.Index.iloc[-2]+1+ self.taillen[0]))
            
            
            
            # where to bind
            self.id1reel += [[
                bnfll[ np.isin( bnfll.name, start[idxreel]) & ((bnfll.strand == '+')|(bnfll.strand == '.')) ].Index.tolist() ,
                bnfll[ np.isin( bnfll.name, start[idxreel]) & ((bnfll.strand == '-')|(bnfll.strand == '.')) ].Index.tolist()
                ]]
            
            # where not to move to
            self.idd1notReel += [[
                bnfll[ ~np.isin( bnfll.annot, reel[idxreel] ) ].Index.tolist() +tail1+tail2 ,
                bnfll[ ~np.isin( bnfll.annot, reel[idxreel] ) ].Index.tolist() +tail1+tail2
                ]]

            # where to disattach
            self.geneend += [[
                bnfll[ np.isin( bnfll.annot, end[idxreel] ) & ((bnfll.strand == '+')|(bnfll.strand == '.')) ].Index.tolist() ,
                bnfll[ np.isin( bnfll.annot, end[idxreel] ) & ((bnfll.strand == '-')|(bnfll.strand == '.')) ].Index.tolist() 
                ]]           

        
            
            




        # =============================================================================
        # CTCF settings
        # =============================================================================
        if len(self.specific_type) > 0:
            self.strSpec = True
        else:
            self.strSpec = False
            
        if self.strSpec:
            
            self.strSpecBond = 'harm' # 'harm', 'fene'
            
            self.kHarmSpec = 8
            self.r0HarmSpec = self.bond_length
            self.rcHarmSpec = self.bond_length * 3.5
            
            self.kfeneSpec = 30
            self.drmaxFeneSpec = .8
            self.r0feneSpec = self.bond_length
            self.bsspec_type = [19]
            self.molspec_type = [10,11]

            self.specMaxHSpring = 1
            self.specMaxHFBSpring = 1

            self.specDeathProb = .0002
            self.specBirthProb = .8
            self.spec_cutoff = self.bond_length * 1.8







        
        # =============================================================================
        # ### DIMERS
        # =============================================================================
        if len(self.anistypebond) > 0:
            self.strDimBond = True
        else:
            self.strDimBond = False 

        if self.strDimBond:
            self.dimLen = self.bond_length
            self.dimRot = (0, 0, 0) # (0, 0, 0) (1, 1, 1)
            # harm
            self.kHarmDimer= 8
            self.r0harmdimer = self.bond_length
            self.rcharmdimer = self.bond_length * 3.5
            # fene
            self.kfenedimer = 30
            self.r0fenedimer = self.bond_length
            self.drmaxfenedimer = .8
    










      
        ###========================================================================
        # WARM-UP settings
        ###========================================================================
        self.epsCapMax, self.epsCapMin = 3 , .1
        
        self.nnn, self.www, self.ttt = 40, 500, 10**5
        self.timeSteps = [ self.TIME_STEP, self.TIME_STEP / 2. ]
        
        
        self.nite = 10     
        
        self.wmpDic = {
            'SIG' : self.SIG,
            'LJCUT' : self.LJCUT,
            'epsCapMax' : self.epsCapMax,
            'epsCapMin' : self.epsCapMin,
            'nnn' : self.nnn,
            'www' : self.www,
            'ttt' : self.ttt,
            'Ehf' : self.Ehf,
            'Ehi' : self.Ehi,
            'Enbf' : self.Enbf,
            'Enbi' : self.Enbi,
            'alltypes' : self.alltypes,
            'allpairs' : self.allpairs,
            'pairsb' : self.pairsb,
            }
        
        
        self.wmpSaveObs = True # only, True, False
        
        
        self.wmpList = [
            {'step':'eq'} ,
            {'step':'pot'}
            ]        
        
        self.strKillv = True
        self.strKillF = False        
        

        self.ljh, self.ljb, self.wmp_sam, self.ljhcap = func.warmup( self.wmpDic)
        
        
        
        
            
            
            



        # =============================================================================
        #         ## dynamics settings
        # =============================================================================
        self.tsteps = []


        
        if self.str_tsam == 'log':
            # log sampling
            self.sampDic = {
                'mint' : self.mint ,
                'maxt' : self.maxt ,
                'tit' : self.tit ,
                'maxt2' : self.maxt2 ,
                }
            self.t = func.logSamp( self.sampDic)
            # t = np.int_( mint * (maxt/mint) ** ( np.arange(0, tit +1)/tit) ) + mint
            self.sampling_v = self.t[1:]-self.t[:-1]
            
            
        elif self.str_tsam == 'lin':
            # lin sampling 
            if self.maxt2 is False:       
                self.sampling_v = np.ones( self.tit, dtype='int' ) * np.int_( self.maxt / self.tit ) 
            else:
                self.sampling_v = np.ones( self.tit, dtype='int' ) * np.int_( (self.maxt2 - self.maxt) / self.tit ) 
            
            

            
            
        self.stage1 = {
                'sampling_v': self.sampling_v,
                'name':'dynamics 1'
                }
        
        self.stage1['potentials'] = {
            'ljb eps' : self.Enbf ,
            'ljb sig' : self.ljb['sig'][ :, :, -1] ,
            'ljb cut' : self.ljb['cut'][ :, :, -1] ,
            }
            
        self.tsteps += [self.stage1]
    
    

    
       
        
        
        
        
        
        # =============================================================================
        # Costraints
        # =============================================================================
        self.BOX_L = [self.b, self.b, self.b]
        self.beads_per_chain = self.npolall   
        
                
                
        
        
        self.strCos = True
        self.polf = func.polGen2
        self.cosDic = {
            'startPos' : np.array([self.b/2.,self.b/2.+self.bond_length,self.b/2.]), 
            'startPosi' : np.array([[self.b/2.,self.b/2.+.1,self.b/2.]]), 
            'boxl' : self.b
            }
        
        
        
        
        # =============================================================================
        #    I/O options
        # =============================================================================
        self.str_saveMode = 'pandas'
        self.str_write = 'custom' # 'vmd', 'h5md', 'custom', None, 'vmd+h5md'
        self.strVel = True
        self.savePrecision = 3
        
        
        # =============================================================================
        #     Run options
        # =============================================================================
        self.str_run = 'multithread' # multithread, sequential    
        
        
        
        print('Model class bpdl loaded')

            

