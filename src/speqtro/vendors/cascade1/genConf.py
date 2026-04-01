#!/usr/bin/python
from __future__ import print_function, absolute_import

#######################################################################
# Molecular conformer generator in progress
# genConf.py -isdf file_input.sdf -osdf file_output.sdf
# -n number_of_conformers (optional, if not specified is based
# on the nomber of rotable bonds) -rtpre rms_threshold_pre_opt(optional)
# -rtpost rms_threshold_post_opt(optional) -e energy_window (optional, Kcal/mol)
# -t number_of_threads (if not specify 1)
#######################################################################

## known issues / to-do list
## logging doesn't work properly
## need to print out rotatable bond atoms for reference
## comparison of printing vs. old full monte - more options
## comparison of printing vs. GoodVibes
## tabulation of results in addition to sdf output file
## pythonify where possible
## currently RMS but torsion_list would be better?

from rdkit import Chem
from rdkit.Chem import AllChem
from concurrent import futures
import argparse, logging, os, sys, time, copy
logger = logging.getLogger(__name__)

# PHYSICAL CONSTANTS
GAS_CONSTANT, PLANCK_CONSTANT, BOLTZMANN_CONSTANT, SPEED_OF_LIGHT, AVOGADRO_CONSTANT, AMU_to_KG, atmos = 8.3144621, 6.62606957e-34, 1.3806488e-23, 2.99792458e10, 6.0221415e23, 1.66053886E-27, 101.325
# UNIT CONVERSION
j_to_au = 4.184 * 627.509541 * 1000.0

# version number
__version__ = "1.0.1"

# Formatted output to command line and log file
class Logger:
    # Designated initializer
    def __init__(self, filein, ext):
        # check to see if already exists
        if os.path.exists(filein+"."+ext):
            var = input("\no  Logger file %s already exists! OK to proceed? (Y/N) " % (filein+"."+ext))
            if var.lower().strip() == "n" or var.lower().strip() == "no":
                logger.error("\n   OK. Exiting gracefully ...\n"); sys.exit(1)
        # Create the log file at the input path
        self.log = open(filein+"."+ext, 'w' )

    # Write a message to the log
    def Write(self, message):
        # Print the message
        logger.info(message)
        # Write to log
        self.log.write(message + "\n")

    # Write a message only to the log and not to the terminal
    def Writeonlyfile(self, message):
        # Write to log
        self.log.write(message)

    # Write a fatal error, finalize and terminate the program
    def Fatal(self, message):
        # Print the message
        logger.error(message+"\n")
        # Write to log
        self.log.write(message + "\n")
        # Finalize the log
        self.Finalize()
        # End the program
        sys.exit(1)

    # Finalize the log file
    def Finalize(self):
        self.log.close()

dashedline = "   ------------------------------------------------------------------------------------------------------------------"
emptyline = "   |                                                                                                                     |"
normaltermination = "\n   -----------------       N   O   R   M   A   L      T   E   R   M   I   N   A   T   I   O   N      ----------------\n"
leftcol=97
rightcol=12

# algorithm to generate nc conformations
def genConf(m, nc, rms, efilter, rmspost):
    nr = int(AllChem.CalcNumRotatableBonds(m))
    #m = Chem.AddHs(m)
    Chem.AssignAtomChiralTagsFromStructure(m, replaceExistingTags=True)
    if not nc: nc = 3**nr

    if not rms: rms = -1
    ids=AllChem.EmbedMultipleConfs(m, numConfs=nc)

    if len(ids)== 0:
        ids = m.AddConformer(m.GetConformer, assignID=True)

    diz = []
    diz2 = []
    diz3 = []
    for id in ids:
        prop = AllChem.MMFFGetMoleculeProperties(m, mmffVariant="MMFF94s")
        ff = AllChem.MMFFGetMoleculeForceField(m, prop, confId=id)
        ff.Minimize()
        en = float(ff.CalcEnergy())
        econf = (en, id)
        diz.append(econf)

    if efilter != "Y":
        n, diz2 = energy_filter(m, diz, efilter)
    else:
        n = m
        diz2 = diz

    if rmspost != None and n.GetNumConformers() > 1:
        o, diz3 = postrmsd(n, diz2, rmspost)
    else:
        o = n
        diz3 = diz2

    return o, diz3, nr

# filter conformers based on relative energy
def energy_filter(m, diz, efilter):
    diz.sort()
    mini = float(diz[0][0])
    sup = mini + efilter
    n = Chem.Mol(m)
    n.RemoveAllConformers()
    n.AddConformer(m.GetConformer(int(diz[0][1])))
    nid = []
    ener = []
    nid.append(int(diz[0][1]))
    ener.append(float(diz[0][0])-mini)
    del diz[0]
    for x,y in diz:
        if x <= sup:
            n.AddConformer(m.GetConformer(int(y)))
            nid.append(int(y))
            ener.append(float(x-mini))
        else:
            break
    diz2 = list(zip(ener, nid))
    return n, diz2

# filter conformers based on geometric RMS
def postrmsd(n, diz2, rmspost):
    diz2.sort(key=lambda x: x[0])
    o = Chem.Mol(n)
    confidlist = [diz2[0][1]]
    enval = [diz2[0][0]]
    nh = Chem.RemoveHs(n)
    del diz2[0]
    for z,w in diz2:
        confid = int(w)
        p=0
        for conf2id in confidlist:
            rmsd = AllChem.GetBestRMS(nh, nh, prbId=confid, refId=conf2id)
            if rmsd < rmspost:
                p=p+1
                break
        if p == 0:
            confidlist.append(int(confid))
            enval.append(float(z))
    diz3 = list(zip(enval, confidlist))
    return o, diz3
