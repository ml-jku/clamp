import numpy as np
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.rdmolops import FastFindRings
from rdkit.Chem.Scaffolds import MurckoScaffold

# from https://bitsilla.com/blog/2021/06/standardizing-a-molecule-using-rdkit/
def standardize(smiles):
    # follows the steps in
    # https://github.com/greglandrum/RSC_OpenScience_Standardization_202104/blob/main/MolStandardize%20pieces.ipynb
    # as described **excellently** (by Greg) in
    # https://www.youtube.com/watch?v=eWTApNX8dJQ
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return mol
    except:
        return smiles
    
    # removeHs, disconnect metal atoms, normalize the molecule, reionize the molecule
    try:
        clean_mol = rdMolStandardize.Cleanup(mol)
    except:
        clean_mol = mol
        
    try:
        mol.UpdatePropertyCache(strict=False) #Correcting valence info # important operation
        Chem.GetSymmSSSR(mol) #get ring information
    except:
        pass
     
    # if many fragments, get the "parent" (the actual mol we are interested in)
    
    try: 
        parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)
        uncharger = rdMolStandardize.Uncharger() # annoying, but necessary as no convenience method exists
        uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)
    except: 
        uncharged_parent_clean_mol = clean_mol
    # note that no attempt is made at reionization at this step
    # nor at ionization at some pH (rdkit has no pKa caculator)
    # the main aim to to represent all molecules from different sources
    # in a (single) standard way, for use in ML, catalogue, etc.
    try:
        te = rdMolStandardize.TautomerEnumerator() # idem
        taut_uncharged_parent_clean_mol = te.Canonicalize(uncharged_parent_clean_mol)
    except:
        taut_uncharged_parent_clean_mol = uncharged_parent_clean_mol
    
    return Chem.MolToSmiles(taut_uncharged_parent_clean_mol)


def canonicalize(smi, remove_atom_mapping=True, sanitize=True, steps=2):
    r"""Force read a smiles or smarts and canonicalize it"""#
    if len(smi.split('.'))>=2:
        outs = []
        for part in smi.split('.'):
            out = canonicalize(part, remove_atom_mapping=remove_atom_mapping, sanitize=sanitize, steps=steps)
            if not out:
                return False
            outs.append(str(out))
        outs.sort()
        ret = '.'.join(outs)
        return ret

    m = Chem.MolFromSmiles(smi)
    #if m is None:
    #    m = Chem.MolFromSmarts(smi) #may break the kernel :S
    #if m is None:
    #    try:
    #        m = Chem.MolFromSmarts(AllChem.ReactionToSmiles(AllChem.ReactionFromSmarts(smi+'>>')).split('>>')[0].split('.')[0])
    #    except:
    #        pass
    if m is None:
        #print(smi, 'not parseable')
        return False
        #raise ValueError("Molecule not canonicalizable")
    if remove_atom_mapping:
        for atom in m.GetAtoms():
            if atom.HasProp("molAtomMapNumber"):
                atom.ClearProp("molAtomMapNumber")
    if sanitize:
        try:
            Chem.SanitizeMol(m, catchErrors=True)
            FastFindRings(m) #Providing ring info
            m.UpdatePropertyCache(strict=False) #Correcting valence info # important operation4
        except:
            pass
    can_smi = Chem.MolToSmiles(m)
    #while steps>=1:
    #    can_smi = canonicalize(can_smi, remove_atom_mapping=remove_atom_mapping, sanitize=sanitize, steps=steps-1)
    return can_smi #canoncialize
