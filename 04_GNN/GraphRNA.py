import RNA
import matplotlib.pyplot as plt
import forgi.visual.mplotlib as fvm
import forgi
import sys
from os.path import join

# Predicted vs Ideal
struc = input('Enter ''P'' for Predicted Secondary structure or ''I'' for Ideal Secondary structure: ')

switch = None
if struc == 'I':
    switch = input('Enter ''ON'' or ''OFF'' for the switch functionality: ')

# The RNA sequence
seq = input('Enter the whole 1D RNA sequence (with nucleobases within A-U/T-C-G: ')

ss = None
filename = None
if struc == 'I' and switch == 'OFF':
    filename = 'ideal_switchOFF'
    ss = '...................................(((((((((...((((((...........))))))...)))))))))..(((.......(((((.....)))))..)))....'
elif struc == 'I' and switch == 'ON':
    filename = 'ideal_switchON'
    ss = '...((((((((((((((((((((((((((((((....................))))))))))))))))))))))))))))))...............................(((.......(((((.....)))))..)))....'
elif struc == 'P':
    # compute minimum free energy (MFE) and corresponding secondary structure
    (ss, mfe) = RNA.fold(seq)
    if len(seq) == 148:
        filename = 'pred_switchON'
    elif len(seq) == 118:
        filename = 'pred_switchOFF'
    else:
        print(len(seq))
        sys.exit(1)

print(f'{ss}')
with open('rna_struc.fx', 'w') as outfile:
    outfile.write('>rna_struc\n')
    outfile.write(seq + '\n')
    outfile.write(ss)

cg = forgi.load_rna('rna_struc.fx', allow_many=False)
fvm.plot_rna(cg, text_kwargs={'fontweight': 'black', 'fontsize': 6}, lighten=0.5, backbone_kwargs={'linewidth': 1})
filepath = join('RNA_SS', filename)
plt.savefig(filepath, dpi=150)

print(f'File saved to {filepath}')
