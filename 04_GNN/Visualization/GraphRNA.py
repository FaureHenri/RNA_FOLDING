import os
import RNA
import matplotlib.pyplot as plt
import forgi.visual.mplotlib as fvm
import forgi.graph.bulge_graph as fgb
import forgi
from os.path import join
from os import mkdir
from subprocess import call

# saved directory
dirname = input('Enter Toehold ID num: ')
dirname += '_Toehold'
dirpath = join('RNA_SS', dirname)
if not os.path.isdir(dirpath):
    mkdir(dirpath)

# The RNA sequence
raw_seq = input('Enter the 1D RNA sequence in ON state: ')
raw_seq = raw_seq.replace('T', 'U')

on_seq = raw_seq
off_seq = 'GGG' + raw_seq[33:]

off_ss = '...................................(((((((((...((((((...........))))))...)))))))))..(((.......(((((' \
         '.....)))))..))).... '
on_ss = '...((((((((((((((((((((((((((((((' \
        '....................))))))))))))))))))))))))))))))...............................(((.......(((((' \
        '.....)))))..))).... '

seq = None
ss = None
filename = None
c = 0
for ss_type in ['ideal', 'pred']:
    for switch in ['ON', 'OFF']:
        filename = ss_type + '_switch' + switch + '.png'
        if ss_type == 'ideal' and switch == 'ON':
            seq = on_seq
            ss = on_ss
        elif ss_type == 'ideal' and switch == 'OFF':
            seq = off_seq
            ss = off_ss
        elif ss_type == 'pred' and switch == 'ON':
            seq = on_seq
            (ss, mfe) = RNA.fold(seq)
        elif ss_type == 'pred' and switch == 'OFF':
            seq = off_seq
            (ss, mfe) = RNA.fold(seq)

        with open('rna_struc.txt', 'w') as outfile:
            outfile.write('>rna_struc\n')
            outfile.write(seq + '\n')
            outfile.write(ss)

        plt.figure(c)
        cg = forgi.load_rna('rna_struc.txt', allow_many=False)
        # bg = fgb.BulgeGraph.from_dotbracket(ss, seq)
        fvm.plot_rna(cg, text_kwargs={'fontweight': 'black', 'fontsize': 6}, lighten=0.5, backbone_kwargs={'linewidth': 1})
        filepath = join(dirpath, filename)
        plt.savefig(filepath, dpi=150)
        print(f'File saved to {filepath}')
        c += 1

        plt.figure(c)
        filename = 'el_' + filename
        filepath = join(dirpath, filename)
        call('python rnaConvert.py rna_struc.txt -T neato | neato -Tpng -o ' + filepath, shell=True)
        c += 1
