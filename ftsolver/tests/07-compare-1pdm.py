
import pickle, numpy as np
symm = 'UHF'
for r in np.arange(1.4, 4.0, 0.4):
    x = pickle.load(open('FTDMRG-H4-%.1f-%s/dm.tmp' % (r, symm), 'rb'))
    y = pickle.load(open('H4-%.1f-%s-dm.tmp' % (r, symm), 'rb'))
    assert np.allclose(x, y, atol=1E-5)
