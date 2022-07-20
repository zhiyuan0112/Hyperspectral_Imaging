
from .admmn.admmn import ADMMN_ALPHA, ADMMN_BASE, ADMMN_SPECTRUM


"""Define commonly used architecture"""
# ============= Task 1 =============== #
def admmn_16channel_base():
    net = ADMMN_BASE(n_resblocks=4, n_admmblocks=9, in_channels=16, n_feats=256, n_convs=3)
    net.use_2dconv = True
    net.bandwise = False
    return net

def admmn_16channel_alpha():
    net = ADMMN_ALPHA(n_resblocks=4, n_admmblocks=9, in_channels=16, n_feats=256, n_convs=3)
    net.use_2dconv = True
    net.bandwise = False
    return net


# ============= Task 2 =============== #
def admmn_base():
    net = ADMMN_BASE(n_resblocks=4, n_admmblocks=9, in_channels=31, n_feats=256, n_convs=3)
    net.use_2dconv = True
    net.bandwise = False
    return net

def admmn_alpha():  # selection
    net = ADMMN_ALPHA(n_resblocks=4, n_admmblocks=9, in_channels=31, n_feats=256, n_convs=3)
    net.use_2dconv = True
    net.bandwise = False
    return net

def admmn_spectrum():  # design
    net = ADMMN_SPECTRUM(n_resblocks=4, n_admmblocks=9, in_channels=31, n_feats=256, n_convs=3)
    net.use_2dconv = True
    net.bandwise = False
    return net

