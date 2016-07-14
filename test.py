import gaussian_mixtures as gmg
import sine_wave as swg

if False:
    x_dim = 2
    gm = gmg.GaussianMixture(n_class=3, dim=x_dim)
    gmg.visualize_2D(gm.Classes, gm.class_colors)

sw = swg.SineWave()
swg.visualize_1D_regression(sw)
