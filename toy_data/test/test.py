import toy_data.color as c
import toy_data.sine_wave as swg
import toy_data.models as m
import toy_data.cov_shift as cov_shift
import toy_data.data_types as tp
import numpy as np
import toy_data.visualize as vs

from toy_data import gaussian_mixtures as gmg

COLOR = False
COV_SHIFT = False
MODEL = False
VISUALIZE = True

if COLOR:
    # print(c.get_N_by_hue(3))
    cs = c.map_color([1, 2, 3, 4], 100, (23, 123, 29))
    print(cs)

if False:
    x_dim = 2
    gm = gmg.GaussianMixture(n_class=3, dim=x_dim)
    gmg.visualize_2D(gm.Classes, gm.class_colors)

if False:
    sw = swg.SineWave()
    swg.visualize_1D_regression(sw)

if False:
    l = tp.DataSet.from_X([1, 2, 3, 4, 5], lambda x: [k + 1 for k in x])
    print(l.X)
    print(l.y)

if COV_SHIFT:
    the_model = m.Sine_1D()
    cs = cov_shift.Gaussian_Shift_1D(model=the_model)
    # cov_shift.visualize_1D_regression(cs)
    cov_shift.visualize_1D_regression_with_tr_weights(cs, cs.tr.X)
    the_model = m.LinearBinary(2)
    cs = cov_shift.Gaussian_Shift_2D_BinaryClassification(model=the_model)
    # cov_shift.visualize_2D_classification(cs, classifyF=lambda x: x[:, 0] + x[:, 1] > 0)
    cov_shift.visualize_2D_classification_with_tr_weights(cs,
                                                          np.ravel(np.random.random(cs.tr.y.shape)),
                                                          classifyF=lambda x: x[:, 0] + x[:, 1] > 0)
    # cov_shift.visualize_2D_classification_with_tr_weights(cs,
    #                                                      np.ravel(np.random.random(cs.tr.y.shape)))

if MODEL:
    the_model = m.rotatedSine2D()
    r = the_model(np.random.uniform(-1, 1, (10, 2)))
    print(r)
    cs = cov_shift.Gaussian_Shift_2D_BinaryClassification(model=the_model)
    # cov_shift.visualize_2D_classification(cs, classifyF=lambda x: x[:, 0] + x[:, 1] > 0)
    # cov_shift.visualize_2D_classification(cs, classifyF=the_model)

if VISUALIZE:
    x = np.linspace(0, 10, 100)
    d = tp.DataSet.from_X(x[:, np.newaxis], m.Sine_1D())
    vs.regression_1D(d)
