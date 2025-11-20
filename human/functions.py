"""
Utility functions for TBMF (Transformer-Based MRI Parameter Estimation).

This module contains data loading, model creation, and evaluation utilities
for training and inference with Vision Transformer models.
"""

import glob
from torch.utils.data import Dataset, DataLoader
import os
import h5py
import torch
import numpy as np
from transformer_architecture_prod import *
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
from matplotlib import cm
from tqdm import tqdm
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torch.nn.functional import mse_loss
from torchvision import transforms
import scipy as sc
import torch.nn as nn

os.environ["NCCL_P2P_DISABLE"] = "1"


# ============================================================================
# COLOR MAP DATA
# ============================================================================

cm_data_vik = [[0.0013282, 0.069836, 0.37953],
               [0.0023664, 0.076475, 0.38352],
               [0.0033042, 0.083083, 0.38749],
               [0.0041459, 0.08959, 0.39148],
               [0.0048968, 0.095948, 0.39545],
               [0.0055632, 0.10227, 0.39941],
               [0.0061512, 0.1085, 0.40339],
               [0.0066676, 0.11469, 0.40734],
               [0.0071192, 0.12085, 0.41129],
               [0.0075116, 0.12696, 0.41523],
               [0.0078503, 0.13307, 0.41917],
               [0.0081413, 0.13909, 0.42308],
               [0.0083912, 0.14517, 0.42701],
               [0.0086057, 0.15114, 0.43091],
               [0.0087895, 0.15714, 0.43481],
               [0.0089466, 0.16315, 0.43869],
               [0.0090802, 0.16914, 0.44259],
               [0.0091935, 0.1751, 0.44646],
               [0.0092897, 0.18105, 0.45034],
               [0.0093719, 0.18705, 0.45421],
               [0.0094431, 0.19303, 0.45808],
               [0.0095061, 0.199, 0.46195],
               [0.0095639, 0.20501, 0.46582],
               [0.0096192, 0.21102, 0.46971],
               [0.0096751, 0.21705, 0.47357],
               [0.0097347, 0.22308, 0.47746],
               [0.0098019, 0.22912, 0.48135],
               [0.0098809, 0.23521, 0.48525],
               [0.0099771, 0.24128, 0.48916],
               [0.010098, 0.24739, 0.49308],
               [0.010254, 0.25352, 0.49702],
               [0.010463, 0.25968, 0.50097],
               [0.010755, 0.26585, 0.50494],
               [0.011176, 0.27204, 0.50893],
               [0.011716, 0.2783, 0.51292],
               [0.012286, 0.28455, 0.51695],
               [0.012934, 0.29087, 0.521],
               [0.01379, 0.29721, 0.52507],
               [0.014838, 0.30358, 0.52918],
               [0.016131, 0.31002, 0.53331],
               [0.017711, 0.31647, 0.53748],
               [0.01963, 0.32299, 0.54168],
               [0.021948, 0.32955, 0.54593],
               [0.02473, 0.33614, 0.55021],
               [0.028047, 0.34283, 0.55454],
               [0.03198, 0.34954, 0.55891],
               [0.036812, 0.35633, 0.56334],
               [0.042229, 0.36317, 0.56781],
               [0.048008, 0.37009, 0.57235],
               [0.054292, 0.37708, 0.57693],
               [0.060963, 0.38413, 0.58157],
               [0.068081, 0.39126, 0.58628],
               [0.075457, 0.39846, 0.59104],
               [0.083246, 0.40574, 0.59587],
               [0.091425, 0.41309, 0.60075],
               [0.099832, 0.4205, 0.6057],
               [0.10859, 0.428, 0.61071],
               [0.11769, 0.43557, 0.61577],
               [0.12704, 0.44319, 0.6209],
               [0.1367, 0.45089, 0.62606],
               [0.14661, 0.45864, 0.63129],
               [0.15679, 0.46646, 0.63656],
               [0.16719, 0.47432, 0.64187],
               [0.17781, 0.48224, 0.64722],
               [0.18861, 0.49019, 0.6526],
               [0.19958, 0.49819, 0.65802],
               [0.21078, 0.5062, 0.66346],
               [0.22212, 0.51426, 0.66892],
               [0.2336, 0.52232, 0.6744],
               [0.24523, 0.53041, 0.67989],
               [0.257, 0.53852, 0.68541],
               [0.26887, 0.54662, 0.69091],
               [0.2808, 0.55472, 0.69643],
               [0.29285, 0.56282, 0.70193],
               [0.30498, 0.57091, 0.70745],
               [0.31717, 0.579, 0.71295],
               [0.32944, 0.58706, 0.71845],
               [0.34173, 0.59512, 0.72393],
               [0.35407, 0.60316, 0.72941],
               [0.36646, 0.61119, 0.73488],
               [0.37886, 0.61919, 0.74032],
               [0.39131, 0.62716, 0.74576],
               [0.40376, 0.63511, 0.75118],
               [0.41623, 0.64305, 0.75658],
               [0.42871, 0.65096, 0.76197],
               [0.4412, 0.65884, 0.76734],
               [0.4537, 0.6667, 0.7727],
               [0.4662, 0.67454, 0.77804],
               [0.4787, 0.68235, 0.78337],
               [0.49121, 0.69014, 0.78868],
               [0.50369, 0.69791, 0.79398],
               [0.51618, 0.70566, 0.79926],
               [0.52868, 0.71339, 0.80452],
               [0.54115, 0.72109, 0.80977],
               [0.55362, 0.72878, 0.81501],
               [0.5661, 0.73644, 0.82023],
               [0.57856, 0.74409, 0.82544],
               [0.59101, 0.75172, 0.83063],
               [0.60347, 0.75931, 0.83579],
               [0.61591, 0.7669, 0.84094],
               [0.62835, 0.77445, 0.84606],
               [0.64078, 0.78199, 0.85115],
               [0.6532, 0.78949, 0.85621],
               [0.66563, 0.79694, 0.86121],
               [0.67805, 0.80437, 0.86617],
               [0.69046, 0.81174, 0.87106],
               [0.70287, 0.81905, 0.87587],
               [0.71526, 0.82629, 0.88057],
               [0.72765, 0.83344, 0.88515],
               [0.74002, 0.84048, 0.88957],
               [0.75235, 0.84738, 0.89381],
               [0.76466, 0.85413, 0.89782],
               [0.77692, 0.86068, 0.90156],
               [0.7891, 0.86699, 0.90499],
               [0.80117, 0.87303, 0.90804],
               [0.81311, 0.87874, 0.91065],
               [0.82487, 0.88406, 0.91276],
               [0.8364, 0.88893, 0.9143],
               [0.84762, 0.89329, 0.9152],
               [0.85847, 0.89707, 0.91539],
               [0.86887, 0.90021, 0.91481],
               [0.87873, 0.90264, 0.91342],
               [0.88797, 0.9043, 0.91116],
               [0.8965, 0.90518, 0.90803],
               [0.90424, 0.90522, 0.90401],
               [0.91115, 0.90442, 0.89913],
               [0.91718, 0.9028, 0.89341],
               [0.92229, 0.90037, 0.88691],
               [0.92648, 0.89717, 0.87969],
               [0.92979, 0.89326, 0.87183],
               [0.93224, 0.8887, 0.8634],
               [0.93388, 0.88355, 0.85448],
               [0.93478, 0.87789, 0.84515],
               [0.93501, 0.8718, 0.83549],
               [0.93464, 0.86531, 0.82556],
               [0.93375, 0.85852, 0.81542],
               [0.93241, 0.85147, 0.80511],
               [0.93068, 0.84421, 0.79468],
               [0.92862, 0.83678, 0.78417],
               [0.9263, 0.82921, 0.77358],
               [0.92375, 0.82154, 0.76296],
               [0.92102, 0.8138, 0.75231],
               [0.91815, 0.806, 0.74166],
               [0.91516, 0.79816, 0.73101],
               [0.91208, 0.79029, 0.72037],
               [0.90893, 0.78242, 0.70975],
               [0.90574, 0.77454, 0.69915],
               [0.90251, 0.76667, 0.68859],
               [0.89925, 0.75881, 0.67805],
               [0.89597, 0.75097, 0.66755],
               [0.89269, 0.74315, 0.65709],
               [0.8894, 0.73535, 0.64666],
               [0.88612, 0.72757, 0.63627],
               [0.88283, 0.71983, 0.62592],
               [0.87956, 0.71211, 0.61562],
               [0.87629, 0.70442, 0.60536],
               [0.87303, 0.69676, 0.59514],
               [0.86978, 0.68914, 0.58497],
               [0.86655, 0.68154, 0.57483],
               [0.86333, 0.67398, 0.56475],
               [0.86012, 0.66645, 0.55471],
               [0.85692, 0.65896, 0.54471],
               [0.85373, 0.6515, 0.53475],
               [0.85056, 0.64406, 0.52484],
               [0.8474, 0.63667, 0.51497],
               [0.84426, 0.6293, 0.50515],
               [0.84113, 0.62196, 0.49537],
               [0.838, 0.61465, 0.48563],
               [0.83489, 0.60739, 0.47594],
               [0.8318, 0.60014, 0.46628],
               [0.82872, 0.59294, 0.45668],
               [0.82564, 0.58576, 0.44711],
               [0.82258, 0.5786, 0.43759],
               [0.81953, 0.57148, 0.42811],
               [0.8165, 0.56439, 0.41866],
               [0.81346, 0.55733, 0.40926],
               [0.81045, 0.55028, 0.39989],
               [0.80744, 0.54327, 0.39058],
               [0.80445, 0.53629, 0.3813],
               [0.80145, 0.52933, 0.37204],
               [0.79847, 0.52238, 0.36284],
               [0.7955, 0.51546, 0.35366],
               [0.79253, 0.50857, 0.34452],
               [0.78957, 0.50169, 0.33544],
               [0.78662, 0.49483, 0.32634],
               [0.78366, 0.48798, 0.31731],
               [0.7807, 0.48112, 0.3083],
               [0.77774, 0.4743, 0.29933],
               [0.77476, 0.46746, 0.29035],
               [0.77179, 0.46062, 0.28142],
               [0.76879, 0.45378, 0.27251],
               [0.76578, 0.44693, 0.26364],
               [0.76272, 0.44005, 0.25476],
               [0.75964, 0.43315, 0.24587],
               [0.75651, 0.4262, 0.23705],
               [0.75332, 0.41922, 0.22819],
               [0.75005, 0.41216, 0.21933],
               [0.7467, 0.40503, 0.21047],
               [0.74324, 0.39782, 0.20159],
               [0.73965, 0.39049, 0.19274],
               [0.7359, 0.38306, 0.18385],
               [0.73199, 0.37547, 0.17498],
               [0.72786, 0.36774, 0.16605],
               [0.72352, 0.35985, 0.15713],
               [0.71891, 0.35177, 0.14821],
               [0.71403, 0.3435, 0.13928],
               [0.70884, 0.33505, 0.13046],
               [0.70332, 0.32635, 0.12154],
               [0.69745, 0.3175, 0.11284],
               [0.69123, 0.30846, 0.10413],
               [0.68465, 0.29926, 0.095633],
               [0.67773, 0.28992, 0.08735],
               [0.67048, 0.28048, 0.079197],
               [0.6629, 0.27102, 0.07151],
               [0.65505, 0.26152, 0.064079],
               [0.64697, 0.25208, 0.057104],
               [0.63869, 0.24271, 0.050618],
               [0.63026, 0.23349, 0.04475],
               [0.62172, 0.22445, 0.039414],
               [0.61313, 0.21566, 0.034829],
               [0.60454, 0.20709, 0.031072],
               [0.59595, 0.19874, 0.028212],
               [0.5874, 0.1907, 0.026019],
               [0.57894, 0.18292, 0.024396],
               [0.57054, 0.17542, 0.023257],
               [0.56227, 0.16817, 0.022523],
               [0.55408, 0.1612, 0.02211],
               [0.54601, 0.1544, 0.021861],
               [0.53804, 0.14785, 0.021737],
               [0.53018, 0.14149, 0.021722],
               [0.52242, 0.13528, 0.0218],
               [0.51478, 0.12921, 0.021957],
               [0.50721, 0.12327, 0.022179],
               [0.49973, 0.11749, 0.022455],
               [0.49235, 0.11182, 0.022775],
               [0.48503, 0.10621, 0.02313],
               [0.4778, 0.10061, 0.023513],
               [0.47064, 0.095156, 0.023916],
               [0.46353, 0.089668, 0.024336],
               [0.45649, 0.084258, 0.024766],
               [0.44952, 0.078741, 0.025203],
               [0.4426, 0.073404, 0.025644],
               [0.43574, 0.067904, 0.026084],
               [0.42892, 0.062415, 0.026522],
               [0.42215, 0.056832, 0.026954],
               [0.41544, 0.051116, 0.027378],
               [0.40877, 0.045352, 0.02779],
               [0.40213, 0.039448, 0.028189],
               [0.39556, 0.033385, 0.02857],
               [0.38902, 0.027844, 0.028932],
               [0.3825, 0.022586, 0.029271],
               [0.37603, 0.017608, 0.029583],
               [0.36958, 0.01289, 0.029866],
               [0.36316, 0.0082428, 0.030115],
               [0.35679, 0.0040345, 0.030327],
               [0.35042, 6.1141e-05, 0.030499]]

cm_data_brok = [[0.17081, 0.10027, 0.29985],
                [0.17075, 0.10634, 0.30558],
                [0.17061, 0.11229, 0.3113],
                [0.1704, 0.11821, 0.31705],
                [0.17013, 0.12404, 0.32282],
                [0.16981, 0.12995, 0.3286],
                [0.16943, 0.13572, 0.33441],
                [0.169, 0.14158, 0.34021],
                [0.1685, 0.14739, 0.34603],
                [0.16797, 0.15317, 0.35187],
                [0.16745, 0.159, 0.35775],
                [0.16688, 0.1648, 0.36361],
                [0.16626, 0.17063, 0.36951],
                [0.16562, 0.17645, 0.37541],
                [0.16499, 0.18231, 0.38135],
                [0.16438, 0.18822, 0.38728],
                [0.16373, 0.19411, 0.39324],
                [0.16306, 0.19999, 0.39922],
                [0.16241, 0.20596, 0.40522],
                [0.16178, 0.21193, 0.41124],
                [0.16114, 0.21792, 0.41729],
                [0.1605, 0.22393, 0.42333],
                [0.15991, 0.22999, 0.42941],
                [0.1594, 0.23611, 0.43551],
                [0.15895, 0.24223, 0.44162],
                [0.15856, 0.24842, 0.44774],
                [0.15825, 0.25463, 0.45388],
                [0.15805, 0.26087, 0.46004],
                [0.15799, 0.26715, 0.4662],
                [0.1581, 0.27351, 0.47238],
                [0.15841, 0.27989, 0.47855],
                [0.15892, 0.28631, 0.48472],
                [0.15966, 0.2928, 0.4909],
                [0.16075, 0.29934, 0.49703],
                [0.16216, 0.30592, 0.50316],
                [0.16391, 0.31252, 0.50927],
                [0.16599, 0.31919, 0.51534],
                [0.16854, 0.32588, 0.52137],
                [0.17149, 0.33265, 0.52735],
                [0.1749, 0.33942, 0.53327],
                [0.17874, 0.3462, 0.53913],
                [0.183, 0.35303, 0.54493],
                [0.18776, 0.35987, 0.55065],
                [0.19288, 0.36673, 0.55629],
                [0.19842, 0.37358, 0.56186],
                [0.20439, 0.38045, 0.56734],
                [0.21065, 0.3873, 0.57274],
                [0.21727, 0.39416, 0.57805],
                [0.22416, 0.40103, 0.5833],
                [0.23136, 0.40786, 0.58847],
                [0.23875, 0.4147, 0.59358],
                [0.24637, 0.42151, 0.59862],
                [0.25418, 0.42834, 0.60361],
                [0.26213, 0.43515, 0.60855],
                [0.27024, 0.44195, 0.61342],
                [0.27844, 0.44874, 0.61828],
                [0.28674, 0.45553, 0.6231],
                [0.29515, 0.46231, 0.62789],
                [0.30363, 0.4691, 0.63266],
                [0.31216, 0.47588, 0.63742],
                [0.32079, 0.48266, 0.64216],
                [0.32944, 0.48945, 0.64689],
                [0.33811, 0.49624, 0.65162],
                [0.34685, 0.50304, 0.65634],
                [0.35562, 0.50985, 0.66106],
                [0.3644, 0.51666, 0.66578],
                [0.37322, 0.52348, 0.67051],
                [0.38206, 0.5303, 0.67522],
                [0.39094, 0.53714, 0.67995],
                [0.39982, 0.54397, 0.68468],
                [0.40874, 0.55082, 0.68943],
                [0.41768, 0.55768, 0.69416],
                [0.42663, 0.56454, 0.69891],
                [0.4356, 0.57141, 0.70367],
                [0.44459, 0.57829, 0.70843],
                [0.45359, 0.58519, 0.71319],
                [0.46261, 0.59209, 0.71796],
                [0.47166, 0.59899, 0.72273],
                [0.4807, 0.6059, 0.72751],
                [0.48977, 0.61282, 0.73231],
                [0.49887, 0.61976, 0.7371],
                [0.50797, 0.6267, 0.74189],
                [0.51709, 0.63365, 0.7467],
                [0.52622, 0.6406, 0.75151],
                [0.53536, 0.64757, 0.75633],
                [0.54452, 0.65454, 0.76115],
                [0.55369, 0.66152, 0.76598],
                [0.56289, 0.66851, 0.77081],
                [0.57208, 0.67551, 0.77565],
                [0.5813, 0.68251, 0.7805],
                [0.59052, 0.68954, 0.78535],
                [0.59977, 0.69656, 0.79021],
                [0.60904, 0.70359, 0.79507],
                [0.6183, 0.71063, 0.79995],
                [0.62758, 0.71768, 0.80482],
                [0.63689, 0.72474, 0.8097],
                [0.64619, 0.73181, 0.81459],
                [0.65552, 0.73889, 0.81948],
                [0.66485, 0.74597, 0.82438],
                [0.67421, 0.75305, 0.82929],
                [0.68357, 0.76016, 0.8342],
                [0.69295, 0.76727, 0.83911],
                [0.70233, 0.77438, 0.84402],
                [0.71174, 0.78151, 0.84894],
                [0.72114, 0.78864, 0.85385],
                [0.73057, 0.79577, 0.85877],
                [0.74, 0.80291, 0.86368],
                [0.74943, 0.81005, 0.86857],
                [0.75888, 0.81719, 0.87343],
                [0.76832, 0.82432, 0.87828],
                [0.77776, 0.83145, 0.88308],
                [0.7872, 0.83856, 0.88783],
                [0.79662, 0.84564, 0.89251],
                [0.80602, 0.8527, 0.8971],
                [0.81538, 0.85971, 0.90156],
                [0.82469, 0.86666, 0.90589],
                [0.83394, 0.87352, 0.91001],
                [0.84309, 0.88029, 0.91391],
                [0.85212, 0.88692, 0.91752],
                [0.86097, 0.89338, 0.92078],
                [0.86962, 0.89963, 0.92365],
                [0.87801, 0.90562, 0.92603],
                [0.88606, 0.91129, 0.92785],
                [0.8937, 0.91659, 0.92905],
                [0.90086, 0.92145, 0.92956],
                [0.90746, 0.92581, 0.92931],
                [0.9134, 0.9296, 0.92825],
                [0.91862, 0.93277, 0.92635],
                [0.92307, 0.93528, 0.92359],
                [0.92667, 0.9371, 0.91997],
                [0.92943, 0.93822, 0.91554],
                [0.93133, 0.93863, 0.91034],
                [0.93239, 0.93838, 0.90443],
                [0.93266, 0.93749, 0.8979],
                [0.93219, 0.93601, 0.89081],
                [0.93105, 0.93401, 0.88326],
                [0.9293, 0.93154, 0.87532],
                [0.92702, 0.92866, 0.86706],
                [0.9243, 0.92545, 0.85854],
                [0.92119, 0.92194, 0.84982],
                [0.91776, 0.91818, 0.84094],
                [0.91406, 0.91422, 0.83194],
                [0.91015, 0.91011, 0.82285],
                [0.90606, 0.90586, 0.81368],
                [0.90184, 0.90151, 0.80447],
                [0.89751, 0.89708, 0.79521],
                [0.89309, 0.89259, 0.78594],
                [0.88861, 0.88805, 0.77664],
                [0.88407, 0.88347, 0.76733],
                [0.87949, 0.87886, 0.75803],
                [0.87489, 0.87423, 0.74871],
                [0.87026, 0.86958, 0.7394],
                [0.86561, 0.86491, 0.73009],
                [0.86094, 0.86024, 0.72077],
                [0.85625, 0.85553, 0.71146],
                [0.85153, 0.8508, 0.70213],
                [0.84679, 0.84606, 0.69282],
                [0.84202, 0.84128, 0.68348],
                [0.8372, 0.83645, 0.67414],
                [0.83232, 0.83158, 0.66478],
                [0.82739, 0.82665, 0.65542],
                [0.82238, 0.82164, 0.64602],
                [0.81728, 0.81655, 0.63661],
                [0.81208, 0.81135, 0.62715],
                [0.80676, 0.80603, 0.61766],
                [0.80129, 0.80058, 0.60815],
                [0.79568, 0.79499, 0.59859],
                [0.7899, 0.78922, 0.589],
                [0.78394, 0.78328, 0.5794],
                [0.77778, 0.77715, 0.56976],
                [0.77143, 0.77082, 0.56013],
                [0.76486, 0.76428, 0.5505],
                [0.75809, 0.75754, 0.54089],
                [0.75112, 0.75061, 0.53133],
                [0.74395, 0.74348, 0.52183],
                [0.73661, 0.73616, 0.51242],
                [0.72909, 0.72869, 0.50312],
                [0.72143, 0.72106, 0.49397],
                [0.71365, 0.71331, 0.48495],
                [0.70576, 0.70546, 0.47611],
                [0.69778, 0.69751, 0.46744],
                [0.68975, 0.68951, 0.45895],
                [0.68166, 0.68144, 0.45065],
                [0.67355, 0.67336, 0.44254],
                [0.66542, 0.66525, 0.4346],
                [0.65729, 0.65714, 0.42684],
                [0.64917, 0.64903, 0.41924],
                [0.64105, 0.64093, 0.41179],
                [0.63296, 0.63285, 0.40447],
                [0.62488, 0.62478, 0.39727],
                [0.61684, 0.61674, 0.39019],
                [0.60881, 0.60873, 0.38321],
                [0.6008, 0.60072, 0.3763],
                [0.59283, 0.59275, 0.36945],
                [0.58488, 0.5848, 0.36268],
                [0.57695, 0.57687, 0.35596],
                [0.56904, 0.56896, 0.34928],
                [0.56116, 0.56108, 0.34265],
                [0.55329, 0.55322, 0.33604],
                [0.54546, 0.54538, 0.32949],
                [0.53765, 0.53757, 0.32294],
                [0.52985, 0.52977, 0.31642],
                [0.52207, 0.52198, 0.30993],
                [0.51433, 0.51424, 0.30344],
                [0.50659, 0.5065, 0.29701],
                [0.49888, 0.49879, 0.29056],
                [0.4912, 0.49111, 0.28415],
                [0.48352, 0.48342, 0.27777],
                [0.47588, 0.47578, 0.27139],
                [0.46825, 0.46814, 0.26503],
                [0.46064, 0.46053, 0.2587],
                [0.45307, 0.45296, 0.25239],
                [0.4455, 0.44539, 0.24608],
                [0.43798, 0.43786, 0.23981],
                [0.43046, 0.43033, 0.23354],
                [0.42296, 0.42283, 0.22732],
                [0.4155, 0.41537, 0.22111],
                [0.40805, 0.40792, 0.21489],
                [0.40065, 0.4005, 0.20872],
                [0.39323, 0.39309, 0.20257],
                [0.38586, 0.38571, 0.19647],
                [0.37851, 0.37836, 0.19034],
                [0.37118, 0.37103, 0.18426],
                [0.36389, 0.36373, 0.17824],
                [0.35663, 0.35647, 0.17218],
                [0.34937, 0.34922, 0.16619],
                [0.34215, 0.342, 0.16021],
                [0.33497, 0.33482, 0.1543],
                [0.32778, 0.32764, 0.1484],
                [0.32065, 0.32052, 0.14258],
                [0.31352, 0.31341, 0.13676],
                [0.30644, 0.30635, 0.13103],
                [0.29937, 0.29931, 0.12531],
                [0.29233, 0.2923, 0.11965],
                [0.28532, 0.28532, 0.11406],
                [0.27836, 0.27841, 0.10854],
                [0.2714, 0.27151, 0.10314],
                [0.26449, 0.26467, 0.097785],
                [0.25761, 0.25787, 0.092474],
                [0.25076, 0.25114, 0.087347],
                [0.24394, 0.24442, 0.082262],
                [0.2372, 0.2378, 0.077192],
                [0.23042, 0.2312, 0.072349],
                [0.22373, 0.22463, 0.067628],
                [0.21709, 0.21818, 0.063025],
                [0.21046, 0.21177, 0.058419],
                [0.20394, 0.20539, 0.053781],
                [0.19749, 0.19907, 0.048969],
                [0.19115, 0.19284, 0.043546],
                [0.18494, 0.18665, 0.037943],
                [0.17882, 0.18051, 0.031907],
                [0.1728, 0.17444, 0.026044],
                [0.16695, 0.1684, 0.020145],
                [0.16119, 0.16243, 0.014202],
                [0.15554, 0.15649, 0.0080014],
                [0.15005, 0.15056, 0.0019303]]


# ============================================================================
# DATA LOADING
# ============================================================================

class Load_Dataset(Dataset):
    """Dataset loader for HDF5 files with proper normalization."""

    def __init__(self, config):
        """
        Initialize dataset.

        Args:
            config (dict): Configuration dictionary containing:
                - 'data_paths': path to data directory
                - 'scale_data': scaling factor for data
                - 'scale_params': scaling factor for parameters
        """
        self.config = config
        self.data_folder = self.config["data_paths"]

        self.data_paths = sorted(glob.glob(os.path.join(self.data_folder, "*/dataset/*.h5")))
        self.param_paths = sorted(glob.glob(os.path.join(self.data_folder, "*/params/*.h5")))
        self.label_paths = sorted(glob.glob(os.path.join(self.data_folder, "*/labels/*.h5")))

        self.scale_data = self.config["scale_data"]
        self.scale_params = self.config["scale_params"]

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index: int):
        dataset_idx = h5py.File(self.data_paths[index])['res'][:] / self.scale_data
        params_idx = h5py.File(self.param_paths[index])['res'][:] / self.scale_params
        labels = h5py.File(self.label_paths[index])['res'][:]

        labels[0, :, :] = labels[0, :, :] / 100  # KSSW
        labels[1, :, :] = labels[1, :, :] * 100 / 27.27  # MT
        labels[2, :, :] = (labels[2, :, :] + 1) / (1.7 + 1)  # B0
        labels[3, :, :] = labels[3, :, :] / 3.4944  # B1
        labels[4, :, :] = labels[4, :, :] / 10000  # T1
        labels[5, :, :] = labels[5, :, :] / 1000  # T2

        return (torch.from_numpy(dataset_idx.astype(np.float32)),
                torch.from_numpy(params_idx.astype(np.float32)),
                torch.from_numpy(labels.astype(np.float32)))


# ============================================================================
# MODEL CREATION
# ============================================================================

def create_model_v0(args, weights_path):
    """
    Create Vision Transformer model with specific architecture.

    Args:
        args (dict): Model hyperparameters containing:
            - 'image_size': input image size
            - 'sequence_len': number of input channels
            - 'patch_size': patch size
            - 'embedding_dim': embedding dimension
            - 'dropout': dropout rate
            - 'mlp_size': MLP size
            - 'num_transformer_layers': number of transformer layers
            - 'num_heads': number of attention heads
            - 'device': torch device
        weights_path (str): Path to model weights

    Returns:
        Model: Trained model instance
    """
    model = Model(img_size=args["image_size"],
                  num_channels=args["sequence_len"],
                  patch_size=args["patch_size"],
                  embedding_dim=args["embedding_dim"],
                  dropout=args["dropout"],
                  mlp_size=args["mlp_size"],
                  num_transformer_layers=args["num_transformer_layers"],
                  num_heads=args["num_heads"])

    model.conv_layers[9] = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
    model.conv_layers[10] = nn.BatchNorm2d(num_features=32)
    model.conv_layers.append(nn.ReLU())

    model.conv_layers.append(nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1))
    model.conv_layers.append(nn.BatchNorm2d(num_features=16))
    model.conv_layers.append(nn.ReLU())

    model.conv_layers.append(nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1))
    model.conv_layers.append(nn.BatchNorm2d(num_features=8))
    model.conv_layers.append(nn.ReLU())

    model.conv_layers.append(nn.Conv2d(in_channels=8, out_channels=6, kernel_size=3, stride=1, padding=1))
    model.conv_layers.append(nn.Sigmoid())

    try:
        model.load_state_dict(torch.load(weights_path, map_location=args["device"]))
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        raise

    return model


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def pad_images(matrix, output_shape, size_idx):
    """
    Pad images to output shape.

    Args:
        matrix (np.array): Input image matrix
        output_shape (tuple): Desired output shape (height, width)
        size_idx (tuple): Indices of height and width dimensions

    Returns:
        np.array: Padded matrix
    """
    orig_height, orig_width = matrix.shape[size_idx[0]], matrix.shape[size_idx[1]]

    pad_height = output_shape[0] - orig_height
    pad_width = output_shape[1] - orig_width

    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    if len(matrix.shape) == 3:
        pad_width = [(0, 0), (0, 0), (0, 0)]
    else:
        pad_width = [(0, 0), (0, 0), (0, 0), (0, 0)]
    pad_width[size_idx[0]] = (pad_top, pad_bottom)
    pad_width[size_idx[1]] = (pad_left, pad_right)

    padded_images = np.pad(matrix, tuple(pad_width), mode='constant', constant_values=np.nan)

    return padded_images


def calc_psnr(y_pred, y_true, device=None):
    """Calculate PSNR metric."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    psnr = PeakSignalNoiseRatio(data_range=1).to(device=device)
    slices = y_pred.shape[1]
    if slices == 6:
        psnr_val = 0
        for i in range(slices):
            psnr_val += psnr(y_pred[:, i, :, :].unsqueeze(1),
                             y_true[:, i, :, :].unsqueeze(1))
        psnr_val = psnr_val / slices
    else:  # 30 slices
        psnr_val = 0
        for i in range(24):
            psnr_val += psnr(y_pred[:, i + 6, :, :].unsqueeze(1),
                             y_true[:, i + 6, :, :].unsqueeze(1))
        psnr_val = psnr_val / 24

    return psnr_val


def calc_nrmse(y_pred, y_true):
    """Calculate NRMSE metric."""
    slices = y_pred.shape[1]
    if slices == 6:
        nrmse_val = 0
        for i in range(slices):
            y_pred_idx = y_pred[:, i, :, :].unsqueeze(1)
            y_true_idx = y_true[:, i, :, :].unsqueeze(1)
            mse = mse_loss(y_pred_idx, y_true_idx)
            rmse = torch.sqrt(mse)
            nrmse_val += rmse / (torch.max(y_true_idx) - torch.min(y_true_idx))
        nrmse_val = nrmse_val / slices
    else:
        nrmse_val = 0
        for i in range(24):
            y_pred_idx = y_pred[:, i + 6, :, :].unsqueeze(1)
            y_true_idx = y_true[:, i + 6, :, :].unsqueeze(1)
            mse = mse_loss(y_pred_idx, y_true_idx)
            rmse = torch.sqrt(mse)
            nrmse_val += rmse / (torch.max(y_true_idx) - torch.min(y_true_idx))
        nrmse_val = nrmse_val / 24

    return nrmse_val


def calc_ssim(y_pred, y_true, device=None):
    """Calculate SSIM metric."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ssim_fn = StructuralSimilarityIndexMeasure().to(device)
    slices = y_pred.shape[1]
    if slices == 6:
        ssim_val = 0
        for i in range(slices):
            ssim_val += ssim_fn(y_pred[:, i, :, :].unsqueeze(1),
                                y_true[:, i, :, :].unsqueeze(1))
        ssim_val = ssim_val / 6
    else:
        ssim_val = 0
        for i in range(24):
            ssim_val += ssim_fn(y_pred[:, i + 6, :, :].unsqueeze(1),
                                y_true[:, i + 6, :, :].unsqueeze(1))
        ssim_val = ssim_val / 24

    return ssim_val


# ============================================================================
# METRICS AND VISUALIZATION
# ============================================================================

class Plot_Metrics():
    """Generate plots and metrics for model predictions."""

    def __init__(self, config):
        """
        Initialize metrics plotter.

        Args:
            config (dict): Configuration with:
                - 'dataset': DataLoader
                - 'model': PyTorch model
                - 'out_path': output directory
                - 'device': torch device
                - 'plot_specific_seq': sequence to plot
                - 'wanted_output': dict with plotting flags
        """
        self.config = config
        self.dataset = config["dataset"]
        self.model = config['model']
        self.out_path_images = os.path.join(config["out_path"], "images")
        self.out_path_metrics = os.path.join(config["out_path"], "metrics")
        self.device = config["device"]
        self.seq = config["plot_specific_seq"]
        os.makedirs(self.out_path_images, exist_ok=True)
        os.makedirs(self.out_path_metrics, exist_ok=True)

        self.psnr = PeakSignalNoiseRatio(data_range=1).to(self.device)
        self.ssim = StructuralSimilarityIndexMeasure().to(self.device)
        self.channel_names = ['Kssw', 'MT_perc', 'B0', 'B1', 'T1', 'T2']

        # Initialize metrics storage
        self.metrics = {'batch': [], 'ssim': [], 'psnr': [], 'nrmse': []}
        for i in range(len(self.channel_names)):
            self.metrics.update({
                f'ssim_metric_{i}': [], f'psnr_metric_{i}': [], f'nrmse_metric_{i}': []
            })

    def _create_colormaps(self):
        """Create and return colormaps for visualization."""
        buda_map = LinearSegmentedColormap.from_list('buda', cm_data_vik)
        lipari_map = LinearSegmentedColormap.from_list('lipari', cm_data_brok)

        maps = {}
        for name, cmap in [('viridis', plt.get_cmap('viridis')), ('buda', buda_map),
                           ('lipari', lipari_map), ('winter', plt.get_cmap('winter'))]:
            color_mat = cmap(np.arange(cmap.N))
            color_mat[0, 0:3] = 0
            maps[f'b_{name}'] = mcolors.LinearSegmentedColormap.from_list('colormap', color_mat)

        return maps

    def _plot_true_and_pred_sequences(self, test_labels, predicted_labels, save_path):
        """Plot ground truth and predicted parameter maps."""
        pred, true = predicted_labels.cpu().detach().numpy(), test_labels.cpu().detach().numpy()
        maps = self._create_colormaps()

        fig, axs = plt.subplots(nrows=2, ncols=6, figsize=(20, 10))

        # Define parameters for each channel
        channels = [
            (100, 0, 100, 'magma'), (27.27, 0, 27.27, maps['b_viridis']),
            (2.7, -0.6, 0.6, maps['b_buda']), (3.4944, 0.5, 1.5, maps['b_lipari']),
            (10000, 0, 3000, 'hot'), (1000, 0, 200, maps['b_winter'])
        ]

        titles = ["Kssw", "MT_perc", u'B\u2080', u'B\u2081', u'T\u2081', u'T\u2082']

        for i, (scale, vmin, vmax, cmap) in enumerate(channels):
            # Ground truth
            img = true[0, i, :, :] * scale
            if i == 2:  # B0 special handling
                img = img - 1
                img = img.copy()
                img[img == 0] = np.nan
                cmap = cmap.copy()
                cmap.set_bad('black')

            axs[0, i].imshow(img, vmin=vmin, vmax=vmax, cmap=cmap)
            axs[0, i].set_title(titles[i], fontsize=20)
            axs[0, i].axis('off')

            # Predictions
            pred_img = pred[0, :, :, i] * scale
            if i == 2:
                pred_img = pred_img - 1
            axs[1, i].imshow(pred_img, vmin=vmin, vmax=vmax, cmap=cmap)
            axs[1, i].axis('off')

        # Set labels
        axs[0, 0].set_ylabel("Ground\nTruth", rotation=0, labelpad=50, fontsize=20)
        axs[1, 0].set_ylabel("Predicted\nImages", rotation=0, labelpad=50, fontsize=20)
        for ax in [axs[0, 0], axs[1, 0]]:
            ax.set_yticks([])
            ax.set_xticks([])

        # Adjust layout
        shape_adjustments = {
            (116, 116): -0.65, (116, 88): -0.5, (126, 88): -0.4,
            (144, 88): -0.5, (126, 144): -0.7
        }
        hspace = shape_adjustments.get(pred_img.shape, -0.5)
        plt.subplots_adjust(top=0.99, bottom=0.01, hspace=hspace, wspace=0)

        # Add colorbars
        cbar_data = [('magma', 0, 100), (maps['b_viridis'], 0, 27.27), (maps['b_buda'], -0.6, 0.6),
                     (maps['b_lipari'], 0.5, 1.5), ('hot', 0, 3000), (maps['b_winter'], 0, 1000)]

        for j, (cmap, vmin, vmax) in enumerate(cbar_data):
            cax = fig.add_axes([axs[0, j].get_position().x0 + 0.01, 0.1,
                                axs[0, j].get_position().width, 0.02])
            sm = cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
            sm.set_array([])
            plt.colorbar(sm, cax=cax, orientation='horizontal',
                         ticks=np.linspace(vmin, vmax, 5))

        plt.savefig(save_path)
        plt.close()

    def _calc_metrics(self, y_pred, y_true):
        """Calculate all metrics for each channel."""
        slices = y_pred.shape[1]
        metrics = {'ssim': [], 'psnr': [], 'nrmse': []}

        for i in range(slices):
            pred_slice = y_pred[:, i, :, :].unsqueeze(1)
            true_slice = y_true[:, i, :, :].unsqueeze(1)

            # Special handling for B0 (channel 2)
            if i == 2:
                pred_slice_processed = pred_slice * 2.7 - 1
                true_slice_processed = true_slice * 2.7 - 1

                pred_mask = pred_slice_processed != 0
                true_mask = true_slice_processed != 0
                combined_mask = pred_mask & true_mask

                if combined_mask.sum() > 0:
                    pred_masked = pred_slice_processed * combined_mask.float()
                    true_masked = true_slice_processed * combined_mask.float()

                    if combined_mask.sum() > 1:
                        min_val = torch.min(true_masked[combined_mask])
                        pred_masked[~combined_mask] = min_val
                        true_masked[~combined_mask] = min_val

                        try:
                            ssim_val = self.ssim(pred_masked, true_masked).item()
                            psnr_val = self.psnr(pred_masked, true_masked).item()
                            mse = mse_loss(pred_masked, true_masked)
                            rmse = torch.sqrt(mse)
                            nrmse_val = (rmse / (torch.max(true_masked) - torch.min(true_masked))).item()
                        except:
                            ssim_val = self.ssim(pred_slice, true_slice).item()
                            psnr_val = self.psnr(pred_slice, true_slice).item()
                            mse = mse_loss(pred_slice, true_slice)
                            rmse = torch.sqrt(mse)
                            nrmse_val = (rmse / (torch.max(true_slice) - torch.min(true_slice))).item()
                    else:
                        ssim_val = self.ssim(pred_slice, true_slice).item()
                        psnr_val = self.psnr(pred_slice, true_slice).item()
                        mse = mse_loss(pred_slice, true_slice)
                        rmse = torch.sqrt(mse)
                        nrmse_val = (rmse / (torch.max(true_slice) - torch.min(true_slice))).item()
                else:
                    ssim_val = self.ssim(pred_slice, true_slice).item()
                    psnr_val = self.psnr(pred_slice, true_slice).item()
                    mse = mse_loss(pred_slice, true_slice)
                    rmse = torch.sqrt(mse)
                    nrmse_val = (rmse / (torch.max(true_slice) - torch.min(true_slice))).item()
            else:
                # Standard calculation for all other channels
                ssim_val = self.ssim(pred_slice, true_slice).item()
                psnr_val = self.psnr(pred_slice, true_slice).item()
                mse = mse_loss(pred_slice, true_slice)
                rmse = torch.sqrt(mse)
                nrmse_val = (rmse / (torch.max(true_slice) - torch.min(true_slice))).item()

            metrics['ssim'].append(ssim_val)
            metrics['psnr'].append(psnr_val)
            metrics['nrmse'].append(nrmse_val)

        # Calculate averages
        avg_metrics = {k: sum(v) / len(v) for k, v in metrics.items()}
        return metrics, avg_metrics

    def load_and_predict(self):
        """Run inference and generate plots."""
        with torch.no_grad():
            for batch, (X, p, y) in tqdm(enumerate(self.dataset), total=len(self.dataset)):
                if self.seq is not None and batch % 19 != self.seq:
                    continue

                X, p, y = X.to(self.device), p.to(self.device), y.to(self.device)
                y_pred = self.model(X, p)
                y_pred_perm = y_pred.permute(0, 3, 1, 2)

                if self.config["wanted_output"]['image_plotting']:
                    save_path = os.path.join(self.out_path_images, f"batch_{batch}.jpg")
                    self._plot_true_and_pred_sequences(y, y_pred, save_path)

                if self.config["wanted_output"]['metrics']:
                    per_metric, avg_metrics = self._calc_metrics(y_pred_perm, y)

                    # Store metrics
                    self.metrics['batch'].append(batch)
                    for metric in ['ssim', 'psnr', 'nrmse']:
                        self.metrics[metric].append(avg_metrics[metric])
                        for i in range(len(self.channel_names)):
                            self.metrics[f'{metric}_metric_{i}'].append(per_metric[metric][i])

    def calculate_summary_statistics(self):
        """Calculate overall and per-parameter statistics."""
        if not self.metrics['ssim']:
            print("No metrics data. Run load_and_predict() first.")
            return None

        # Overall stats
        overall = {metric: {'mean': np.mean(self.metrics[metric]),
                            'std': np.std(self.metrics[metric])}
                   for metric in ['ssim', 'psnr', 'nrmse']}

        # Per-parameter stats
        per_param = {}
        for i, channel in enumerate(self.channel_names):
            per_param[channel] = {
                metric.upper(): {'mean': np.mean(self.metrics[f'{metric}_metric_{i}']),
                                 'std': np.std(self.metrics[f'{metric}_metric_{i}'])}
                for metric in ['ssim', 'psnr', 'nrmse']
            }

        # Print statistics
        print("\n" + "=" * 60 + "\nSUMMARY STATISTICS\n" + "=" * 60)
        print("\n1. OVERALL STATISTICS")
        print("-" * 50)
        for metric in ['ssim', 'psnr', 'nrmse']:
            unit = " dB" if metric == 'psnr' else ""
            print(f"{metric.upper()}: {overall[metric]['mean']:.4f} ± {overall[metric]['std']:.4f}{unit}")

        print("\n2. PER-PARAMETER STATISTICS")
        print("-" * 50)
        for channel, stats in per_param.items():
            print(f"\n{channel}:")
            for metric in ['SSIM', 'PSNR', 'NRMSE']:
                unit = " dB" if metric == 'PSNR' else ""
                print(f"  {metric}: {stats[metric]['mean']:.4f} ± {stats[metric]['std']:.4f}{unit}")

        return {'overall_stats': overall, 'per_parameter_stats': per_param}

    def create_summary_table(self):
        """Create formatted summary statistics table."""
        if not self.metrics['ssim']:
            print("No metrics data. Run load_and_predict() first.")
            return

        # Prepare table data
        table_data = []

        # Overall row
        overall_data = [np.mean(self.metrics[m]) for m in ['ssim', 'psnr', 'nrmse']]
        overall_std = [np.std(self.metrics[m]) for m in ['ssim', 'psnr', 'nrmse']]
        table_data.append(['Overall'] + [f'{mean:.4f} ± {std:.4f}'
                                         for mean, std in zip(overall_data, overall_std)])

        # Per-parameter rows
        for i, channel in enumerate(self.channel_names):
            row_data = []
            for metric in ['ssim', 'psnr', 'nrmse']:
                mean = np.mean(self.metrics[f'{metric}_metric_{i}'])
                std = np.std(self.metrics[f'{metric}_metric_{i}'])
                row_data.append(f'{mean:.4f} ± {std:.4f}')
            table_data.append([channel] + row_data)

        # Create table
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('tight')
        ax.axis('off')

        table = ax.table(cellText=table_data, colLabels=['Parameter', 'SSIM', 'PSNR (dB)', 'NRMSE'],
                         cellLoc='center', loc='center')

        # Style table
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 2)

        # Header styling
        for i in range(4):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
            table[(1, i)].set_facecolor('#E8F5E8')
            table[(1, i)].set_text_props(weight='bold')

        # Alternate row colors
        for i in range(2, len(table_data) + 1):
            color = '#F5F5F5' if i % 2 == 0 else 'white'
            for j in range(4):
                table[(i, j)].set_facecolor(color)

        plt.title('Summary Statistics: Mean ± Standard Deviation', fontsize=16, fontweight='bold', pad=20)
        plt.savefig(os.path.join(self.out_path_metrics, 'summary_statistics_table.png'),
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Summary table saved to {self.out_path_metrics}")

    def create_boxplots(self):
        """Create boxplots for per-parameter metrics."""
        import seaborn as sns
        import pandas as pd

        if not self.metrics['ssim']:
            print("No metrics data. Run load_and_predict() first.")
            return

        sns.set(style="whitegrid")

        # Create combined plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        for idx, metric in enumerate(['ssim', 'psnr', 'nrmse']):
            data = []
            for i in range(6):
                for value in self.metrics[f'{metric}_metric_{i}']:
                    data.append({'Channel': self.channel_names[i], 'Value': value})

            df = pd.DataFrame(data)
            sns.boxplot(x='Channel', y='Value', data=df, ax=axes[idx], hue='Channel', legend=False)

            title = f'{metric.upper()}'
            if metric == 'psnr':
                title += ' (dB)'
            axes[idx].set_title(f'{title} by Channel', fontsize=14)
            axes[idx].set_xlabel('Channel', fontsize=12)
            axes[idx].set_ylabel(title, fontsize=12)
            axes[idx].grid(axis='y', linestyle='--', alpha=0.7)

        plt.suptitle('Performance Metrics by Channel', fontsize=16, y=1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_path_metrics, 'metrics_per_channel_boxplot.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Boxplots saved to {self.out_path_metrics}")

    def generate_summary_report(self):
        """Generate complete summary report."""
        print("Generating summary report...")

        if not self.calculate_summary_statistics():
            return

        self.create_summary_table()
        self.create_boxplots()

        print(f"Complete summary report saved to {self.out_path_metrics}")
