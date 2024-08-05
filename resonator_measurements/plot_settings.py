import matplotlib.pyplot as plt

# Define Standard Units
fontsize = 10
axlabelsize = 12
titlesize = 18
figtitlesize = 24
legendsize = 12


tickdir = 'in'
major = 4.0
minor = 2.0
style = 'default'

# Set all parameters for the plot
plt.style.use(style)
plt.rcParams['figure.figsize'] =  8, 8
plt.rcParams['text.usetex'] = False
plt.rcParams['font.size'] = fontsize
plt.rcParams['axes.labelsize'] = axlabelsize
plt.rcParams['figure.titlesize'] = figtitlesize
plt.rcParams['legend.fontsize'] = legendsize


# plt.rcParams['font.family'] = 'times'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['xtick.direction'] = tickdir
plt.rcParams['ytick.direction'] = tickdir
plt.rcParams['xtick.major.size'] = major
plt.rcParams['xtick.minor.size'] = minor
plt.rcParams['ytick.major.size'] = major
plt.rcParams['ytick.minor.size'] = minor
plt.rcParams['xtick.labelsize'] = fontsize
plt.rcParams['ytick.labelsize'] = fontsize
plt.rcParams['axes.formatter.useoffset'] = False


# for k in plt.rcParams.keys():
#     print(k)