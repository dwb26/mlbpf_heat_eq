import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import seaborn as sns

hmm_data = open("hmm_data.txt", "r")
raw_times = open("raw_times.txt", "r")
raw_ks = open("raw_ks.txt", "r")
raw_mse = open("raw_mse.txt", "r")
raw_srs = open("raw_srs.txt", "r")
ml_parameters = open("ml_parameters.txt", "r")
N1s_data = open("N1s_data.txt", "r")
raw_bpf_times = open("raw_bpf_times.txt", "r")
raw_bpf_ks = open("raw_bpf_ks.txt", "r")
raw_bpf_mse = open("raw_bpf_mse.txt", "r")
alloc_counters_f = open("alloc_counters.txt", "r")


# --------------------------------------------------- HMM data ------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------------------------- #
length = int(hmm_data.readline())
sig_sd, obs_sd = list(map(float, hmm_data.readline().split()))
hmm_data.readline()
nx, nt = list(map(int, hmm_data.readline().split()))
for n in range(2):
    hmm_data.readline()
lag = int(hmm_data.readline())


# ----------------------------------------------- BPF parameters ---------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------- #
N_bpf = int(raw_bpf_times.readline())
bpf_times = list(map(float, raw_bpf_times.readline().split()))
bpf_ks = list(map(float, raw_bpf_ks.readline().split()))
bpf_mse = list(map(float, raw_bpf_mse.readline().split()))

bpf_mean_time = np.mean(bpf_times)
bpf_median_time = np.median(bpf_times)
bpf_mean_ks = np.mean(bpf_ks)
bpf_median_ks = np.median(bpf_ks)
bpf_mean_mse = np.mean(bpf_mse)
bpf_median_mse = np.median(bpf_mse)

bpf_mean_ks_log10 = np.log10(bpf_mean_ks)
bpf_median_ks_log10 = np.log10(bpf_median_ks)
bpf_mean_mse_log10 = np.log10(bpf_mean_mse)
bpf_median_mse_log10 = np.log10(bpf_median_mse)


# ------------------------------------------- Multilevel parameters ------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------- #
N_trials, total_allocs, N_meshes = list(map(int, ml_parameters.readline().split()))
level0s = list(map(int, ml_parameters.readline().split()))
N1s = list(map(int, N1s_data.readline().split()))
alloc_counters = np.array(list(map(int, alloc_counters_f.readline().split())))
max_allocs = np.max(alloc_counters)
mse_arr = np.zeros((N_trials, N_meshes, max_allocs))
ks_arr = np.zeros((N_trials, N_meshes, max_allocs))
times_arr = np.zeros((N_trials, N_meshes, max_allocs))
srs_arr = np.zeros((N_trials, N_meshes, max_allocs))
for i_mesh in range(N_meshes):
	for n_alloc in range(alloc_counters[i_mesh]):
		mse_arr[:, i_mesh, n_alloc] = list(map(float, raw_mse.readline().split()))
		ks_arr[:, i_mesh, n_alloc] = list(map(float, raw_ks.readline().split()))
		times_arr[:, i_mesh, n_alloc] = list(map(float, raw_times.readline().split()))
		srs_arr[:, i_mesh, n_alloc] = list(map(float, raw_srs.readline().split()))



# ------------------------------------------------------------------------------------------------------------------- #
#
# Plotting
#
# ------------------------------------------------------------------------------------------------------------------- #
colors = ["mediumpurple", "royalblue", "powderblue", "mediumseagreen", "greenyellow", "orange", "tomato", "firebrick"]
fig_width = 8; fig_height = 7
hspace = 0.9
fig1, axs = plt.subplots(nrows=N_meshes, ncols=1, figsize=(fig_width, fig_height))
fig2, axs2 = plt.subplots(nrows=4, ncols=1, figsize=(fig_width, fig_height))
fig3 = plt.figure(figsize=(fig_width, fig_height))
ax3 = plt.subplot(111)
fig1.subplots_adjust(hspace=0.9)
fig1.suptitle(r"N_trials = {}, nx = {}, nt = {}, N_bpf = {}, $\sigma_{} = {}$, $\sigma_{} = {}$, len = {}, lag = {}".format(N_trials, nx, nt, N_bpf, "s", sig_sd, "o", obs_sd, length, lag))
fig2.subplots_adjust(hspace=hspace)
fig2.suptitle(r"N_trials = {}, nx = {}, nt = {}, N_bpf = {}, $\sigma_{} = {}$, $\sigma_{} = {}$, len = {}, lag = {}".format(N_trials, nx, nt, N_bpf, "s", sig_sd, "o", obs_sd, length, lag))



# ------------------------------------------------------------------------------------------------------------------- #
#
# Boxplots
#
# ------------------------------------------------------------------------------------------------------------------- #
if N_meshes > 1:
	for i_mesh in range(N_meshes):
		ax = sns.boxplot(data=np.log10(pd.DataFrame(mse_arr[:, i_mesh, :alloc_counters[i_mesh]], columns=N1s[:alloc_counters[i_mesh]])), ax=axs[i_mesh], color=colors[i_mesh])
		ax.plot(range(max_allocs), bpf_median_mse_log10 * np.ones(max_allocs), color="limegreen", label="BPF")
		ax.set_title("Level 0 mesh size = {}".format(level0s[i_mesh]), fontsize=9)
else:
    for i_mesh in range(N_meshes):
        ax = sns.boxplot(data=np.log10(pd.DataFrame(mse_arr[:, i_mesh, :alloc_counters[i_mesh]], columns=N1s[:alloc_counters[i_mesh]])), ax=axs, color=colors[i_mesh])
        ax.plot(range(max_allocs), bpf_median_mse_log10 * np.ones(max_allocs), color="limegreen", label="BPF")
        ax.set_title("Level 0 mesh size = {}".format(level0s[i_mesh]), fontsize=9)



# ------------------------------------------------------------------------------------------------------------------- #
#
# Mean MSE from reference point estimates
#
# ------------------------------------------------------------------------------------------------------------------- #
axs2[0].set_title("log10(Mean MSE)", fontsize=9)
axs2[0].plot(N1s[:max_allocs], bpf_mean_mse_log10 * np.ones(max_allocs), color="black", label="BPF")
for i_mesh in range(N_meshes):
	axs2[0].plot(N1s[:alloc_counters[i_mesh]], np.log10(np.mean(mse_arr[:, i_mesh, :alloc_counters[i_mesh]], axis=0)), label=level0s[i_mesh], marker="o", color=colors[i_mesh], markersize=3)
axs2[0].set_xticks([])



# ------------------------------------------------------------------------------------------------------------------- #
#
# KS statistics from the reference distribution
#
# ------------------------------------------------------------------------------------------------------------------- #
axs2[1].set_title("log10(Mean KS statistics)", fontsize=9)
axs2[1].plot(N1s[:max_allocs], bpf_mean_ks_log10 * np.ones(max_allocs), color="black", label="BPF")
for i_mesh in range(N_meshes):
	axs2[1].plot(N1s[:alloc_counters[i_mesh]], np.log10(np.mean(ks_arr[:, i_mesh, :alloc_counters[i_mesh]], axis=0)), label=level0s[i_mesh], marker="o", color=colors[i_mesh], markersize=3)
axs2[1].set_xticks([])



# ------------------------------------------------------------------------------------------------------------------- #
#
# Sign ratios
#
# ------------------------------------------------------------------------------------------------------------------- #
axs2[2].set_title("Mean sign ratios", fontsize=9)
for i_mesh in range(N_meshes):
    mean_srs = np.mean(srs_arr[:, i_mesh, :alloc_counters[i_mesh]], axis=0)
    axs2[2].plot(N1s[:alloc_counters[i_mesh]], mean_srs, label=level0s[i_mesh], marker="o", color=colors[i_mesh], markersize=3)
axs2[2].set_xlabel("N1")
axs2[2].legend(loc=3, prop={'size': 8})



# ------------------------------------------------------------------------------------------------------------------- #
#
# Trial times
#
# ------------------------------------------------------------------------------------------------------------------- #
times_list = []
for i_mesh in range(N_meshes):
	for n_alloc in range(alloc_counters[i_mesh]):
		times_list.extend(times_arr[:, i_mesh, n_alloc])
total_time_length = len(np.array(times_list).flatten())
axs2[3].set_title("Trial times", fontsize=9)
axs2[3].plot(range(total_time_length), np.array(times_list).flatten(), linewidth=0.5)
axs2[3].plot(range(total_time_length), np.mean(bpf_times) * np.ones(total_time_length), label="bpf mean", color="black")

fig1.show()
fig2.show()
input()































