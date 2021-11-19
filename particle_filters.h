#ifndef _PARTICLE_FILTERS_TRAFFIC_H_
#define _PARTICLE_FILTERS_TRAFFIC_H_
#endif

typedef struct {
	double * signal;
	double * observations;
	int length;
	int nx;
	int nt;
	double sig_sd;
	double obs_sd;
	double space_left;
	double space_right;
	double T_stop;
	double alpha;
	int lag;
} HMM;


typedef struct weighted_double {
	double x;
	double w;
} w_double;

int weighted_double_cmp(const void * a, const void * b);

void bootstrap_particle_filter_gsl(HMM * hmm, int N, gsl_rng * rng, w_double ** weighted, double * data);

// void ml_bootstrap_particle_filter_gsl(HMM * hmm, int * sample_sizes, int * nxs, gsl_rng * rng, w_double ** ml_weighted, double * sign_ratios);

void ml_bootstrap_particle_filter_gsl(HMM * hmm, int * sample_sizes, int * nxs, gsl_rng * rng, w_double ** ml_weighted, double * sign_ratios, double * ref_xhats);

void adaptive_mlbpf(HMM * hmm, int * N0s_var_nx, int N1, int * nxs, gsl_rng * rng, w_double ** ml_weighted, double * data, double * times, int * N0s_rec, int * nx0_rec);

void bootstrap_particle_filter(HMM * hmm, int N, gsl_rng * rng, w_double ** weighted, double * data);

void ml_bootstrap_particle_filter(HMM * hmm, int * sample_sizes, int * nxs, gsl_rng * rng, w_double ** ml_weighted, double * sign_ratios, double * times);