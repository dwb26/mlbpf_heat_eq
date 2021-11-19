void generate_hmm(gsl_rng * rng);

double generate_model(gsl_rng * rng, HMM * hmm, int ** N0s, int * N1s, w_double ** weighted_ref, int N_ref, int N_trials, int N_bpf, int * level0_meshes, double * ref_xhats);

void read_hmm(HMM * hmm);

void run_reference_filter(HMM * hmm, int N_ref, gsl_rng * rng, w_double ** weighted_ref);

void output_cdf(w_double ** w_particles, HMM * hmm, int N, char file_name[100]);

void read_cdf(w_double ** w_particles, HMM * hmm);

void mesh_test(int * level0_meshes);

double perform_BPF_trials(HMM * hmm, int N_bpf, gsl_rng * rng, int N_trials, int N_ref, w_double ** weighted_ref, double * ref_xhats);

void compute_sample_sizes(HMM * hmm, gsl_rng * rng, int * level0_meshes, double T, int ** N0s, int * N1s, int N_bpf, int N_trials, double * ref_xhats);

double read_sample_sizes(HMM * hmm, int ** N0s, int * N1s, int N_trials);

double ks_statistic(int N_ref, w_double * weighted_ref, int N, w_double * weighted);

double compute_mse(w_double ** weighted1, w_double ** weighted2, int length, int N1, int N2);