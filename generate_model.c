#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <assert.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include "particle_filters.h"
#include "solvers.h"
#include "generate_model.h"

const int N_TOTAL_MAX2 = 100000000;
const int N_LEVELS2 = 2;
const int N_ALLOCS2 = 5;
const int N_MESHES2 = 6;

double generate_model(gsl_rng * rng, HMM * hmm, int ** N0s, int * N1s, w_double ** weighted_ref, int N_ref, int N_trials, int N_bpf, int * level0_meshes, double * ref_xhats) {

	// int run_ref = 1;		// REF ON
	int run_ref = 0;		// REF OFF
	int compute_ss = 1;		// SS ON
	// int compute_ss = 0;		// SS OFF


	/* Reference distribution */
	/* ---------------------- */
	mesh_test(level0_meshes);
	if (run_ref == 1)
		run_reference_filter(hmm, N_ref, rng, weighted_ref);
	else
		read_cdf(weighted_ref, hmm);
	for (int n = 0; n < hmm->length; n++) {
		for (int i = 0; i < N_ref; i++)
			ref_xhats[n] += weighted_ref[n][i].x * weighted_ref[n][i].w;
	}


	/* Sample allocation */
	/* ----------------- */
	double T;
	if (compute_ss == 1) {
		T = perform_BPF_trials(hmm, N_bpf, rng, N_trials, N_ref, weighted_ref, ref_xhats);
		compute_sample_sizes(hmm, rng, level0_meshes, T, N0s, N1s, N_bpf, N_trials, ref_xhats);
	}
	else
		T = read_sample_sizes(hmm, N0s, N1s, N_trials);

	return T;
}


void generate_hmm(gsl_rng * rng) {

	/** 
	Generates the HMM data and outputs to file to be read in by read_hmm.
	*/
	// I've taken the difference between the sample dev wrt mx1 and the smallest nx0. This descibes the effect of the solvers. I've then set the obs_sd to be the abs value of this difference. Note we can reverse this.
	int length = 3;
	int nx = 51;
	int nt = 50;
	int obs_pos = (int) (0.5 * (nx + 1) - 1);
	int lag = 0;
	double sig_sd = 1.0;
	double obs_sd = 0.1;
	double space_left = 0.0, space_right = 1.0;
	double T_stop = 5.0;
	double dx = (space_right - space_left) / (double) (nx - 1);	
	double dt = T_stop / (double) (nt - 1);	
	double alpha = 0.1 / (M_PI * M_PI);
	double r = alpha * dt / (dx * dx);
	double s = 5.0, obs;
	double * u = (double *) malloc(nx * sizeof(double));
	gsl_vector * lower = gsl_vector_alloc(nx - 1);
	gsl_vector * main = gsl_vector_alloc(nx);
	gsl_vector * upper = gsl_vector_alloc(nx - 1);
	gsl_matrix * B = gsl_matrix_calloc(nx, nx);
	gsl_vector * v = gsl_vector_alloc(nx);
	gsl_vector * Bv = gsl_vector_alloc(nx);
	construct_forward(r, nx, lower, main, upper);
	construct_present(r, nx, B);
	generate_ics(u, dx, nx, alpha, space_left);

	/* Write the available parameters */
	FILE * DATA_OUT = fopen("hmm_data.txt", "w");
	FILE * CURVE_DATA = fopen("curve_data.txt", "w");
	fprintf(DATA_OUT, "%d\n", length);
	fprintf(DATA_OUT, "%lf %lf\n", sig_sd, obs_sd);
	fprintf(DATA_OUT, "%lf %lf\n", space_left, space_right);
	fprintf(DATA_OUT, "%d %d\n", nx, nt);
	fprintf(DATA_OUT, "%lf\n", T_stop);
	fprintf(DATA_OUT, "%e\n", alpha);
	fprintf(DATA_OUT, "%d\n", lag);

	if (nx % 2 == 0)
		printf("Model requires an odd number of spatial mesh points.\n");

	/* Generate the data */
	for (int n = 0; n < length; n++) {

		/* Generate the observation with respect to the signal */
		crank_nicolson_gsl(nx, dt, s, T_stop, CURVE_DATA, lower, main, upper, B, v, Bv);
		obs = v->data[obs_pos] + gsl_ran_gaussian(rng, obs_sd);
		fprintf(DATA_OUT, "%e %e\n", s, obs);

		/* Evolve the signal with the mutation model */
		s = -1 * s + gsl_ran_gaussian(rng, sig_sd);

	}

	fclose(CURVE_DATA);
	fclose(DATA_OUT);

	gsl_vector_free(lower);
	gsl_vector_free(main);
	gsl_vector_free(upper);
	gsl_matrix_free(B);
	gsl_vector_free(v);
	gsl_vector_free(Bv);

}


void read_hmm(HMM * hmm) {

	FILE * DATA = fopen("hmm_data.txt", "r");

	fscanf(DATA, "%d\n", &hmm->length);
	fscanf(DATA, "%lf %lf\n", &hmm->sig_sd, &hmm->obs_sd);
	fscanf(DATA, "%lf %lf\n", &hmm->space_left, &hmm->space_right);
	fscanf(DATA, "%d %d\n", &hmm->nx, &hmm->nt);
	fscanf(DATA, "%lf\n", &hmm->T_stop);
	fscanf(DATA, "%lf\n", &hmm->alpha);
	fscanf(DATA, "%d\n", &hmm->lag);

	hmm->signal = (double *) malloc(hmm->length * sizeof(double));
	hmm->observations = (double *) malloc(hmm->length * sizeof(double));
	for (int n = 0; n < hmm->length; n++)
		fscanf(DATA, "%lf %lf\n", &hmm->signal[n], &hmm->observations[n]);

	fclose(DATA);

	printf("data length 	= %d\n", hmm->length);
	printf("sig_sd      	= %lf\n", hmm->sig_sd);
	printf("obs_sd      	= %lf\n", hmm->obs_sd);
	printf("nx          	= %d\n", hmm->nx);
	printf("nt          	= %d\n", hmm->nt);
	printf("Stopping time 	= %lf\n", hmm->T_stop);
	printf("alpha           = %lf\n", hmm->alpha);
	printf("Lag             = %d\n", hmm->lag);
	for (int n = 0; n < hmm->length; n++)
		printf("n = %d: signal = %lf, observation = %lf\n", n, hmm->signal[n], hmm->observations[n]);

}


void run_reference_filter(HMM * hmm, int N_ref, gsl_rng * rng, w_double ** weighted_ref) {

	double ref_elapsed, w_sum;
	double * data = (double *) malloc(hmm->length * sizeof(double));	
	char sig_sd_str[50], obs_sd_str[50], nx_str[50], nt_str[50], len_str[50], lag_str[50], ref_name[100];
	snprintf(sig_sd_str, 50, "%lf", hmm->sig_sd);
	snprintf(obs_sd_str, 50, "%lf", hmm->obs_sd);
	snprintf(nx_str, 50, "%d", hmm->nx);
	snprintf(nt_str, 50, "%d", hmm->nt);
	snprintf(len_str, 50, "%d", hmm->length);
	snprintf(lag_str, 50, "%d", hmm->lag);
	sprintf(ref_name, "ref_particles_sig_sd=%s_obs_sd=%s_nx=%s_nt=%s_len=%s_lag=%s.txt", sig_sd_str, obs_sd_str, nx_str, nt_str, len_str, lag_str);
	puts(ref_name);

	/* Run the BPF with the reference number of particles */
	printf("Running reference BPF...\n");
	clock_t ref_timer = clock();
	bootstrap_particle_filter_gsl(hmm, N_ref, rng, weighted_ref, data);
	ref_elapsed = (double) (clock() - ref_timer) / (double) CLOCKS_PER_SEC;
	printf("Reference BPF for %d particles completed in %f seconds\n", N_ref, ref_elapsed);

	/* Sort and output the weighted particles for the KS tests */
	for (int n = 0; n < hmm->length; n++)
		qsort(weighted_ref[n], N_ref, sizeof(w_double), weighted_double_cmp);
	output_cdf(weighted_ref, hmm, N_ref, ref_name);

	free(data);
}


void output_cdf(w_double ** w_particles, HMM * hmm, int N, char file_name[100]) {

	FILE * DATA = fopen(file_name, "w");
	fprintf(DATA, "%d %d\n", N, hmm->length);

	for (int n = 0; n < hmm->length; n++) {
		for (int i = 0; i < N; i++)
			fprintf(DATA, "%e ", w_particles[n][i].x);
		fprintf(DATA, "\n");
		for (int i = 0; i < N; i++)
			fprintf(DATA, "%e ", w_particles[n][i].w);
		fprintf(DATA, "\n");
	}
	fclose(DATA);
}


void read_cdf(w_double ** w_particles, HMM * hmm) {

	int N, length;
	char sig_sd_str[50], obs_sd_str[50], nx_str[50], nt_str[50], len_str[50], lag_str[50], ref_name[100];
	snprintf(sig_sd_str, 50, "%lf", hmm->sig_sd);
	snprintf(obs_sd_str, 50, "%lf", hmm->obs_sd);
	snprintf(nx_str, 50, "%d", hmm->nx);
	snprintf(nt_str, 50, "%d", hmm->nt);
	snprintf(len_str, 50, "%d", hmm->length);
	snprintf(lag_str, 50, "%d", hmm->lag);
	sprintf(ref_name, "ref_particles_sig_sd=%s_obs_sd=%s_nx=%s_nt=%s_len=%s_lag=%s.txt", sig_sd_str, obs_sd_str, nx_str, nt_str, len_str, lag_str);
	puts(ref_name);

	FILE * DATA = fopen(ref_name, "r");
	fscanf(DATA, "%d %d\n", &N, &length);

	double w_sum;
	for (int n = 0; n < length; n++) {
		w_sum = 0.0;
		for (int i = 0; i < N; i++)
			fscanf(DATA, "%lf ", &w_particles[n][i].x);
		for (int i = 0; i < N; i++) {
			fscanf(DATA, "%lf ", &w_particles[n][i].w);
			w_sum += w_particles[n][i].w;
		}
	}
	fclose(DATA);
}


void mesh_test(int * level0_meshes) {
	for (int n = 0; n < N_MESHES2; n++) {
		if (level0_meshes[n] % 2 == 0 || level0_meshes[n] < 3)
			printf("Mesh sizes should be odd and >= 3\n");
	}
}


double perform_BPF_trials(HMM * hmm, int N_bpf, gsl_rng * rng, int N_trials, int N_ref, w_double ** weighted_ref, double * ref_xhats) {

	int length = hmm->length;
	double ks = 0.0, elapsed = 0.0, mse = 0.0;
	double * data = (double *) malloc(length * sizeof(double));
	double * raw_ks = (double *) calloc(N_trials, sizeof(double));
	double * raw_mse = (double *) malloc(N_trials * sizeof(double));
	double * raw_times = (double *) malloc(N_trials * sizeof(double));
	double * mse_by_step = (double *) calloc(length, sizeof(double));
	double * bpf_xhats = (double *) calloc(length, sizeof(double *));	
	w_double ** weighted = (w_double **) malloc(length * sizeof(w_double *));
	for (int n = 0; n < length; n++)
		weighted[n] = (w_double *) malloc(N_bpf * sizeof(w_double));

	FILE * RAW_BPF_TIMES = fopen("raw_bpf_times.txt", "w");
	FILE * RAW_BPF_KS = fopen("raw_bpf_ks.txt", "w");
	FILE * RAW_BPF_MSE = fopen("raw_bpf_mse.txt", "w");
	FILE * BPF_MSE_BY_STEP = fopen("bpf_mse_by_step.txt", "w");

	printf("Running BPF trials...\n");
	for (int n_trial = 0; n_trial < N_trials; n_trial++) {

		printf("n_trial = %d\n", n_trial);

		/* Run the simulation for the BPF */
		clock_t bpf_timer = clock();
		bootstrap_particle_filter_gsl(hmm, N_bpf, rng, weighted, data);
		raw_times[n_trial] = (double) (clock() - bpf_timer) / (double) CLOCKS_PER_SEC;

		/* Average the MSE over the number of trials for each iterate */
		for (int n = 0; n < length; n++) {
			bpf_xhats[n] = 0.0;
			for (int i = 0; i < N_bpf; i++)
				bpf_xhats[n] += weighted[n][i].x * weighted[n][i].w;
			mse_by_step[n] += (bpf_xhats[n] - ref_xhats[n]) * (bpf_xhats[n] - ref_xhats[n]) / (double) N_trials;
		}

		/* Compute the KS statistic for the run */
		for (int n = 0; n < length; n++) {
			qsort(weighted[n], N_bpf, sizeof(w_double), weighted_double_cmp);
			raw_ks[n_trial] += ks_statistic(N_ref, weighted_ref[n], N_bpf, weighted[n]) / (double) length;
		}
		raw_mse[n_trial] = compute_mse(weighted_ref, weighted, length, N_ref, N_bpf);

	}

	ks = 0.0, mse = 0.0;
	fprintf(RAW_BPF_TIMES, "%d\n", N_bpf);
	for (int n_trial = 0; n_trial < N_trials; n_trial++) {
		fprintf(RAW_BPF_TIMES, "%e ", raw_times[n_trial]);
		fprintf(RAW_BPF_KS, "%e ", raw_ks[n_trial]);
		fprintf(RAW_BPF_MSE, "%e ", raw_mse[n_trial]);
		elapsed += raw_times[n_trial] / (double) N_trials;
		ks += raw_ks[n_trial] / (double) N_trials;
		mse += raw_mse[n_trial] / (double) N_trials;
	}
	fprintf(RAW_BPF_KS, "%e ", ks);
	fprintf(RAW_BPF_MSE, "%e ", mse);
	printf("ks from the bpf trials = %e\n", ks);
	printf("mse from the bpf trials = %e\n", mse);
	printf("avg log10 mse from the bpf trials = %e\n", log10(mse));
	for (int n = 0; n < length; n++)
		fprintf(BPF_MSE_BY_STEP, "%e ", mse_by_step[n]);

	fclose(RAW_BPF_TIMES);
	fclose(RAW_BPF_KS);
	fclose(RAW_BPF_MSE);
	fclose(BPF_MSE_BY_STEP);

	free(data);
	free(raw_ks);
	free(raw_mse);
	free(raw_times);
	free(weighted);
	free(bpf_xhats);

	return elapsed;
}


void compute_sample_sizes(HMM * hmm, gsl_rng * rng, int * level0_meshes, double T, int ** N0s, int * N1s, int N_bpf, int N_trials, double * ref_xhats) {


	/* Variables to compute the sample sizes */
	/* ------------------------------------- */
	int N0, N0_lo, dist;
	double T_mlbpf, diff;
	clock_t timer;
	int N1_incr = (int) (N_bpf / (double) N_ALLOCS2);


	/* Variables to run the MLBPF */
	/* -------------------------- */
	int length = hmm->length;
	int nxs[N_LEVELS2] = { 0, hmm->nx };
	int * sample_sizes = (int *) malloc(N_LEVELS2 * sizeof(int));
	double * sign_ratios = (double *) malloc(length * sizeof(double));
	w_double ** ml_weighted = (w_double **) malloc(length * sizeof(w_double *));
	for (int n = 0; n < length; n++)
		ml_weighted[n] = (w_double *) malloc(N_TOTAL_MAX2 * sizeof(w_double));


	/* Variables for printing to file */
	/* ------------------------------ */
	char sig_sd_str[50], obs_sd_str[50], nx_str[50], nt_str[50], len_str[50], lag_str[50], N0_data[100], trials_str[50];
	snprintf(sig_sd_str, 50, "%lf", hmm->sig_sd);
	snprintf(obs_sd_str, 50, "%lf", hmm->obs_sd);
	snprintf(nx_str, 50, "%d", hmm->nx);
	snprintf(nt_str, 50, "%d", hmm->nt);
	snprintf(len_str, 50, "%d", hmm->length);
	snprintf(lag_str, 50, "%d", hmm->lag);
	snprintf(trials_str, 50, "%d", N_trials);
	sprintf(N0_data, "N0_data_sig_sd=%s_obs_sd=%s_nx=%s_nt=%s_len=%s_lag=%s_N_trials=%s.txt", sig_sd_str, obs_sd_str, nx_str, nt_str, len_str, lag_str, trials_str);
	puts(N0_data);
	FILE * N0s_f = fopen(N0_data, "w");
	fprintf(N0s_f, "%e\n", T);
	for (int n_alloc = 0; n_alloc < N_ALLOCS2; n_alloc++)
		fprintf(N0s_f, "%d ", N1s[n_alloc]);
	fprintf(N0s_f, "\n");


	/* Compute the particle allocations */
	/* -------------------------------- */
	for (int i_mesh = 0; i_mesh < N_MESHES2; i_mesh++) {

		nxs[0] = level0_meshes[i_mesh];
		printf("Computing the level 0 allocations for nx0 = %d\n", nxs[0]);

		for (int n_alloc = 0; n_alloc < N_ALLOCS2; n_alloc++) {

			sample_sizes[1] = N1s[n_alloc];
			printf("N1 = %d\n", N1s[n_alloc]);

			N0 = N_bpf;
			sample_sizes[0] = N0;
			N0_lo = N0;

			/* Find a value for N0_init that exceeds the required time */
			clock_t timer = clock();
			ml_bootstrap_particle_filter_gsl(hmm, sample_sizes, nxs, rng, ml_weighted, sign_ratios, ref_xhats);
			T_mlbpf = (double) (clock() - timer) / (double) CLOCKS_PER_SEC;
			diff = (T_mlbpf - T) / T;
			while (diff < 0) {
				N0 *= 2;
				sample_sizes[0] = N0;

				timer = clock();
				ml_bootstrap_particle_filter_gsl(hmm, sample_sizes, nxs, rng, ml_weighted, sign_ratios, ref_xhats);
				T_mlbpf = (double) (clock() - timer) / (double) CLOCKS_PER_SEC;
				diff = (T_mlbpf - T) / T;
			}

			/* Find a value for N0_lo that does not meet the required time */
			sample_sizes[0] = N0_lo;
			timer = clock();
			ml_bootstrap_particle_filter_gsl(hmm, sample_sizes, nxs, rng, ml_weighted, sign_ratios, ref_xhats);
			T_mlbpf = (double) (clock() - timer) / (double) CLOCKS_PER_SEC;
			diff = (T_mlbpf - T) / T;
			while (diff > 0) {
				N0_lo = (int) (N0_lo / 2.0);
				sample_sizes[0] = N0_lo;

				timer = clock();
				ml_bootstrap_particle_filter_gsl(hmm, sample_sizes, nxs, rng, ml_weighted, sign_ratios, ref_xhats);
				T_mlbpf = (double) (clock() - timer) / (double) CLOCKS_PER_SEC;
				diff = (T_mlbpf - T) / T;

				if (N0_lo == 0)
					diff = 0;
			}

			/* Run with the N0 we know exceeds the required time */
			sample_sizes[0] = N0;
			timer = clock();
			ml_bootstrap_particle_filter_gsl(hmm, sample_sizes, nxs, rng, ml_weighted, sign_ratios, ref_xhats);
			T_mlbpf = (double) (clock() - timer) / (double) CLOCKS_PER_SEC;
			diff = (T_mlbpf - T) / T;

			if (N0_lo == 0)
				sample_sizes[0] = 0;

			else {
				/* Halve the interval until a sufficiently accurate root is found */
				while (fabs(diff) >= 0.01) {
					if (diff > 0)
						N0 = (int) (0.5 * (N0_lo + N0));
					else {
						dist = N0 - N0_lo;
						N0_lo = N0;
						N0 += dist;
					}
					sample_sizes[0] = N0;

					timer = clock();
					for (int i = 0; i < 1; i++)
						ml_bootstrap_particle_filter_gsl(hmm, sample_sizes, nxs, rng, ml_weighted, sign_ratios, ref_xhats);
					T_mlbpf = (double) (clock() - timer) / (double) CLOCKS_PER_SEC / 1.0;
					diff = (T_mlbpf - T) / T;

					if (N0_lo == N0)
						diff = 0.0;
				}
			}

			N0s[i_mesh][n_alloc] = sample_sizes[0];
			printf("N0 = %d for N1 = %d and nx0 = %d, timed diff = %.10lf\n", sample_sizes[0], N1s[n_alloc], nxs[0], diff);
			printf("\n");
			fprintf(N0s_f, "%d ", sample_sizes[0]);

		}

		fprintf(N0s_f, "\n");

	}

	fclose(N0s_f);

	free(sign_ratios);
	free(ml_weighted);
	free(sample_sizes);

}


double read_sample_sizes(HMM * hmm, int ** N0s, int * N1s, int N_trials) {

	char sig_sd_str[50], obs_sd_str[50], nx_str[50], nt_str[50], len_str[50], lag_str[50], N0_data[100], trials_str[50];
	snprintf(sig_sd_str, 50, "%lf", hmm->sig_sd);
	snprintf(obs_sd_str, 50, "%lf", hmm->obs_sd);
	snprintf(nx_str, 50, "%d", hmm->nx);
	snprintf(nt_str, 50, "%d", hmm->nt);
	snprintf(len_str, 50, "%d", hmm->length);
	snprintf(lag_str, 50, "%d", hmm->lag);
	snprintf(trials_str, 50, "%d", N_trials);
	sprintf(N0_data, "N0_data_sig_sd=%s_obs_sd=%s_nx=%s_nt=%s_len=%s_lag=%s_N_trials=%s.txt", sig_sd_str, obs_sd_str, nx_str, nt_str, len_str, lag_str, trials_str);
	FILE * N0s_f = fopen(N0_data, "r");

	double T;
	fscanf(N0s_f, "%lf\n", &T);
	for (int n_alloc = 0; n_alloc < N_ALLOCS2; n_alloc++)
		fscanf(N0s_f, "%d ", &N1s[n_alloc]);
	for (int i_mesh = 0; i_mesh < N_MESHES2; i_mesh++) {
		for (int n_alloc = 0; n_alloc < N_ALLOCS2; n_alloc++)
			fscanf(N0s_f, "%d ", &N0s[i_mesh][n_alloc]);
	}

	for (int n_alloc = 0; n_alloc < N_ALLOCS2; n_alloc++)
		printf("N1[%d] = %d ", n_alloc, N1s[n_alloc]);
	printf("\n");
	for (int i_mesh = 0; i_mesh < N_MESHES2; i_mesh++) {
		for (int n_alloc = 0; n_alloc < N_ALLOCS2; n_alloc++)
			printf("N0[%d] = %d ", n_alloc, N0s[i_mesh][n_alloc]);
		printf("\n");
	}

	fclose(N0s_f);
	return T;

}


double ks_statistic(int N_ref, w_double * weighted_ref, int N, w_double * weighted) {

	double record, diff;
	double cum1 = 0, cum2 = 0;
	int j = 0, lim1, lim2;
	w_double * a1;
	w_double * a2;

	if (weighted_ref[0].x < weighted[0].x) {
		a1 = weighted_ref;
		a2 = weighted;
		lim1 = N_ref;
		lim2 = N;
	}
	else {
		a1 = weighted;
		a2 = weighted_ref;
		lim1 = N;
		lim2 = N_ref;
	}

	cum1 = a1[0].w;
	record = cum1;
	for (int i = 1; i < lim1; i++) {
		while (a2[j].x < a1[i].x && j < lim2) {
			cum2 += a2[j].w;
			diff = fabs(cum2 - cum1);
			record = diff > record ? diff : record;
			j++;
		}
		cum1 += a1[i].w;
		diff = fabs(cum2 - cum1);
		record = diff > record ? diff : record;
	}
	return record;
}


double compute_mse(w_double ** weighted1, w_double ** weighted2, int length, int N1, int N2) {

	double mse = 0.0, x1_hat, x2_hat, w1_sum, w2_sum;

	for (int n = 0; n < length; n++) {
		x1_hat = 0.0, x2_hat = 0.0, w1_sum = 0.0, w2_sum = 0.0;
		for (int i = 0; i < N1; i++) {
			x1_hat += weighted1[n][i].w * weighted1[n][i].x;
			w1_sum += weighted1[n][i].w;
		}
		for (int i = 0; i < N2; i++) {
			x2_hat += weighted2[n][i].w * weighted2[n][i].x;
			w2_sum += weighted2[n][i].w;
		}
		if (isnan(w2_sum))
			printf("NaN encountered\n");
		else {
			assert(fabs(w1_sum - 1.0) < 0.001);
			assert(fabs(w2_sum - 1.0) < 0.001);
		}
		mse = mse + (x1_hat - x2_hat) * (x1_hat - x2_hat);
	}
	return mse / (double) length;
}





		// y = v->data[obs_pos];
		// y_std = 0.0;
		// for (int i = 0; i < N_sample; i++) {
		// 	s_arr[i] = s + gsl_ran_gaussian(rng, sig_sd);
		// 	crank_nicolson_gsl(nx, dt, s_arr[i], T_stop, CURVE_DATA, lower, main, upper, B, v, Bv);
		// 	y_std += (v->data[obs_pos] - y) * (v->data[obs_pos] - y);
		// 	gsl_vector_memcpy(v, v_temp);
		// }
		// printf("Empirical std = %.8lf\n", sqrt(y_std / (double) (N_sample - 1)));		
		// printf("%.10lf\n", (obs - v->data[obs_pos]) / obs_sd);