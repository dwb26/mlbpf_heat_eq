// gcc -o equal_runtimes -lm -lgsl -lgslcblas equal_runtimes.c particle_filters.c solvers.c generate_model.c
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include "particle_filters.h"
#include "solvers.h"
#include "generate_model.h"

const int N_TOTAL_MAX = 100000000;
const int N_LEVELS = 2;
const int N_ALLOCS = 5;
const int N_MESHES = 6;

void output_ml_data(HMM * hmm, int N_trials, double *** raw_times, double *** raw_ks, double *** raw_mse, double *** raw_srs, int * level0_meshes, int * N1s, double * sr_tracker, double ** mse_tracker, int * alloc_counters);


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// This is the heat equation model
int main(void) {

	clock_t timer = clock();
	gsl_rng * rng = gsl_rng_alloc(gsl_rng_taus);


	/* Generate the HMM data */
	/* --------------------- */
	HMM * hmm = (HMM *) malloc(sizeof(HMM));
	generate_hmm(rng);
	read_hmm(hmm);
	int length = hmm->length;


	/* Generate the reference distribution and sample allocations */
	/* ---------------------------------------------------------- */
	int N_ref = 1000000;
	int N_trials = 100;
	int N_bpf = 1000;
	int level0_meshes[N_MESHES] = { 41, 31, 21, 11, 5, 3 };
	int * N1s = (int *) malloc(N_ALLOCS * sizeof(int));
	int ** N0s = (int **) malloc(N_MESHES * sizeof(int *));
	for (int n_alloc = 0; n_alloc < N_ALLOCS; n_alloc++)
		N1s[n_alloc] = (int) (n_alloc * N_bpf / (double) N_ALLOCS);
	for (int i_mesh = 0; i_mesh < N_MESHES; i_mesh++)
		N0s[i_mesh] = (int *) malloc(N_ALLOCS * sizeof(int));
	w_double ** weighted_ref = (w_double **) malloc(length * sizeof(w_double *));
	for (int n = 0; n < length; n++)
		weighted_ref[n] = (w_double *) malloc(N_ref * sizeof(w_double));
	double * ref_xhats = (double *) calloc(length, sizeof(double));
	double T = generate_model(rng, hmm, N0s, N1s, weighted_ref, N_ref, N_trials, N_bpf, level0_meshes, ref_xhats);


	/* Main experiment parameters */
	/* -------------------------- */
	int N0, N1, N_tot;	
	int nxs[N_LEVELS] = { 0, hmm->nx };
	double ks, sr, ml_xhat;	
	int * sample_sizes = (int *) malloc(N_LEVELS * sizeof(int));
	int * alloc_counters = (int *) malloc(N_MESHES * sizeof(int));
	for (int i_mesh = 0; i_mesh < N_MESHES; i_mesh++)
		alloc_counters[i_mesh] = N_ALLOCS;
	double * sign_ratios = (double *) calloc(length, sizeof(double));
	w_double ** ml_weighted = (w_double **) malloc(length * sizeof(w_double *));
	for (int n = 0; n < length; n++)
		ml_weighted[n] = (w_double *) malloc(N_TOTAL_MAX * sizeof(w_double));
	double *** raw_ks = (double ***) malloc(N_trials * sizeof(double **));
	double *** raw_mse = (double ***) malloc(N_trials * sizeof(double **));
	double *** raw_times = (double ***) malloc(N_trials * sizeof(double **));
	double *** raw_srs = (double ***) malloc(N_trials * sizeof(double **));
	for (int n_trial = 0; n_trial < N_trials; n_trial++) {		
		raw_ks[n_trial] = (double **) malloc(N_MESHES * sizeof(double *));
		raw_mse[n_trial] = (double **) malloc(N_MESHES * sizeof(double *));
		raw_times[n_trial] = (double **) malloc(N_MESHES * sizeof(double *));
		raw_srs[n_trial] = (double **) malloc(N_MESHES * sizeof(double *));
		for (int i_mesh = 0; i_mesh < N_MESHES; i_mesh++) {
			raw_ks[n_trial][i_mesh] = (double *) calloc(N_ALLOCS, sizeof(double));
			raw_mse[n_trial][i_mesh] = (double *) calloc(N_ALLOCS, sizeof(double));
			raw_times[n_trial][i_mesh] = (double *) calloc(N_ALLOCS, sizeof(double));
			raw_srs[n_trial][i_mesh] = (double *) calloc(N_ALLOCS, sizeof(double));	
		}
	}
	double ** mse_tracker = (double **) malloc(length * sizeof(double *));
	double * sr_tracker = (double *) calloc(length, sizeof(double));
	for (int n = 0; n < length; n++)
		mse_tracker[n] = (double *) calloc(N_ALLOCS, sizeof(double));



	/* ------------------------------------------------------------------------------------------------------------- */
	/* 																												 */
	/* MLBPF accuracy trials 																						 */
	/* 																												 */
	/* ------------------------------------------------------------------------------------------------------------- */
	for (int n_alloc = 0; n_alloc < N_ALLOCS; n_alloc++) {

		printf("--------------\n");
		printf("|  N1 = %d  |\n", N1s[n_alloc]);
		printf("--------------\n");

		for (int i_mesh = 0; i_mesh < N_MESHES; i_mesh++) {

			printf("nx0 = %d\n", level0_meshes[i_mesh]);
			printf("**********************************************************\n");

			N1 = N1s[n_alloc], nxs[0] = level0_meshes[i_mesh];
			N0 = N0s[i_mesh][n_alloc];
			N_tot = N0 + N1;
			sample_sizes[0] = N0, sample_sizes[1] = N1;

			if (N0 == 0)
				alloc_counters[i_mesh] = n_alloc < alloc_counters[i_mesh] ? n_alloc : alloc_counters[i_mesh];
			else {
				for (int n_trial = 0; n_trial < N_trials; n_trial++) {
					clock_t trial_timer = clock();
					ml_bootstrap_particle_filter_gsl(hmm, sample_sizes, nxs, rng, ml_weighted, sign_ratios, ref_xhats);
					double elapsed = (double) (clock() - trial_timer) / (double) CLOCKS_PER_SEC;
					// printf("Time elapsed = %lf seconds, BPF time taken = %lf\n", elapsed, T);

					/* Average the statistics over the length of the time series */
					ks = 0.0, sr = 0.0;
					for (int n = 0; n < length; n++) {
						qsort(ml_weighted[n], N_tot, sizeof(w_double), weighted_double_cmp);
						ks += ks_statistic(N_ref, weighted_ref[n], N_tot, ml_weighted[n]) / (double) length;
						sr += sign_ratios[n] / (double) length;
					}
					raw_mse[n_trial][i_mesh][n_alloc] = compute_mse(weighted_ref, ml_weighted, length, N_ref, N_tot);
					raw_ks[n_trial][i_mesh][n_alloc] = ks;					
					raw_times[n_trial][i_mesh][n_alloc] = elapsed;
					raw_srs[n_trial][i_mesh][n_alloc] = sr;
				}
			}
			printf("\n");
		}
	}

	output_ml_data(hmm, N_trials, raw_times, raw_ks, raw_mse, raw_srs, level0_meshes, N1s, sr_tracker, mse_tracker, alloc_counters);

	free(hmm);
	free(weighted_ref);
	free(alloc_counters);
	free(ml_weighted);
	free(raw_ks);
	free(raw_mse);
	free(raw_times);
	free(raw_srs);
	free(sign_ratios);
	free(sample_sizes);

	double total_elapsed = (double) (clock() - timer) / (double) CLOCKS_PER_SEC;
	int hours = (int) floor(total_elapsed / 3600.0);
	int minutes = (int) floor((total_elapsed - hours * 3600) / 60.0);
	int seconds = (int) (total_elapsed - hours * 3600 - minutes * 60);
	printf("Total time for experiment = %d hours, %d minutes and %d seconds\n", hours, minutes, seconds);

	return 0;
}


/* --------------------------------------------------------------------------------------------------------------------
 *
 * Functions
 *
 * ----------------------------------------------------------------------------------------------------------------- */
void output_ml_data(HMM * hmm, int N_trials, double *** raw_times, double *** raw_ks, double *** raw_mse, double *** raw_srs, int * level0_meshes, int * N1s, double * sr_tracker, double ** mse_tracker, int * alloc_counters) {

	/* Initiate output files and print parameters */
	FILE * RAW_TIMES = fopen("raw_times.txt", "w");
	FILE * RAW_KS = fopen("raw_ks.txt", "w");
	FILE * RAW_MSE = fopen("raw_mse.txt", "w");
	FILE * RAW_SRS = fopen("raw_srs.txt", "w");
	FILE * ML_PARAMETERS = fopen("ml_parameters.txt", "w");
	FILE * N1s_DATA = fopen("N1s_data.txt", "w");
	FILE * SR_TRACKER = fopen("sr_tracker.txt", "w");
	FILE * MSE_TRACKER = fopen("mse_tracker.txt", "w");
	FILE * ALLOC_COUNTERS = fopen("alloc_counters.txt", "w");

	/* Write the experiment parameters and mesh sizes */
	fprintf(ML_PARAMETERS, "%d %d %d \n", N_trials, N_ALLOCS, N_MESHES);
	for (int m = 0; m < N_MESHES; m++)
		fprintf(ML_PARAMETERS, "%d ", level0_meshes[m]);
	fprintf(ML_PARAMETERS, "\n");
	fprintf(ML_PARAMETERS, "%d ", hmm->nx);
	for (int n_alloc = 0; n_alloc < N_ALLOCS; n_alloc++)
		fprintf(N1s_DATA, "%d ", N1s[n_alloc]);

	/* Work horizontally from top left to bottom right, writing the result from each trial, new line when finished */
	for (int i_mesh = 0; i_mesh < N_MESHES; i_mesh++) {
		fprintf(ALLOC_COUNTERS, "%d ", alloc_counters[i_mesh]);
		printf("Alloc counters for nx0 = %d = %d\n", level0_meshes[i_mesh], alloc_counters[i_mesh]);
		for (int n_alloc = 0; n_alloc < alloc_counters[i_mesh]; n_alloc++) {
			for (int n_trial = 0; n_trial < N_trials; n_trial++) {

				fprintf(RAW_TIMES, "%e ", raw_times[n_trial][i_mesh][n_alloc]);
				fprintf(RAW_KS, "%e ", raw_ks[n_trial][i_mesh][n_alloc]);
				fprintf(RAW_MSE, "%e ", raw_mse[n_trial][i_mesh][n_alloc]);
				fprintf(RAW_SRS, "%e ", raw_srs[n_trial][i_mesh][n_alloc]);

			}

			fprintf(RAW_TIMES, "\n");
			fprintf(RAW_KS, "\n");
			fprintf(RAW_MSE, "\n");
			fprintf(RAW_SRS, "\n");

		}
	}

	/* This is currently only applicable to a one mesh situation */
	for (int n = 0; n < hmm->length; n++) {
		fprintf(SR_TRACKER, "%e ", sr_tracker[n]);
		for (int n_alloc = 0; n_alloc < alloc_counters[0]; n_alloc++)
			fprintf(MSE_TRACKER, "%e ", mse_tracker[n][n_alloc]);
		fprintf(MSE_TRACKER, "\n");
	}
	
	fclose(RAW_TIMES);
	fclose(RAW_KS);
	fclose(RAW_MSE);
	fclose(RAW_SRS);
	fclose(ML_PARAMETERS);
	fclose(N1s_DATA);
	fclose(SR_TRACKER);
	fclose(MSE_TRACKER);
	fclose(ALLOC_COUNTERS);

}



						// ml_xhat = 0.0;
						// for (int i = 0; i < N_tot; i++)
							// ml_xhat += ml_weighted[n][i].x * ml_weighted[n][i].w;
						// mse_tracker[n][n_alloc] += (ref_xhats[n] - ml_xhat) * (ref_xhats[n] - ml_xhat) / (double) N_trials;