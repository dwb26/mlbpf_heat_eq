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
#include <gsl/gsl_interp.h>
#include "solvers.h"
#include "particle_filters.h"


int weighted_double_cmp(const void * a, const void * b) {

	struct weighted_double d1 = * (struct weighted_double *) a;
	struct weighted_double d2 = * (struct weighted_double *) b;

	if (d1.x < d2.x)
		return -1;
	if (d2.x < d1.x)
		return 1;
	return 0;
}


double * construct_space_mesh(size_t size, double space_left, double dx, int nx) {
	double * xs = (double *) malloc(size);
	xs[0] = space_left;
	for (int j = 1; j < nx; j++)
		xs[j] = space_left + j * dx;
	return xs;
}


void regression(double * corrections, double * s, int N0, int N1, double * theta) {

	double alpha, alpha_top = 0.0, alpha_bottom = 0.0, beta = 0.0, s_bar = 0.0, h_bar = 0.0;
	for (int i = 0; i < N1; i++) {
		s_bar += s[i + N0];
		h_bar += corrections[i];
	}
	s_bar /= (double) N1;
	h_bar /= (double) N1;

	for (int i = 0; i < N1; i++) {
		alpha_top += ((s[i + N0] - s_bar) * (corrections[i] - h_bar));
		alpha_bottom += ((s[i + N0] - s_bar) * (s[i + N0] - s_bar));
	}
	alpha = alpha_top / alpha_bottom;
	beta = h_bar - alpha * s_bar;
	theta[0] = alpha, theta[1] = beta;
}


void resample(long size, double * w, long * ind, gsl_rng * r) {

	/* Generate the exponentials */
	double * e = (double *) malloc((size + 1) * sizeof(double));
	double g = 0;
	for (long i = 0; i <= size; i++) {
		e[i] = gsl_ran_exponential(r, 1.0);
		g += e[i];
	}
	/* Generate the uniform order statistics */
	double * u = (double *) malloc((size + 1) * sizeof(double));
	u[0] = 0;
	for (long i = 1; i <= size; i++)
		u[i] = u[i - 1] + e[i - 1] / g;

	/* Do the actual sampling with C_inv_gsl cdf */
	double cdf = w[0];
	long j = 0;
	for (long i = 0; i < size; i++) {
		while (cdf < u[i + 1]) {
			j++;
			cdf += w[j];
		}
		ind[i] = j;
	}

	free(e);
	free(u);
}


void mutate(int N_tot, double * s, double * s_res, double sig_sd, gsl_rng * rng, int n) {
	for (int i = 0; i < N_tot; i++)
		s[i] = -1 * s_res[i] + gsl_ran_gaussian(rng, sig_sd);
}


void ml_bootstrap_particle_filter_gsl(HMM * hmm, int * sample_sizes, int * nxs, gsl_rng * rng, w_double ** ml_weighted, double * sign_ratios, double * ref_xhats) {


	/* --------------------------------------------------- Setup --------------------------------------------------- */
	/* ------------------------------------------------------------------------------------------------------------- */

	/* General parameters */
	/* ------------------ */
	int length = hmm->length;
	int nx0 = nxs[0], nx1 = nxs[1];
	int nt = hmm->nt;
	int obs_pos0 = (int) (0.5 * (nx0 + 1) - 1);
	int obs_pos1 = (int) (0.5 * (nx1 + 1) - 1);
	int lag = hmm->lag, start_point = 0, counter0, counter1;
	int N0 = sample_sizes[0], N1 = sample_sizes[1], N_tot = N0 + N1;
	double sign_rat = 0.0, coarse_scaler = 0.5;
	double sig_sd = hmm->sig_sd, obs_sd = hmm->obs_sd;
	double space_left = hmm->space_left, space_right = hmm->space_right;
	double space_length = space_right - space_left;
	double T_stop = hmm->T_stop;
	double dx0 = space_length / (double) (nx0 - 1);
	double dx1 = space_length / (double) (nx1 - 1);
	double dt = T_stop / (double) (nt - 1);
	double alpha = hmm->alpha;
	double r0 = alpha * dt / (dx0 * dx0), r1 = alpha * dt / (dx1 * dx1);
	double obs, normaliser, abs_normaliser, g0, g1, ml_xhat;	
	short * signs = (short *) malloc(N_tot * sizeof(double));
	short * res_signs = (short *) malloc(N_tot * sizeof(short));
	long * ind = (long *) malloc(N_tot * sizeof(long));
	double * s = (double *) malloc(N_tot * sizeof(double));
	double * s_res = (double *) malloc(N_tot * sizeof(double));
	double * weights = (double *) malloc(N_tot * sizeof(double));
	double * absolute_weights = (double *) malloc(N_tot * sizeof(double));
	double * h1s = (double *) malloc(N1 * sizeof(double));
	double * h0s = (double *) malloc(N1 * sizeof(double));
	double * corrections = (double *) malloc(N1 * sizeof(double));
	double * theta = (double *) calloc(2, sizeof(double));	
	double * ml_xhats = (double *) malloc(length * sizeof(double));
	double * v0_temp = (double *) malloc(nx0 * sizeof(double));
	double * v1_temp = (double *) malloc(nx1 * sizeof(double));
	double * xs0 = construct_space_mesh(nx0 * sizeof(double), space_left, dx0, nx0);
	double * xs1 = construct_space_mesh(nx1 * sizeof(double), space_left, dx1, nx1);
	double ** X = (double **) malloc(N_tot * sizeof(double *));
	for (int i = 0; i < N_tot; i++)
		X[i] = (double *) malloc((lag + 1) * sizeof(double));
	

	/* Level 0 solver matrix construction */
	/* ---------------------------------- */
	/* Construct the forward time matrix */
	gsl_vector * lower0 = gsl_vector_alloc(nx0 - 1);
	gsl_vector * main0 = gsl_vector_alloc(nx0);
	gsl_vector * upper0 = gsl_vector_alloc(nx0 - 1);
	construct_forward(r0, nx0, lower0, main0, upper0);

	/* Construct the present time matrix */
	gsl_matrix * B0 = gsl_matrix_calloc(nx0, nx0);
	gsl_vector * v0 = gsl_vector_alloc(nx0);
	gsl_vector * Bv0 = gsl_vector_alloc(nx0);
	construct_present(r0, nx0, B0);


	/* Level 1 solver matrix construction */
	/* ---------------------------------- */
	/* Construct the forward time matrix */
	gsl_vector * lower1 = gsl_vector_alloc(nx1 - 1);
	gsl_vector * main1 = gsl_vector_alloc(nx1);
	gsl_vector * upper1 = gsl_vector_alloc(nx1 - 1);
	construct_forward(r1, nx1, lower1, main1, upper1);

	/* Construct the present time matrix */
	gsl_matrix * B1 = gsl_matrix_calloc(nx1, nx1);
	gsl_vector * v1 = gsl_vector_alloc(nx1);
	gsl_vector * Bv1 = gsl_vector_alloc(nx1);
	construct_present(r1, nx1, B1);


	/* Initial conditions */
	/* ------------------ */
	double s0 = hmm->signal[0];
	generate_ics(v0_temp, dx0, nx0, alpha, space_left);
	generate_ics(v1_temp, dx1, nx1, alpha, space_left);
	for (int j = 0; j < nx0; j++)
		v0->data[j] = v0_temp[j];
	for (int j = 0; j < nx1; j++)
		v1->data[j] = v1_temp[j];
	for (int i = 0; i < N_tot; i++) {
		s[i] = gsl_ran_gaussian(rng, sig_sd) + s0;
		X[i][0] = s[i];
		res_signs[i] = 1;
	}
	gsl_interp * ics_interp = gsl_interp_alloc(gsl_interp_linear, nx1);
	gsl_interp_accel * acc = gsl_interp_accel_alloc();

	/* Output files */
	FILE * CURVE_DATA = fopen("curve_data.txt", "w");
	FILE * ML_XHATS = fopen("ml_xhats.txt", "w");



	/* ---------------------------------------------- Time iterations ---------------------------------------------- */
	/* ------------------------------------------------------------------------------------------------------------- */
	for (int n = 0; n < length; n++) {

		obs = hmm->observations[n];
		if (n > lag)
			start_point++;


		/* --------------------------------------------------------------------------------------------------------- */
		/*																											 */
		/* Level 1 solutions																						 */
		/*																											 */
		/* --------------------------------------------------------------------------------------------------------- */
		normaliser = 0.0, abs_normaliser = 0.0;
		for (int i = N0; i < N_tot; i++) {


			/* Fine solution */
			/* ------------- */
			/* Fine solve with respect to the historical particles */
			counter1 = 0;
			for (int m = start_point; m < n; m++) {
				crank_nicolson_gsl(nx1, dt, X[i][counter1], T_stop, CURVE_DATA, lower1, main1, upper1, B1, v1, Bv1);
				counter1++;
			}
			crank_nicolson_gsl(nx1, dt, s[i], T_stop, CURVE_DATA, lower1, main1, upper1, B1, v1, Bv1);
			h1s[i - N0] = v1->data[obs_pos1];			


			/* Coarse solution */
			/* --------------- */
			/* Coarse solve with respect to the historical particles */
			counter0 = 0;
			for (int m = start_point; m < n; m++) {
				crank_nicolson_gsl(nx0, dt, X[i][counter0], T_stop, CURVE_DATA, lower0, main0, upper0, B0, v0, Bv0);
				counter0++;
			}
			crank_nicolson_gsl(nx0, dt, s[i], T_stop, CURVE_DATA, lower0, main0, upper0, B0, v0, Bv0);
			h0s[i - N0] = v0->data[obs_pos0];


			/* Reset the initial conditions to the current time level for the next particle weighting */
			corrections[i - N0] = h1s[i - N0] - h0s[i - N0];
			for (int j = 0; j < nx1; j++)
				v1->data[j] = v1_temp[j];
			for (int j = 0; j < nx0; j++)
				v0->data[j] = v0_temp[j];

		}

		if (N1 > 0)
			regression(corrections, s, N0, N1, theta);


		/* --------------------------------------------------------------------------------------------------------- */
		/*																											 */
		/* Level 1 weight generation																				 */
		/*																											 */
		/* --------------------------------------------------------------------------------------------------------- */
		for (int i = N0; i < N_tot; i++) {

			g1 = gsl_ran_gaussian_pdf(h1s[i - N0] - obs, obs_sd);
			g0 = gsl_ran_gaussian_pdf(h0s[i - N0] + theta[0] * s[i] + theta[1] - obs, obs_sd);

			weights[i] = (g1 - g0) * (double) res_signs[i] / (double) N1;
			absolute_weights[i] = fabs(weights[i]);
			normaliser += weights[i];
			abs_normaliser += absolute_weights[i];

		}



		/* --------------------------------------------------------------------------------------------------------- */
		/*																											 */
		/* Level 0 weight generation																				 */
		/*																											 */
		/* --------------------------------------------------------------------------------------------------------- */
		for (int i = 0; i < N0; i++) {


			/* Coarse solution */
			/* --------------- */
			/* Coarse solve with respect to the historical particles */
			counter0 = 0;
			for (int m = start_point; m < n; m++) {
				crank_nicolson_gsl(nx0, dt, X[i][counter0], T_stop, CURVE_DATA, lower0, main0, upper0, B0, v0, Bv0);
				counter0++;
			}
			crank_nicolson_gsl(nx0, dt, s[i], T_stop, CURVE_DATA, lower0, main0, upper0, B0, v0, Bv0);			
			g0 = gsl_ran_gaussian_pdf(v0->data[obs_pos0] + theta[0] * s[i] + theta[1] - obs, obs_sd);


			/* Weight computation */
			/* ------------------ */
			weights[i] = g0 * (double) res_signs[i] / (double) N0;
			absolute_weights[i] = fabs(weights[i]);
			normaliser += weights[i];
			abs_normaliser += absolute_weights[i];

			/* Set the initial conditions to the current time level for the next particle weighting */
			for (int j = 0; j < nx0; j++)
				v0->data[j] = v0_temp[j];

		}



		/* --------------------------------------------------------------------------------------------------------- */
		/*																											 */
		/* Normalisation 																							 */
		/*																											 */
		/* --------------------------------------------------------------------------------------------------------- */
		ml_xhat = 0.0, sign_rat = 0.0;
		for (int i = 0; i < N_tot; i++) {
			absolute_weights[i] /= abs_normaliser;
			weights[i] /= normaliser;
			signs[i] = weights[i] < 0 ? -1 : 1;
			ml_weighted[n][i].x = s[i];
			ml_weighted[n][i].w = weights[i];
			ml_xhat += s[i] * weights[i];
			sign_rat += signs[i] / (double) N_tot;
		}
		ml_xhats[n] = ml_xhat;
		sign_ratios[n] = sign_rat;
		fprintf(ML_XHATS, "%e ", ml_xhat);



		/* --------------------------------------------------------------------------------------------------------- */
		/*																											 */
		/* Resample and mutate 																						 */
		/*																											 */
		/* --------------------------------------------------------------------------------------------------------- */
		resample(N_tot, absolute_weights, ind, rng);
		for (int i = 0; i < N_tot; i++) {
			s_res[i] = s[ind[i]];
			res_signs[i] = signs[ind[i]];
		}
		mutate(N_tot, s, s_res, sig_sd, rng, n);

		if (n < lag) {

			/* Load the mutated particles into the historical particle array */
			for (int i = 0; i < N_tot; i++)
				X[i][n + 1] = s[i];

		}
		else {

			/* Shift the particles left one position and lose the oldest ancestor */
			for (int m = 0; m < lag; m++) {
				for (int i = 0; i < N_tot; i++)
					X[i][m] = X[i][m + 1];
			}

			/* Load the mutated particles into the vacant entry in the historical particle array */
			for (int i = 0; i < N_tot; i++)
				X[i][lag] = s[i];

			/* Evolve the fine initial condition with respect to the (n - lag)-th MSE-minimising point estimate */
			crank_nicolson_gsl(nx1, dt, ml_xhats[n - lag], T_stop, CURVE_DATA, lower1, main1, upper1, B1, v1, Bv1);
			for (int j = 0; j < nx1; j++)
				v1_temp[j] = v1->data[j];

			/* Interpolate the coarse initial condition */
			gsl_interp_init(ics_interp, xs1, v1_temp, nx1);
			for (int j = 0; j < nx0; j++)
				v0->data[j] = gsl_interp_eval(ics_interp, xs1, v1_temp, xs0[j], acc);
			for (int j = 0; j < nx0; j++)
				v0_temp[j] = v0->data[j];

		}		

	}

	fclose(CURVE_DATA);
	fclose(ML_XHATS);
	fclose(CORRECTIONS);

	free(signs);
	free(res_signs);
	free(ind);
	free(s);
	free(s_res);
	free(weights);
	free(absolute_weights);
	free(h1s);
	free(h0s);
	free(corrections);
	free(theta);
	free(ml_xhats);
	free(v0_temp);
	free(v1_temp);
	free(X);
	free(xs0);
	free(xs1);

	gsl_interp_free(ics_interp);
	gsl_interp_accel_free(acc);

	gsl_vector_free(lower0);
	gsl_vector_free(main0);
	gsl_vector_free(upper0);
	gsl_matrix_free(B0);
	gsl_vector_free(v0);
	gsl_vector_free(Bv0);
	gsl_vector_free(lower1);
	gsl_vector_free(main1);
	gsl_vector_free(upper1);	
	gsl_matrix_free(B1);
	gsl_vector_free(v1);
	gsl_vector_free(Bv1);

}


void bootstrap_particle_filter_gsl(HMM * hmm, int N, gsl_rng * rng, w_double ** weighted, double * data) {


	/* --------------------------------------------------- Setup --------------------------------------------------- */
	/* ------------------------------------------------------------------------------------------------------------- */
	/* Particle filter parameters */
	int length = hmm->length;
	int nx = hmm->nx;
	int nt = hmm->nt;
	int obs_pos = (int) (0.5 * (nx + 1) - 1);
	int lag = hmm->lag, start_point = 0, counter;
	double sig_sd = hmm->sig_sd, obs_sd = hmm->obs_sd;
	double space_left = hmm->space_left, space_right = hmm->space_right;
	double T_stop = hmm->T_stop;
	double dx = (space_right - space_left) / (double) (nx - 1);
	double dt = T_stop / (double) (nt - 1);
	double alpha = hmm->alpha;
	double r = alpha * dt / (dx * dx);
	double obs, normaliser, x_hat;
	size_t size = nx * sizeof(double);
	long * ind = (long *) malloc(N * sizeof(long));
	double * s = (double *) malloc(N * sizeof(double));
	double * s_res = (double *) malloc(N * sizeof(double));
	double * weights = (double *) malloc(N * sizeof(double));
	double * x_hats = (double *) malloc(length * sizeof(double));
	double * u = (double *) malloc(nx * sizeof(double));
	double ** X = (double **) malloc(N * sizeof(double *));
	for (int i = 0; i < N; i++)
		X[i] = (double *) malloc((lag + 1) * sizeof(double));

	/* Construct the forward time matrix */
	gsl_vector * lower = gsl_vector_alloc(nx - 1);
	gsl_vector * main = gsl_vector_alloc(nx);
	gsl_vector * upper = gsl_vector_alloc(nx - 1);
	construct_forward(r, nx, lower, main, upper);

	/* Construct the present time matrix */
	gsl_matrix * B = gsl_matrix_calloc(nx, nx);
	gsl_vector * v = gsl_vector_alloc(nx);
	gsl_vector * Bv = gsl_vector_alloc(nx);
	construct_present(r, nx, B);

	/* Initial conditions */
	double s0 = hmm->signal[0];
	generate_ics(u, dx, nx, alpha, space_left);
	for (int j = 0; j < nx; j++)
		v->data[j] = u[j];
	for (int i = 0; i < N; i++) {
		s[i] = gsl_ran_gaussian(rng, sig_sd) + s0;
		X[i][0] = s[i];
	}

	FILE * CURVE_DATA = fopen("curve_data.txt", "w");
	FILE * X_HATS = fopen("x_hats.txt", "w");




	/* ---------------------------------------------- Time iterations ---------------------------------------------- */
	/* ------------------------------------------------------------------------------------------------------------- */
	for (int n = 0; n < length; n++) {

		/* Read in the observation that the particles will be weighted on */
		obs = hmm->observations[n];
		if (n > lag)
			start_point++;



		/* --------------------------------------------------------------------------------------------------------- */
		/*																											 */
		/* Weight generation																						 */
		/*																											 */
		/* --------------------------------------------------------------------------------------------------------- */
		normaliser = 0.0;
		for (int i = 0; i < N; i++) {

			/* Solve with respect to the historical particles */
			counter = 0;
			for (int m = start_point; m < n; m++) {
				crank_nicolson_gsl(nx, dt, X[i][counter], T_stop, CURVE_DATA, lower, main, upper, B, v, Bv);
				counter++;
			}

			/* Generate the weight */
			crank_nicolson_gsl(nx, dt, s[i], T_stop, CURVE_DATA, lower, main, upper, B, v, Bv);
			weights[i] = gsl_ran_gaussian_pdf(v->data[obs_pos] - obs, obs_sd);
			normaliser += weights[i];

			for (int j = 0; j < nx; j++)
				v->data[j] = u[j];

		}



		/* --------------------------------------------------------------------------------------------------------- */
		/*																											 */
		/* Normalisation 																							 */
		/*																											 */
		/* --------------------------------------------------------------------------------------------------------- */
		x_hat = 0.0;
		for (int i = 0; i < N; i++) {
			weights[i] /= normaliser;
			weighted[n][i].x = s[i];
			weighted[n][i].w = weights[i];
			x_hat += s[i] * weights[i];
		}
		x_hats[n] = x_hat;
		fprintf(X_HATS, "%e ", x_hat);
		printf("n = %d, x_hat = %lf\n", n, x_hat);



		/* --------------------------------------------------------------------------------------------------------- */
		/*																											 */
		/* Resample and mutate 																						 */
		/*																											 */
		/* --------------------------------------------------------------------------------------------------------- */
		resample(N, weights, ind, rng);
		for (int i = 0; i < N; i++)
			s_res[i] = s[ind[i]];
		mutate(N, s, s_res, sig_sd, rng, n);

		if (n < lag) {
			for (int i = 0; i < N; i++)
				X[i][n + 1] = s[i];
		}
		else {
			/* Shift the particles left one position and drop the far left ones */
			for (int m = 0; m < lag; m++) {
				for (int i = 0; i < N; i++)
					X[i][m] = X[i][m + 1];
			}
			for (int i = 0; i < N; i++)
				X[i][lag] = s[i];

			/* Initial condition update */
			crank_nicolson_gsl(nx, dt, x_hats[n - lag], T_stop, CURVE_DATA, lower, main, upper, B, v, Bv);
			for (int j = 0; j < nx; j++)
				u[j] = v->data[j];
		}

	}
	fclose(CURVE_DATA);
	fclose(X_HATS);

	free(ind);
	free(s);
	free(s_res);
	free(weights);
	free(x_hats);
	free(u);
	free(X);

}



		// printf("log10 mse = %.8lf, sign_rat = %.6lf\n", log10((ml_xhats[n] - ref_xhats[n]) * (ml_xhats[n] - ref_xhats[n]) / (double) length), sign_rat);


	// char nx0_str[50], name[100];
	// snprintf(nx0_str, 50, "%d", nx0);
	// sprintf(name, "corrections_nx0=%s.txt", nx0_str);
	// FILE * CORRECTIONS = fopen(name, "w");

		// for (int i = N0; i < N_tot; i++)
			// fprintf(CORRECTIONS, "%e ", s[i]);
		// fprintf(CORRECTIONS, "\n");
		// for (int i = N0; i < N_tot; i++)
			// fprintf(CORRECTIONS, "%e ", corrections[i - N0]);
		// fprintf(CORRECTIONS, "\n");

	// char nx0_str[50], N1_str[50], name[100], w_name[100];
	// snprintf(N1_str, 50, "%d", N1);
	// snprintf(nx0_str, 50, "%d", nx0);
	// sprintf(w_name, "ml_w_dist_N1=%s_nx0=%s.txt", N1_str, nx0_str);
	// FILE * W_DISTS = fopen(w_name, "w");
	// fprintf(W_DISTS, "%d\n", N_tot);
	// fprintf(W_DISTS, "%d\n", N0);
		// double w_sum = 0.0;
		// for (int i = 0; i < N0; i++)
		// 	fprintf(W_DISTS, "%.16lf ", s[i]);
		// fprintf(W_DISTS, "\n");
		// for (int i = 0; i < N0; i++)
		// 	fprintf(W_DISTS, "%.16lf ", weights[i]);
		// fprintf(W_DISTS, "\n");