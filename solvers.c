#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_blas.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "solvers.h"
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>


void generate_ics(double * ics, double dx, int nx, double alpha, double space_left) {
	double x = space_left;
	for (int j = 0; j < nx; j++) {
		ics[j] = alpha * sin(M_PI * x);
		x += dx;
	}
}


void compute_coefficients(int nx, double r, double * a_vec, double * b_vec, double * c_vec, double * c_p) {

	/* Sub diagonal */
	for (int i = 0; i < nx - 1; i++)
		a_vec[i] = -r;
	a_vec[nx - 2] = 0.0;

	/* Main diagonal */
	for (int i = 0; i < nx; i++)
		b_vec[i] = 2 + 2 * r;
	b_vec[0] = 1.0, b_vec[nx - 1] = 1.0;

	/* Super diagonal */
	for (int i = 0; i < nx - 1; i++)
		c_vec[i] = -r;
	c_vec[0] = 0.0;

	/* Transform the super diagonal */
	c_p[0] = c_vec[0] / b_vec[0];
	for (int i = 1; i < nx - 1; i++)
		c_p[i] = c_vec[i] / (b_vec[i] - a_vec[i] * c_p[i - 1]);

}


void construct_present_matrix(double ** B, int nx, double r) {

	for (int n = 0; n < nx; n++)
		B[n] = (double *) calloc(nx, sizeof(double));

	/* Main diagonal */
	for (int i = 0; i < nx; i++)
		B[i][i] = 2 - 2 * r;
	B[0][0] = 1.0, B[nx - 1][nx - 1] = 1.0;

	/* Sub diagonal */
	for (int i = 1; i < nx; i++)
		B[i][i - 1] = r;
	B[nx - 1][nx - 2] = 0.0;

	/* Super diagonal */
	for (int i = 0; i < nx - 1; i++)
		B[i][i + 1] = r;
	B[0][1] = 0.0;

}


void construct_forward(double r, int nx, gsl_vector * lower, gsl_vector * main, gsl_vector * upper) {

	/* Construct the sub diagonals */
	for (int n = 0; n < nx - 1; n++) {
		lower->data[n] = -r;
		upper->data[n] = -r;		
	}
	lower->data[nx - 2] = 0.0;
	upper->data[0] = 0.0;

	/* Construct the main diagonal */
	for (int n = 0; n < nx; n++)
		main->data[n] = 2 + 2 * r;
	main->data[0] = 1.0, main->data[nx - 1] = 1.0;	

}


void construct_present(double r, int nx, gsl_matrix * B) {

	/* Construct the sub diagonals */
	for (int n = 1; n < nx; n++)
		gsl_matrix_set(B, n, n - 1, r);
	gsl_matrix_set(B, nx - 1, nx - 2, 0.0);

	for (int n = 0; n < nx - 1; n++)
		gsl_matrix_set(B, n, n + 1, r);
	gsl_matrix_set(B, 0, 1, 0.0);

	/* Construct the main diagonal */
	for (int n = 1; n < nx - 1; n++)
		gsl_matrix_set(B, n, n, 2 - 2 * r);
	gsl_matrix_set(B, 0, 0, 1.0); 
	gsl_matrix_set(B, nx - 1, nx - 1, 1.0);
	
}


void crank_nicolson_gsl(int nx, double dt, double s, double T_stop, FILE * CURVE_DATA, gsl_vector * lower, gsl_vector * main, gsl_vector * upper, gsl_matrix * B, gsl_vector * v, gsl_vector * Bv) {

	/* Set the left boundary condition */
	v->data[0] = s;

	/* Construct and solve the system */
	double t = 0.0;
	while (t < T_stop) {
		// for (int j = 0; j < nx; j++)
			// fprintf(CURVE_DATA, "%lf ", v->data[j]);
		// fprintf(CURVE_DATA, "\n");
		gsl_blas_dgemv(CblasNoTrans, 1.0, B, v, 0.0, Bv);
		gsl_linalg_solve_tridiag(main, upper, lower, Bv, v);
		t += dt;
	}
	
}


void crank_nicolson(int nx, double dt, double s, double T_stop, FILE * CURVE_DATA, double ** B, double * v, double * a_vec, double * b_vec, double * d_vec, double * c_p, double * d_p) {

	/* Set the left boundary condition */
	v[0] = s;

	/* Construct and solve the system over the integration time */
	double t = 0.0;
	while (t < T_stop) {
		// for (int j = 0; j < nx; j++)
			// fprintf(CURVE_DATA, "%lf ", v[j]);
		// fprintf(CURVE_DATA, "\n");

		/* Compute Bv */
		for (int j = 0; j < nx; j++) {
			d_vec[j] = 0.0;
			for (int k = 0; k < nx; k++)
				d_vec[j] += B[j][k] * v[k];
		}

		/* Solve Av = d */
		tridiagonal_solve(a_vec, b_vec, d_vec, c_p, d_p, nx, v);
		t += dt;

	}

}


void tridiagonal_solve(double * a_vec, double * b_vec, double * d_vec, double * c_p, double * d_p, int nx, double * v) {

	/* Translate the target vector */
	d_p[0] = d_vec[0] / b_vec[0];
	for (int i = 1; i < nx; i++)
		d_p[i] = (d_vec[i] - a_vec[i] * d_p[i - 1]) / (b_vec[i] - a_vec[i] * c_p[i - 1]);

	/* Apply the backwards substitution formula */
	v[nx - 1] = d_p[nx - 1];
	for (int i = nx - 2; i >= 0; i--)
		v[i] = d_p[i] - c_p[i] * v[i + 1];

}





