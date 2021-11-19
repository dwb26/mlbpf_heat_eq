#ifndef _SOLVERS_H_
#define _SOLVERS_H_
#endif

void generate_ics(double * ics, double dx, int nx, double alpha, double space_left);

void construct_forward_matrix(double ** A, int nx, double r);

void construct_present_matrix(double ** B, int nx, double r);

void construct_forward(double r, int nx, gsl_vector * lower, gsl_vector * main, gsl_vector * upper);

void construct_present(double r, int nx, gsl_matrix * B);

void crank_nicolson_gsl(int nx, double dt, double s, double T_stop, FILE * CURVE_DATA, gsl_vector * lower, gsl_vector * main, gsl_vector * upper, gsl_matrix * B, gsl_vector * v, gsl_vector * Bv);

void crank_nicolson(int nx, double dt, double s, double T_stop, FILE * CURVE_DATA, double ** B, double * v, double * a_vec, double * b_vec, double * d_vec, double * c_p, double * d_p);

void compute_coefficients(int nx, double r, double * a_vec, double * b_vec, double * c_vec, double * c_p);

void tridiagonal_solve(double * a_vec, double * b_vec, double * d_vec, double * c_p, double * d_p, int nx, double * v);