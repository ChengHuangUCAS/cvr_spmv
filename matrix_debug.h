//matrix_debug.h
#include<stdio.h>

typedef struct triple{
	int x;
	int y;
	float val;
}triple_t;

typedef struct coo{
	triple_t *triple;
	int ncol;
	int nrow;
	int nnz;
}coo_t; // coordinate format

typedef struct csr{
	float *val;
	int *col_idx;
	int *row_ptr;
	int ncol;
	int nrow;
	int nnz;
}csr_t; // compressed sparse row format

typedef struct cvr{
	float **val_ptr;
	int **colidx_ptr;
	int **rec_ptr;
	int *lrrec_ptr;
	int **tail_ptr;
	int ncol;
	int nrow;
	int nnz;
}cvr_t; // compressed vactorization-oriented sparse row format

void print_coo(coo_t *coo);
void print_coo_n(coo_t *coo, int n);
void print_csr(csr_t *csr);
void print_csr_n(csr_t *csr, int n);
void print_cvr(cvr_t *cvr);
void print_cvr_n(cvr_t *cvr, int n);

void print_thread(int thread_num);
//void print_cvr_info(cvr_t *cvr);
//void print_cvr_thread_info(cvr_t *cvr, int thread_num);

void print_matrix(csr_t *csr);
void print_vector(float *x, int n);

void print_cvr_detail(cvr_t *cvr, int thread_num, int thread_nnz, int thread_nrow, int n_lanes);


void print_coo_n(coo_t *coo, int n){
	int i;
	printf("\ncoo format:\n");
	for(i = 0; i < n; i++){
		printf("%d %d %f\n", coo->triple[i].x, coo->triple[i].y, coo->triple[i].val);
	}
	printf("\n");
}

void print_coo(coo_t *coo){
	print_coo_n(coo, coo->nnz);
}

void print_csr_n(csr_t *csr, int n){
	int i;
	printf("\ncsr format:\nval: ");
	for(i = 0; i < n; i++){
		printf("%.2f ", csr->val[i]);
	}
	printf("\ncol: ");
	for(i = 0; i < n; i++){
		printf("%4d ", csr->col_idx[i]);
	}
	printf("\nrow: ");
	i = 0;
	while(n > csr->row_ptr[i]){
		printf("%4d ", csr->row_ptr[i]);
		i++;
	}
	printf("\n");
}

void print_csr(csr_t *csr){
	print_csr_n(csr, csr->nnz);
}

void print_cvr_n(cvr_t *cvr, int n){
	int i, j;
	int n_threads = 1;
	printf("\ncvr format\n");
	for(i = 0; i < n_threads; i++){
		printf("   val_ptr: ");
		for(j = 0; j < n / n_threads; j++){
			printf("%.2f ", cvr->val_ptr[i][j]);
		}
		printf("    \ncolidx_ptr: ");
		for(j = 0; j < n / n_threads; j++){
			printf("%4d ", cvr->colidx_ptr[i][j]);
		}
		printf("\n");
	}
	printf("\n");
}

void print_cvr(cvr_t *cvr){
	print_cvr_n(cvr, cvr->nnz);
}



void print_thread(int thread_num){
	printf("this thread = %d\n", thread_num);
}

/*
void print_cvr_info(cvr_t *cvr){
	printf("\ncvr info:\n");
	printf("   val_ptr = %x\n", cvr->val_ptr);
	printf("colidx_ptr = %x\n", cvr->colidx_ptr);
	printf("   rec_ptr = %x\n", cvr->rec_ptr);
	printf(" lrrec_ptr = %x\n", cvr->lrrec_ptr);
	printf("  tail_ptr = %x\n", cvr->tail_ptr);
	printf("nrow = %d, ncol = %d, nnz = %d\n\n", cvr->nrow, cvr->ncol, cvr->nnz);
}

void print_cvr_thread_info(cvr_t *cvr, int thread_num){
	printf("\ncvr thread info:\n");
	printf("cvr->val_ptr[%d] = %x, ", thread_num, cvr->val_ptr[thread_num]);
	printf("cvr->colidx_ptr[%d] = %x, ", thread_num, cvr->colidx_ptr[thread_num]);
	printf("cvr->rec_ptr[%d] = %x, ", thread_num, cvr->rec_ptr[thread_num]);
	printf("cvr->tail_ptr[%d] = %x\n", thread_num, cvr->tail_ptr[thread_num]);
}
*/

void print_matrix(csr_t *csr){
	int i, j, col = 0;
	printf("\nmatrix:\n");
	for(i = 0; i < csr->nrow; i++){
		for(j = 0; j < csr->ncol; j++){
			if(col >= csr->row_ptr[i] && col < csr->row_ptr[i+1] && j == csr->col_idx[col]){
				printf("%.2f ", csr->val[col]);
				col++;
			}else{
				printf("0.00 ");
			}
		}
		printf("\n");
	}
}
void print_vector(float *x, int n){
	int i;
	printf("\nvector:\n");
	for(i = 0; i < n; i++){
		printf("%.2f \n", x[i]);
	}
}

void print_cvr_detail(cvr_t *cvr, int thread_num, int thread_nnz, int thread_nrow, int n_lanes){
	int i;
	printf("\ncvr detail: thread %d\n", thread_num);

	printf("  *val_ptr:    ");
	int n_vals = (thread_nnz + n_lanes - 1) / n_lanes * n_lanes;
	for(i = 0; i < n_vals; i++){
		printf("%.2f ", cvr->val_ptr[thread_num][i]);
	}
	printf("\n");

	printf("  *colidx_ptr: ");
	for(i = 0; i < n_vals; i++){
		printf("%4d ", cvr->colidx_ptr[thread_num][i]);
	}
	printf("\n");

	printf("  *rec_ptr: ");
	int n_recs = (thread_nrow + n_lanes - 1) / n_lanes * n_lanes;
	for(i = 0; i < n_recs; i++){
		printf("%2d ", cvr->rec_ptr[thread_num][2*i]);
	}
	printf("\n            ");
	for(i = 0; i < n_recs; i++){
		printf("%2d ", cvr->rec_ptr[thread_num][2*i+1]);
	}
	printf("\n");

	printf("  lrrec_ptr: ");
	printf("%d ", cvr->lrrec_ptr[thread_num]);
	printf("\n");

	printf("  *tail_ptr: ");
	for(i = 0; i < n_lanes; i++){
		printf("%d ", cvr->tail_ptr[thread_num][i]);
	}
	printf("\n");

	printf("  nrow = %d, ncol = %d, nnz = %d\n\n", cvr->nrow, cvr->ncol, cvr->nnz);
}

