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

extern void print_coo(coo_t *coo);
extern void print_coo_n(coo_t *coo, int n);
extern void print_csr(csr_t *csr);
extern void print_csr_n(csr_t *csr, int n);
extern void print_cvr(cvr_t *cvr);
extern void print_cvr_n(cvr_t *cvr, int n);


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
	int n_threads = 4;
	printf("\ncvr format\n");
	for(i = 0; i < n_threads; i++){
		for(j = 0; j < n / 4; j++){
			printf("%2f ", cvr->val_ptr[i][j]);
		}
		printf("    ");
		for(j = 0; j < n / 4; j++){
			printf("%4d ", cvr->colidx_ptr[i][j]);
		}
		printf("\n");
	}
	printf("\n");
}

void print_cvr(cvr_t *cvr){
	print_cvr_n(cvr, cvr->nnz);
}

