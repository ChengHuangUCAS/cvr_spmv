//matrix_debug.h
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
