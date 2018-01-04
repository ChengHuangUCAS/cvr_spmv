#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<omp.h>

#define OK 0
#define ERROR 1
#define OVERFLOW 2

#define FIELD_LENGTH 128

typedef struct triple{
	int x;
	int y;
	float val;
}triple_t;

typedef coo{
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
	int **val_ptr;
	int **colidx_ptr;
	int **rec_ptr;
	int *lrrec_ptr;
	int **tail_ptr;
	int ncol;
	int nrow;
	int nnz;
}cvr_t; // compressed vactorization-oriented format

// auxiliary function used in qsort
int func_cmp(const void *a, const void *b){
	triple_t *t1 = (triple_t *)a;
	triple_t *t2 = (triple_t *)b;
	if(t1->x != t2->x){
		return t1->x - t2->x;
	}else{
		return t1->y - t2->y;
	}
}

// auxiliary function to get row number
int func_get_row(int valID, csr_t *csr){
	int start = 0, end = csr->nrow;
	int mid = (start + end) / 2;
	while(start <= end){
		if(csr->row_ptr[mid] > valID){
			end = mid - 1;
		}else if(csr->row_ptr[mid+1] < valID){
			start = mid + 1;
		}else{
			while(csr->row_ptr[mid] == csr->row_ptr[mid+1]){
				mid++;
			}
			return mid;
		}
		mid = (start + end) / 2;
	}
	printf("*** ERROR: a bug occured in func_get_row ***");
	exit(ERROR);
}


// 0-based Matrix Market format -> CSR format
int read_matrix(csr_t *csr, char *filename);
// CSR format -> CVR format
int preprocess(cvr_t *cvr, csr_t *csr);
// CVR format SpMV, y = y + M * x
int spmv(float *y, float *x, cvr_t *cvr);

int n_threads = 4;
int n_lanes = 16;
int n_iterations = 10;

int main(int argc, char **argv){
	csr_t csr;
	cvr_t cvr;
	float *y, *x;

	if(argc < 2){
		printf("ERROR: *** wrong parameter format ***\n");
		return ERROR;
	}
	char *filename = argv[1];
	if(argc > 2){
		n_threads = atoi(argv[2]);
		if(4 == argv){
			n_iterations = atoi(argv[3]);
		}
	}

	if(read_matrix(&csr, filename)){
		printf("ERROR occured in function read_matrix()\n");
		return ERROR;
	}

	if(preprocess(&cvr, &csr)){
		printf("ERROR occured in function preprocess()\n");
		return ERROR;
	}

	x = (float *)malloc(cvr->ncol * sizeof(float));
	y = (float *)malloc(cvr->ncol * sizeof(float));

	if(spmv(y, x, &cvr)){
		printf("ERROR occured in function spmv()\n");
		return ERROR;
	}

	return 0;
}



int read_matrix(csr_t *csr, char *filename){
	FILE *fp = fopen(filename, "r");
	if(NULL == fp){
		printf("ERROR: *** cannot open file: %s ***\n", filename);
		return ERROR;
	}

	char buffer[1024];
	char id[FIELD_LENGTH], object[FIELD_LENGTH], format[FIELD_LENGTH], field[FIELD_LENGTH], symmetry[FIELD_LENGTH];
	int field_pattern = 0, field_complex = 0, symmetry_symmetric = 0;

	//read the header of Matrix Market Format
	if(fgets(buffer, sizeof(buffer), fp)){ 
		sscanf(buffer, "%s %s %s %s %s", id, object, format, field, symmetry);
	}else{
		printf("ERROR: *** empty file: %s ***\n", filename);
		return ERROR;
	}

	//check stored object and format
	if(strcmp(object, "matrix")){
		printf("ERROR: *** file %s does not store a matrix ***\n", filename);
		return ERROR;
	}
	if(strcmp(format, "coordinate")){
		printf("ERROR: *** matrix representation is dense ***\n");
		return ERROR;
	}

	//specific matrix
	if(0 == strcmp(field, "pattern")){
		field_pattern = 1;
	}
	if(0 == strcmp(field, "complex")){
		field_complex = 1;
	}
	if(0 == strcmp(symmetry, "symmetric")){
		symmetry_symmetric = 1;
	}

	//omit comments
	while(!feof(fp)){
		fgets(buffer, sizeof(buffer), fp);
		if('%' != buffer[0]){
			break;
		}
	}

	//number of rows, columns and non-zeros in matrix
	coo_t coo;
	sscanf(buffer, "%d %d %d", &coo->nrow, &coo->ncol, &coo->nnz);
	if(symmetry_symmetric){
		coo->nnz *= 2;
	}
	coo->triple = (triple_t *)malloc(coo->nnz * sizeof(triple_t)); //this pointer is useless out of this function. remember to free it.

	//MMF -> coordinate format
	int i = 0;
	if(symmetry_symmetric){
		if(field_pattern){
			while(!feof(fp)){
				fgets(buffer, sizeof(buffer), fp);
				sscanf(buffer, "%d %d", &coo->triple[i].x, &coo->triple[i].y);
				coo->triple[i].val = 1;
				if(coo->triple[i].x != coo->triple[i].y){
					coo->triple[i+1].x = coo->triple[i].y;
					coo->triple[i+1].y = coo->triple[i].x;
					coo->triple[i+1].val = 1;
					i++;
				}
				i++;
			}
		}else if(field_complex){
			float im;
			while(!feof(fp)){
				fgets(buffer, sizeof(buffer), fp);
				sscanf(buffer, "%d %d %f %f", &coo->triple[i].x, &coo->triple[i].y, &coo->triple[i].val, &im);
				if(coo->triple[i].x != coo->triple[i].y){
					coo->triple[i+1].x = coo->triple[i].y;
					coo->triple[i+1].y = coo->triple[i].x;
					coo->triple[i+1].val = coo->triple[i].val;
					i++;
				}
				i++;
			}
		}else{
			while(!feof(fp)){
				fgets(buffer, sizeof(buffer), fp);
				sscanf(buffer, "%d %d %f", &coo->triple[i].x, &coo->triple[i].y, &coo->triple[i].val);
				if(coo->triple[i].x != coo->triple[i].y){
					coo->triple[i+1].x = coo->triple[i].y;
					coo->triple[i+1].y = coo->triple[i].x;
					coo->triple[i+1].val = coo->triple[i].val;
					i++;
				}
				i++;
			}
		}
	}else{ // if it is not a symmetric matrix
		if(field_pattern){
			while(!feof(fp)){
				fgets(buffer, sizeof(buffer), fp);
				sscanf(buffer, "%d %d", &coo->triple[i].x, &coo->triple[i].y);
				coo->triple[i].val = 1;
				i++;
			}
		}else if(field_complex){
			float im;
			while(!feof(fp)){
				fgets(buffer, sizeof(buffer), fp);
				sscanf(buffer, "%d %d %f %f", &coo->triple[i].x, &coo->triple[i].y, &coo->triple[i].val, &im);
				i++;
			}
		}else{
			while(!feof(fp)){
				fgets(buffer, sizeof(buffer), fp);
				sscanf(buffer, "%d %d %f", &coo->triple[i].x, &coo->triple[i].y, &coo->triple[i].val);
				i++;
			}
		}
	}
	if(i > coo->nnz){
		printf("ERROR: *** too many entries occered ***\n")
		return ERROR;
	}
	printf("\nMatrix is now in coordinate format\n");

	printf("\nMatrix Information:\n");
	printf("Number of rows      : %d\n", coo->nrow);
	printf("Number of columns   : %d\n", coo->ncol);
	printf("Number of non-zeros : %d\n\n", coo->nnz);

	//COO -> CSR
	printf("Coverting to CSR format...\n");

	csr->ncol = coo->ncol;
	csr->nrow = coo->nrow;
	csr->nnz = coo->nnz;
	csr->val = (float *)malloc(csr->nnz * sizeof(float));
	csr->col_idx = (int *)malloc(csr->nnz * sizeof(int));
	csr->row_ptr = (int *)malloc((csr->nrow + 1) * sizeof(int));

	qsort(coo->triple, coo->nnz, sizeof(triple_t), func_cmp);

	csr->row_ptr[0] = 0;
	int r = 0;
	for(i = 0; i < csr->nnz; i++){
		while(coo->triple[i].x != r){
			csr->row_ptr[++r] = i;
		}
		csr->val[i] = coo->triple[i].val;
		csr->col_idx[i] = coo->triple[i].y;
	}
	while(r < csr->nrow){
		csr->row_ptr[++r] = i;
	}
	printf("OK!\n")

	free(coo->triple);

	return OK;
}



int preprocess(cvr_t *cvr, csr_t *csr){
	printf("\nCoverting to CVR format...\n");

	cvr->ncol = csr->ncol;
	cvr->nrow = csr->nrow;
	cvr->nnz = csr->nnz;
	cvr->lrrec_ptr = (int *)malloc(n_threads * sizeof(int));
	cvr->val_ptr = (int **)malloc(n_threads * sizeof(int *));
	cvr->colidx_ptr = (int **)malloc(n_threads * sizeof(int *));
	cvr->rec_ptr = (int **)malloc(n_threads * sizeof(int *));
	cvr->tail_ptr = (int **)malloc(n_threads * sizeof(int *));

	int nnz_per_thread = cvr->nnz / n_threads;
	int change_thread_nnz = cvr->nnz % n_threads;

	#pragma omp parallel num_threads(n_threads)
	{
		int thread_num = omp_get_thread_num();

		int thread_start, thread_end;
		int thread_start_row, thread_end_row;
		//thread whose thread_num is less than change_thread_nnz handle one more non-zero number than the others
		if(thread_num < change_thread_nnz){
			thread_start = thread_num * nnz_per_thread + thread_num * 1;
			thread_end = (thread_num + 1) * nnz_per_thread + (thread_num + 1) * 1;
		}else{
			thread_start = thread_num * nnz_per_thread + change_thread_nnz * 1;
			thread_end = (thread_num + 1) * nnz_per_thread + change_thread_nnz * 1;
		}
		thread_start_row = func_get_row(thread_start, csr);
		thread_end_row = func_get_row(thread_end, csr);

		int *valID = (int *)malloc(n_lanes * sizeof(int));
		int *rowID = (int *)malloc(n_lanes * sizeof(int));
		int *count = (int *)malloc(n_lanes * sizeof(int));
	}

	//-------------------------------
	//  code under here is wrong
	//-------------------------------

	int i;
	int nnz_per_thread = cvr->nnz / n_threads + 1;
	for(i = 0; i < n_threads; i++){
		cvr->val_ptr[i] = (int *)malloc(nnz_per_thread * sizeof(int));
		cvr->colidx_ptr[i] = (int *)malloc(nnz_per_thread * sizeof(int));
	}

	//initialize tracking vectors
	int rs = 0;
	int rec_idx = 0;
	for(i = 0; i < n_threads; i++){
		while(rs < cvr->nrow && csr->row_ptr[rs] == csr->row_ptr[rs+1]){ //skip empty rows
			rs++;
		}
		cvr->valID[i] = csr->row_ptr[rs];
		cvr->rowID[i] = rs;
		cvr->count[i] = csr->row_ptr[rs+1] - csr->row_ptr[rs];
		if(csr->row_ptr[rs+1] == csr->nnz){ //if this is the last row
			n_threads = i + 1;
			break;
		}
		rs++;
	}

	if(csr->row_ptr[rs+1] != csr->nnz){ //conversion and tracker feeding period
		omp_set_num_threads(n_threads);
		omp_set_dynamic(0);
		#pragma omp parallel
		{
			int thread_num = omp_get_thread_num();
			int j = 0;
			do{ //conversion
				cvr->val_ptr[thread_num][j] = csr->val[cvr->valID[thread_num]];
				cvr->colidx_ptr[thread_num][j] = csr->col_idx[cvr->valID[thread_num]];
				cvr->valID[thread_num]++;
				cvr->count[thread_num]--;
				if(0 == cvr->count[thread_num]){ //tracker feeding is needed
					#pragma omp critical //update rs
					{
						if(rs == cvr->nrow){ //other threads reached the last row
							break;
						}
						do{ //skip empty rows
							rs++;
						}while(rs < cvr->nrow && csr->row_ptr[rs] == csr->row_ptr[rs+1]);
						if(rs == cvr->nrow){ //the first thread reached the last row
							cvr->lr_rec = cvr->valID[thread_num];
							break;
						}else{
							cvr->rowID[thread_num] = rs;
						}
					}
					cvr->valID[thread_num] = csr->row_ptr[cvr->rowID[thread_num]];
					cvr->count[thread_num] = csr->row_ptr[cvr->rowID[thread_num]+1] - csr->row_ptr[cvr->rowID[thread_num]];
				}
				j++;
			}while(1);
		}
	}

	//tracker stealing period

	printf("OK!\n");

	return OK;
}



int spmv(float *y, float *x, cvr_t *cvr){

	return OK;
}
