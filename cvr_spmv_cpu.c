#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<omp.h>

#define OK 0
#define ERROR 1
#define OVERFLOW 2

#define FIELD_LENGTH 128
#define COO_BASE 0

#define DEBUG
#ifdef DEBUG
#include"matrix_debug.h"
#else
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
#endif

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
			while(mid < csr->nrow && csr->row_ptr[mid] == csr->row_ptr[mid+1]){
				mid++;
			}
			return mid;
		}
		mid = (start + end) / 2;
	}
	printf("*** ERROR: a bug occured in func_get_row ***\n");
	exit(ERROR);
}

// auxiliary function to get AND result
int func_AND(int *val, int n){
	int result = 1, i;
	for(i = 0; i < n; i++){
		result = result && val[i];
	}
	return result;
}

// auxiliary function to get average count
int func_average(int *count, int n){
	int sum = 0, i;
	for(i = 0; i < n; i++){
		sum += count[i];
	}
	return sum / n;
}


// 0-based Matrix Market format -> CSR format
int read_matrix(csr_t *csr, char *filename);
// CSR format -> CVR format
int preprocess(cvr_t *cvr, csr_t *csr);
// CVR format SpMV, y = y + M * x, parameter csr is only used in func_get_row
int spmv(float *y, float *x, cvr_t *cvr, csr_t *csr);

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
		if(4 == argc){
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

	x = (float *)malloc(cvr.ncol * sizeof(float));
	y = (float *)malloc(cvr.ncol * sizeof(float));
	int i, j, iteration;
	for(i = 0; i < cvr.ncol; i++){
		x[i] = i % 1000;
	}
	memset(y, 0, cvr.ncol * sizeof(float));

	if(spmv(y, x, &cvr, &csr)){
		printf("ERROR occured in function spmv()\n");
		return ERROR;
	}

	float *y_verify = (float *)malloc(csr.ncol * sizeof(float));
	float sum;
	memset(y_verify, 0, csr.ncol * sizeof(float));
	for(iteration = 0; iteration < n_iterations; iteration++){
		for(i = 0; i < csr.nrow; i++){
			sum = 0;
			for(j = csr.row_ptr[i]; j < csr.row_ptr[i+1]; j++){
				sum += csr.val[j] * x[csr.col_idx[j]];
			}
			y_verify[i] += sum;
		}
	}

	int count = 0;
	for(i = 0; i < csr.ncol; i++){
		if(y[i] != y_verify[i]){
			if(count > 10){
				break;
			}
			printf("%dth number of vector y should be %f, but the result is %f\n", i, y_verify[i], y[i]);
		}
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
	sscanf(buffer, "%d %d %d", &coo.nrow, &coo.ncol, &coo.nnz);
	if(symmetry_symmetric){
		coo.nnz *= 2;
	}
	coo.triple = (triple_t *)malloc(coo.nnz * sizeof(triple_t)); //this pointer is useless out of this function. remember to free it.
printf("row=%d, col=%d, nnz=%d\n", coo.nrow, coo.ncol, coo.nnz);
	//MMF -> coordinate format
	int i = 0;
	if(symmetry_symmetric){
		if(field_pattern){
			for(i = 0; i < coo.nnz; i++){
				fgets(buffer, sizeof(buffer), fp);
				sscanf(buffer, "%d %d", &coo.triple[i].x, &coo.triple[i].y);
				coo.triple[i].val = 1;
				if(coo.triple[i].x != coo.triple[i].y){
					coo.triple[i+1].x = coo.triple[i].y;
					coo.triple[i+1].y = coo.triple[i].x;
					coo.triple[i+1].val = 1;
					i++;
				}
			}
		}else if(field_complex){
			float im;
			for(i = 0; i < coo.nnz; i++){
				fgets(buffer, sizeof(buffer), fp);
				sscanf(buffer, "%d %d %f %f", &coo.triple[i].x, &coo.triple[i].y, &coo.triple[i].val, &im);
				if(coo.triple[i].x != coo.triple[i].y){
					coo.triple[i+1].x = coo.triple[i].y;
					coo.triple[i+1].y = coo.triple[i].x;
					coo.triple[i+1].val = coo.triple[i].val;
					i++;
				}
			}
		}else{
			for(i = 0; i < coo.nnz; i++){
				fgets(buffer, sizeof(buffer), fp);
				sscanf(buffer, "%d %d %f", &coo.triple[i].x, &coo.triple[i].y, &coo.triple[i].val);
				if(coo.triple[i].x != coo.triple[i].y){
					coo.triple[i+1].x = coo.triple[i].y;
					coo.triple[i+1].y = coo.triple[i].x;
					coo.triple[i+1].val = coo.triple[i].val;
					i++;
				}
			}
		}
	}else{ // if it is not a symmetric matrix
		if(field_pattern){
			for(i = 0; i < coo.nnz; i++){
				fgets(buffer, sizeof(buffer), fp);
				sscanf(buffer, "%d %d", &coo.triple[i].x, &coo.triple[i].y);
				coo.triple[i].val = 1;
			}
		}else if(field_complex){
			float im;
			for(i = 0; i < coo.nnz; i++){
				fgets(buffer, sizeof(buffer), fp);
				sscanf(buffer, "%d %d %f %f", &coo.triple[i].x, &coo.triple[i].y, &coo.triple[i].val, &im);
			}
		}else{
			for(i = 0; i < coo.nnz; i++){
				fgets(buffer, sizeof(buffer), fp);
				sscanf(buffer, "%d %d %f", &coo.triple[i].x, &coo.triple[i].y, &coo.triple[i].val);
			}
		}
	}
	fclose(fp);

	if(i > coo.nnz){
		printf("ERROR: *** too many entries occered ***\n");
		return ERROR;
	}
	printf("\nMatrix is now in coordinate format\n");

	printf("\nMatrix Information:\n");
	printf("Number of rows      : %d\n", coo.nrow);
	printf("Number of columns   : %d\n", coo.ncol);
	printf("Number of non-zeros : %d\n\n", coo.nnz);

print_coo(&coo);

	//COO -> CSR
	printf("Coverting to CSR format...\n");

	csr->ncol = coo.ncol;
	csr->nrow = coo.nrow;
	csr->nnz = coo.nnz;
	csr->val = (float *)malloc(csr->nnz * sizeof(float));
	csr->col_idx = (int *)malloc(csr->nnz * sizeof(int));
	csr->row_ptr = (int *)malloc((csr->nrow + 1) * sizeof(int));

	qsort(coo.triple, coo.nnz, sizeof(triple_t), func_cmp);

	csr->row_ptr[0] = 0;
	int r = 0;
	for(i = 0; i < csr->nnz; i++){
		while(coo.triple[i].x - COO_BASE != r){
			csr->row_ptr[++r] = i;
		}
		csr->val[i] = coo.triple[i].val;
		csr->col_idx[i] = coo.triple[i].y - COO_BASE;
	}
	while(r < csr->nrow){
		csr->row_ptr[++r] = i;
	}
	printf("OK!\n");

	free(coo.triple);

print_csr(csr);

	return OK;
}



int preprocess(cvr_t *cvr, csr_t *csr){
	printf("\nCoverting to CVR format...\n");

	cvr->ncol = csr->ncol;
	cvr->nrow = csr->nrow;
	cvr->nnz = csr->nnz;
	cvr->lrrec_ptr = (int *)malloc(n_threads * sizeof(int));
	cvr->val_ptr = (float **)malloc(n_threads * sizeof(float *));
	cvr->colidx_ptr = (int **)malloc(n_threads * sizeof(int *));
	cvr->rec_ptr = (int **)malloc(n_threads * sizeof(int *));
	cvr->tail_ptr = (int **)malloc(n_threads * sizeof(int *));

	int nnz_per_thread = cvr->nnz / n_threads;
	int change_thread_nnz = cvr->nnz % n_threads;

	#pragma omp parallel num_threads(n_threads)
	{
		int thread_num = omp_get_thread_num();

		int thread_start, thread_end, thread_nnz;
		int thread_start_row, thread_end_row, thread_nrow;

		//thread information
		//thread whose thread_num is less than change_thread_nnz handle one more non-zero number than the others
		//both No.thread_start non-zero and No.thread_end non-zero are handled by this thread
		if(thread_num < change_thread_nnz){
			thread_start = thread_num * nnz_per_thread + thread_num * 1;
			thread_end = (thread_num + 1) * nnz_per_thread + (thread_num + 1) * 1 - 1;
		}else{
			thread_start = thread_num * nnz_per_thread + change_thread_nnz * 1;
			thread_end = (thread_num + 1) * nnz_per_thread + change_thread_nnz * 1 - 1;
		}
		thread_nnz = thread_end - thread_start + 1;
		thread_start_row = func_get_row(thread_start, csr);
		thread_end_row = func_get_row(thread_end, csr);
		thread_nrow = thread_end_row - thread_start_row + 1;
		
		cvr->val_ptr[thread_num] = (float *)malloc(thread_nnz * sizeof(float));
		cvr->colidx_ptr[thread_num] = (int *)malloc(thread_nnz * sizeof(int));
		cvr->rec_ptr[thread_num] = (int *)malloc(thread_nrow * 2 * sizeof(int));
		cvr->tail_ptr[thread_num] = (int *)malloc(n_lanes * sizeof(int));

		int *thread_valID = (int *)malloc(n_lanes * sizeof(int));
		int *thread_rowID = (int *)malloc(n_lanes * sizeof(int));
		int *thread_count = (int *)malloc(n_lanes * sizeof(int));

		//initialize 
		int thread_rs = thread_start_row;
		int i, j, k; //iteration variables
		for(i = 0; i < n_lanes; i++){
			while(thread_rs <= thread_end_row && csr->row_ptr[thread_rs+1] == csr->row_ptr[thread_rs]){
				thread_rs++;
			}
			thread_rowID[i] = thread_rs;
			thread_valID[i] = csr->row_ptr[thread_rs];
			thread_count[i] = csr->row_ptr[thread_rs+1] - csr->row_ptr[thread_rs];
			if(thread_rs == thread_start_row){
				thread_valID[i] = thread_start;
				thread_count[i] = csr->row_ptr[thread_rs+1] - thread_start;
			}
			if(thread_rs == thread_end_row){
				thread_count[i] = thread_end + 1 - thread_valID[i];
			}
			if(thread_rs > thread_end_row){
				thread_rowID[i] = -1;
				thread_valID[i] = -1;
				thread_count[i] = 0;
			}
			thread_valID[i] = thread_valID[i] - thread_start;
			thread_rs++;
		}
		//because the behavior of func_get_row(), 
		//thread_start_row and thread_end_row contain at least one non-zero
		//IF1: if the number of rows is less than n_lanes
		if(thread_rs > thread_end_row){
			for(i = 0; i < n_lanes; i++){
				cvr->tail_ptr[thread_num][i] = thread_rowID[i];
			}
		}

		int rec_idx = 0;
		cvr->lrrec_ptr[thread_num] = -1;
		//WARNING: condition of this for loop is uncertain
		//well, it is certain now. but i'm not that sure
		// FOR1
		for(i = 0; i <= (thread_nnz + n_lanes - 1) / n_lanes; i++){
			// IF2: if some lanes are empty
			if(0 == func_AND(thread_count, n_lanes)){
				// FOR2: iterate over all lanes, feed or steal
				for(j = 0; j < n_lanes; j++){
					// IF3
					if(0 == thread_count[j]){
						thread_rs++;
						//EQUAL is included because thread_end_row can be reached
						while(thread_rs <= thread_end_row && csr->row_ptr[thread_rs+1] == csr->row_ptr[thread_rs]){
							thread_rs++;
						}

						//recording
						cvr->rec_ptr[thread_num][rec_idx++] = i * n_lanes + j; //valID which triggers write back
						cvr->rec_ptr[thread_num][rec_idx++] = thread_rowID[j]; //write back position

						//tracker feeding
						if(thread_rs <= thread_end_row){
							thread_rowID[j] = thread_rs;
							thread_valID[j] = csr->row_ptr[thread_rs];
							thread_count[j] = csr->row_ptr[thread_rs+1] - csr->row_ptr[thread_rs];
							//ELSE1: if the number of rows is more than n_lanes, 
							//the lane deals thread_end_row must reaches here
							if(thread_rs == thread_end_row){
								thread_count[j] = thread_end + 1 - thread_valID[j];
								for(k = 0; k < n_lanes; k++){
									//WARNING: ASK MR XIE ABOUT THIS
									thread_rowID[k] = k;
									//YES, EXACTLY THE STATEMENT ABOVE
									cvr->tail_ptr[thread_num][k] = thread_rowID[k];
								}
							}
						}else{//tracker stealing, thread_rs is not important since then
							if(-1 == cvr->lrrec_ptr[thread_num]){
								cvr->lrrec_ptr[thread_num] = i * n_lanes + j;
							}

							int average = func_average(thread_count, n_lanes);
							//if reminding number is less than n_lanes
							if(0 == average){
								thread_valID[j] = -1;
								thread_count[j] = 1;
							}else{
								int candidate = 0;
								//get candidate to steal
								for( ; candidate < n_lanes; candidate++){
									if(thread_count[candidate] > average){
										break;
									}
								}
								thread_rowID[j] = candidate;
								thread_valID[j] = thread_valID[candidate];
								thread_count[j] = thread_count[candidate] - average;
								thread_rowID[candidate] = candidate;
								thread_valID[candidate] = thread_valID[candidate] + thread_count[j];
								thread_count[candidate] = thread_count[candidate] - thread_count[j];
							}
						}
					} //ENDIF3
				} //ENDFOR2
			} //ENDIF2

			//continue converting
			int gather_base = 0;
			for(j = 0; j < n_lanes; j++){
				//if thread_nnz is not a multiple of n_lanes, this happens at the last round
				if(-1 == thread_valID[j]){
					cvr->val_ptr[thread_num][j+gather_base] = 0;
					cvr->colidx_ptr[thread_num][j+gather_base] = cvr->colidx_ptr[thread_num][j+gather_base-1];
				}else{
					cvr->val_ptr[thread_num][j+gather_base] = csr->val[thread_valID[j]];
					cvr->colidx_ptr[thread_num][j+gather_base] = csr->col_idx[thread_valID[j]];
				}
				thread_valID[j]++;
				thread_count[j]--;
			}
			gather_base += n_lanes;
		} //ENDFOR1
	} //ENDPRAGMA

	printf("OK!\n");

	return OK;
}



int spmv(float *y, float *x, cvr_t *cvr, csr_t *csr){

	int nnz_per_thread = cvr->nnz / n_threads;
	int change_thread_nnz = cvr->nnz % n_threads;

	int iteration;
	//FOR1
	for(iteration = 0; iteration < n_iterations; iteration++){
		#pragma omp parallel num_threads(n_threads)
		{
			int thread_num = omp_get_thread_num();

			//thread information
			//exactly the same code as in preprocess()
			int thread_start, thread_end, thread_nnz;
			int thread_start_row, thread_end_row, thread_nrow;
			if(thread_num < change_thread_nnz){
				thread_start = thread_num * nnz_per_thread + thread_num * 1;
				thread_end = (thread_num + 1) * nnz_per_thread + (thread_num + 1) * 1 - 1;
			}else{
				thread_start = thread_num * nnz_per_thread + change_thread_nnz * 1;
				thread_end = (thread_num + 1) * nnz_per_thread + change_thread_nnz * 1 - 1;
			}
			thread_nnz = thread_end - thread_start + 1;
			thread_start_row = func_get_row(thread_start, csr);
			thread_end_row = func_get_row(thread_end, csr);
			thread_nrow = thread_end_row - thread_start_row + 1;

			//store the temporary result of this thread
			float *thread_y = (float *)malloc(cvr->ncol * sizeof(float));
			memset(thread_y, 0, cvr->ncol * sizeof(float));

			//store the intermediate result
			float *thread_temp = (float *)malloc(n_lanes * sizeof(float));
			memset(thread_temp, 0, n_lanes * sizeof(float));

			int rec_idx = 0;
			int offset, writeback;
			int i, j;
			//FOR2
			for(i = 0; i < thread_nnz; i += n_lanes){
				for(j = 0; j < n_lanes; j++){
					int x_offset = cvr->colidx_ptr[thread_num][i*n_lanes+j];
					thread_temp[j] += cvr->val_ptr[thread_num][i*n_lanes+j] * x[x_offset];
				}
				//corresponding to tracker feeding part
				if(cvr->rec_ptr[thread_num][rec_idx] < cvr->lrrec_ptr[thread_num]){
					//more than one temporary result could be processed here
					while(cvr->rec_ptr[thread_num][rec_idx] / n_lanes == i / n_lanes){
						offset = cvr->rec_ptr[thread_num][rec_idx++] % n_lanes;
						writeback = cvr->rec_ptr[thread_num][rec_idx++];
						thread_y[writeback] = thread_temp[offset];
						thread_temp[offset] = 0;
					}
				}else{//corresponding to tracker stealing part
					while(cvr->rec_ptr[thread_num][rec_idx] / n_lanes == i / n_lanes){
						offset = cvr->rec_ptr[thread_num][rec_idx++] % n_lanes;
						writeback = cvr->rec_ptr[thread_num][rec_idx++];
						thread_temp[writeback] += thread_temp[offset];
					}
				}
			} //ENDFOR2
			for(i = 0; i < n_lanes; i++){
				thread_y[cvr->tail_ptr[thread_num][i]] = thread_temp[i];
			}

			for(i = 0; i < cvr->ncol; i++){
				#pragma omp atomic
				y[i] += thread_y[i];
			}
		} //ENDPRAGMA
	} //ENDFOR1: iteration

	return OK;
}
