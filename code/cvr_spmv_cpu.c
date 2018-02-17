#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<omp.h>

#include<sys/time.h>

#define OK 0
#define ERROR 1

#define FIELD_LENGTH 128
#define COO_BASE 0

//#define MATRIX_DEBUG
#ifdef MATRIX_DEBUG
#include"matrix_debug.h"
#else
typedef struct triple{
	int x;
	int y;
	double val;
}triple_t;

typedef struct coo{
	triple_t *triple;
	int ncol;
	int nrow;
	int nnz;
}coo_t; // coordinate format

typedef struct csr{
	double *val;
	int *col_idx;
	int *row_ptr;
	int ncol;
	int nrow;
	int nnz;
}csr_t; // compressed sparse row format

typedef struct cvr{
	double **val_ptr;
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
		}else if(mid < csr->nrow && csr->row_ptr[mid+1] <= valID){
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
	return (sum + n - 1) / n;
}

// auxiliary function to compare result
int func_compare(double y, double y_verify){
	if(y - y_verify < -0.0001 || y - y_verify > 0.0001){
		return 1;
	}else{
		return 0;
	}
}


// 0-based Matrix Market format -> CSR format
int read_matrix(csr_t *csr, char *filename);
// CSR format -> CVR format
int preprocess(cvr_t *cvr, csr_t *csr);
// CVR format SpMV, y = y + M * x, parameter csr is only used in func_get_row
int spmv(double *y, double *x, cvr_t *cvr, csr_t *csr);

int n_threads = 2;
int n_lanes = 4;
int n_iterations = 1;

int main(int argc, char **argv){
	csr_t csr;
	cvr_t cvr;
	double *y, *x;

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
	
	printf("Input file: %s\n", filename);
	printf("Number of threads: %d\n", n_threads);
	printf("Number of iterations: %d\n\n", n_iterations);

	if(read_matrix(&csr, filename)){
		printf("ERROR occured in function read_matrix()\n");
		return ERROR;
	}

	
	x = (double *)malloc(csr.ncol * sizeof(double));
	if(NULL == x){
		printf("ERROR: *** memory overflow in main(), unsufficient memory for x ***\n");
		return ERROR;
	}
	y = (double *)malloc(csr.nrow * sizeof(double));
	if(NULL == y){
		printf("ERROR: *** memory overflow in main(), unsufficient memory for y ***\n");
		return ERROR;
	}

	int i, j, iteration;
	for(i = 0; i < csr.nrow; i++){
		x[i] = i % 1000;
	}
	memset(y, 0, csr.nrow * sizeof(double));

	struct timeval tv1, tv2;
	struct timezone tz1, tz2;
	long tv_diff1, tv_diff2;

	gettimeofday(&tv1, &tz1);
	// PREPROCESS
	if(preprocess(&cvr, &csr)){
		printf("ERROR occured in function preprocess()\n");
		return ERROR;
	}

	// SPMV KERNEL
	if(spmv(y, x, &cvr, &csr)){
		printf("ERROR occured in function spmv()\n");
		return ERROR;
	}
	gettimeofday(&tv2, &tz2);
	tv_diff1 = (tv2.tv_sec - tv1.tv_sec) * 1000000 + tv2.tv_usec - tv1.tv_usec;
	printf("cvr time(usec): %ld\n", tv_diff1);

	double *y_verify = (double *)malloc(csr.nrow * sizeof(double));
	if(NULL == y_verify){
		printf("ERROR: *** memory overflow in main(), unsufficient memory for y_verify ***\n");
		return ERROR;
	}
	double sum;
	memset(y_verify, 0, csr.nrow * sizeof(double));

	gettimeofday(&tv1, &tz1);
	for(iteration = 0; iteration < n_iterations; iteration++){
		#pragma omp parallel for num_threads(n_threads) private(sum, j)
		for(i = 0; i < csr.nrow; i++){
			sum = 0;
			for(j = csr.row_ptr[i]; j < csr.row_ptr[i+1]; j++){
				sum += csr.val[j] * x[csr.col_idx[j]];
			}
			y_verify[i] += sum;
		}
	}
	gettimeofday(&tv2, &tz2);
	tv_diff2 = (tv2.tv_sec - tv1.tv_sec) * 1000000 + tv2.tv_usec - tv1.tv_usec;
	printf("csr time(usec): %ld\n", tv_diff2);


//DEBUG: Y_VERIFY
//print_vector(y_verify, csr.nrow);
//ENDDEBUG: Y_VERIFY

	int count = 0;
	for(i = 0; i < csr.nrow; i++){
		if(func_compare(y[i], y_verify[i])){
			count++;
			printf("y[%d] should be %lf, but the result is %lf\n", i, y_verify[i], y[i]);	
		}
		if(count > 10){
			return 0;
		}
	}
	if(0 == count){
		printf("Correct\n");
	}

	return 0;
}



int read_matrix(csr_t *csr, char *filename){
	FILE *fp = fopen(filename, "r");
	if(!fp){
		printf("ERROR: *** cannot open file: %s ***\n", filename);
		return ERROR;
	}
	
	printf("Reading matrix...\n");

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
	if(NULL == coo.triple){
		printf("ERROR: *** memory overflow in read_matrix(), unsufficient memory for coo ***\n");
		return ERROR;
	}

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
			double im;
			for(i = 0; i < coo.nnz; i++){
				fgets(buffer, sizeof(buffer), fp);
				sscanf(buffer, "%d %d %lf %lf", &coo.triple[i].x, &coo.triple[i].y, &coo.triple[i].val, &im);
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
				sscanf(buffer, "%d %d %lf", &coo.triple[i].x, &coo.triple[i].y, &coo.triple[i].val);
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
			double im;
			for(i = 0; i < coo.nnz; i++){
				fgets(buffer, sizeof(buffer), fp);
				sscanf(buffer, "%d %d %lf %lf", &coo.triple[i].x, &coo.triple[i].y, &coo.triple[i].val, &im);
			}
		}else{
			for(i = 0; i < coo.nnz; i++){
				fgets(buffer, sizeof(buffer), fp);
				sscanf(buffer, "%d %d %lf", &coo.triple[i].x, &coo.triple[i].y, &coo.triple[i].val);
			}
		}
	}
	fclose(fp);

	if(i > coo.nnz){
		printf("ERROR: *** too many entries occered ***\n");
		return ERROR;
	}
	printf("\nMatrix is in coordinate format now\n");

	printf("\nMatrix Information:\n");
	printf("Number of rows      : %d\n", coo.nrow);
	printf("Number of columns   : %d\n", coo.ncol);
	printf("Number of non-zeros : %d\n\n", coo.nnz);

//DEBUG: COORDINATE
//print_coo(&coo);
//ENDDEBUG: COORDINATE

	//COO -> CSR
	printf("Coverting to CSR format...\n");

	csr->ncol = coo.ncol;
	csr->nrow = coo.nrow;
	csr->nnz = coo.nnz;
	csr->val = (double *)malloc(csr->nnz * sizeof(double));
	if(NULL == csr->val){
		printf("ERROR: *** memory overflow in read_matrix(), unsufficient memory for csr->val ***\n");
		return ERROR;
	}
	csr->col_idx = (int *)malloc(csr->nnz * sizeof(int));
	if(NULL == csr->col_idx){
		printf("ERROR: *** memory overflow in read_matrix(), unsufficient memory for csr->col_idx ***\n");
		return ERROR;
	}
	csr->row_ptr = (int *)malloc((csr->nrow + 1) * sizeof(int));
	if(NULL == csr->row_ptr){
		printf("ERROR: *** memory overflow in read_matrix(), unsufficient memory for csr->row_ptr ***\n");
		return ERROR;
	}

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

//DEBUG: MATRIX
//print_csr(csr);
//print_matrix(csr);
//ENDDEBUG: MATRIX

	return OK;
}



int preprocess(cvr_t *cvr, csr_t *csr){
	printf("\nCoverting to CVR format...\n");

	cvr->ncol = csr->ncol;
	cvr->nrow = csr->nrow;
	cvr->nnz = csr->nnz;
	cvr->lrrec_ptr = (int *)malloc(n_threads * sizeof(int));
	if(NULL == cvr->lrrec_ptr){
		printf("ERROR: *** memory overflow in preprocess(), unsufficient memory for cvr->lrrec_ptr ***\n");
		return ERROR;
	}
	cvr->val_ptr = (double **)malloc(n_threads * sizeof(double *));
	if(NULL == cvr->val_ptr){
		printf("ERROR: *** memory overflow in preprocess(), unsufficient memory for cvr->val_ptr ***\n");
		return ERROR;
	}
	cvr->colidx_ptr = (int **)malloc(n_threads * sizeof(int *));
	if(NULL == cvr->colidx_ptr){
		printf("ERROR: *** memory overflow in preprocess(), unsufficient memory for cvr->colidx_ptr ***\n");
		return ERROR;
	}
	cvr->rec_ptr = (int **)malloc(n_threads * sizeof(int *));
	if(NULL == cvr->rec_ptr){
		printf("ERROR: *** memory overflow in preprocess(), unsufficient memory for cvr->rec_ptr ***\n");
		return ERROR;
	}
	cvr->tail_ptr = (int **)malloc(n_threads * sizeof(int *));
	if(NULL == cvr->tail_ptr){
		printf("ERROR: *** memory overflow in preprocess(), unsufficient memory for cvr->tail_ptr ***\n");
		return ERROR;
	}


	int nnz_per_thread = cvr->nnz / n_threads;
	int change_thread_nnz = cvr->nnz % n_threads;

	#pragma omp parallel num_threads(n_threads)
	{
		int thread_num = omp_get_thread_num();

//DEBUG: THREAD
//#pragma omp critical
//{
//printf("preprocess thread_%d start\n", thread_num);

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
		// IF0: if this thread has at least one number to deal with
		if(thread_nnz > 0){
			thread_start_row = func_get_row(thread_start, csr);
			thread_end_row = func_get_row(thread_end, csr);
			thread_nrow = thread_end_row - thread_start_row + 1;

			//padding is needed
			int thread_n_vals = (thread_nnz + n_lanes - 1) / n_lanes * n_lanes;
			// note: ADD n_lanes is necessary because "stealing" also needs to be recorded,
			//       and stealing time is less than n_lanes
			int thread_n_recs = (thread_nrow + n_lanes - 1) / n_lanes * n_lanes + n_lanes;
		
			cvr->val_ptr[thread_num] = (double *)malloc(thread_n_vals * sizeof(double));
			if(NULL == cvr->val_ptr[thread_num]){
				printf("ERROR: *** memory overflow in preprocess() in thread_%d, unsufficient memory for cvr->val_ptr[%d]\n", thread_num, thread_num);
				exit(ERROR);
			}
			cvr->colidx_ptr[thread_num] = (int *)malloc(thread_n_vals * sizeof(int));
			if(NULL == cvr->colidx_ptr[thread_num]){
				printf("ERROR: *** memory overflow in preprocess() in thread_%d, unsufficient memory for cvr->colidx_ptr[%d]\n", thread_num, thread_num);
				exit(ERROR);
			}
			cvr->rec_ptr[thread_num] = (int *)malloc(thread_n_recs * 2 * sizeof(int));
			if(NULL == cvr->rec_ptr[thread_num]){
				printf("ERROR: *** memory overflow in preprocess() in thread_%d, unsufficient memory for cvr->rec_ptr[%d]\n", thread_num, thread_num);
				exit(ERROR);
			}
			cvr->tail_ptr[thread_num] = (int *)malloc(n_lanes * sizeof(int));
			if(NULL == cvr->tail_ptr[thread_num]){
				printf("ERROR: *** memory overflow in preprocess() in thread_%d, unsufficient memory for cvr->tail_ptr[%d]\n", thread_num, thread_num);
				exit(ERROR);
			}


			int *thread_valID = (int *)malloc(n_lanes * sizeof(int));
			if(NULL == thread_valID){
				printf("ERROR: *** memory overflow in preprocess() in thread_%d, unsufficient memory for thread_valID\n", thread_num);
				exit(ERROR);
			}
			int *thread_rowID = (int *)malloc(n_lanes * sizeof(int));
			if(NULL ==thread_rowID){
				printf("ERROR: *** memory overflow in preprocess() in thread_%d, unsufficient memory for thread_rowID\n", thread_num);
				exit(ERROR);
			}
			int *thread_count = (int *)malloc(n_lanes * sizeof(int));
			if(NULL ==thread_count){
				printf("ERROR: *** memory overflow in preprocess() in thread_%d, unsufficient memory for thread_count\n", thread_num);
				exit(ERROR);
			}


			//initialize 
			int thread_rs = thread_start_row;
			int i, j, k; //iteration variables
			for(i = 0; i < n_lanes; i++){
				while(thread_rs <= thread_end_row && csr->row_ptr[thread_rs+1] == csr->row_ptr[thread_rs]){
					thread_rs++;
				}

				if(thread_rs > thread_end_row){
					thread_rowID[i] = -1;
					thread_valID[i] = -1;
					thread_count[i] = 0;
				}else{
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
			
					thread_valID[i] = thread_valID[i] - thread_start;
					thread_rs++;
				}
			}
			//because the behavior of func_get_row(), 
			//thread_start_row(or thread_end_row) contains at least one non-zero
			//IF1: if the number of rows is less than n_lanes, initialize tail_ptr
			if(thread_rs > thread_end_row){
				for(i = 0; i < n_lanes; i++){
					cvr->tail_ptr[thread_num][i] = thread_rowID[i];
					if(thread_rowID[i] != -1){
						thread_rowID[i] = i;
					}
				}
			}

			int rec_idx = 0, gather_base = 0;
			cvr->lrrec_ptr[thread_num] = -1;
			// FOR1
			for(i = 0; i <= thread_n_vals / n_lanes; i++){
//DEBUG: ITERATOR
//printf("iteration %d/%d\n", i, (thread_nnz + n_lanes - 1) / n_lanes);
//ENDDEBUG: ITERATOR
				// IF2: if some lanes are empty
				if(0 == func_AND(thread_count, n_lanes)){
					// FOR2: iterate over all lanes, feed or steal
					for(j = 0; j < n_lanes; j++){
						// IF3: lane[j] is empty
						if(0 == thread_count[j]){
							//EQUAL is included because thread_end_row can be reached
							while(thread_rs <= thread_end_row && csr->row_ptr[thread_rs+1] == csr->row_ptr[thread_rs]){
								thread_rs++;
							}

							//recording
							if(-1 != thread_rowID[j]){
								cvr->rec_ptr[thread_num][rec_idx++] = (i - 1) * n_lanes + j; //valID which triggers write back
								cvr->rec_ptr[thread_num][rec_idx++] = thread_rowID[j]; //write back position
							}

							//IF4: tracker feeding
							//if next non-empty line exsits, non-empty is guaranteed by previous while loop
							if(thread_rs <= thread_end_row){
								thread_rowID[j] = thread_rs;
								thread_valID[j] = csr->row_ptr[thread_rs] - thread_start;
								thread_count[j] = csr->row_ptr[thread_rs+1] - csr->row_ptr[thread_rs];
								//ELSE1: if the number of rows is more than n_lanes, initialize tail_ptr
								//the lane deals thread_end_row must reaches here
								if(thread_rs == thread_end_row){
									thread_count[j] = thread_end + 1 - thread_valID[j] - thread_start;
									for(k = 0; k < n_lanes; k++){
										cvr->tail_ptr[thread_num][k] = thread_rowID[k];
										thread_rowID[k] = k;
									}
								}
								thread_rs++;

							}else{ //ELSE4: tracker stealing, thread_rs is not important since then
								if(-1 == cvr->lrrec_ptr[thread_num]){
									int temp_lrrec = (i - 1) * n_lanes + j;
									cvr->lrrec_ptr[thread_num] = temp_lrrec < 0 ? 0 : temp_lrrec;
								}

								int average = func_average(thread_count, n_lanes);
								//if no remainding numbers exist
								if(0 == average){
									if(i != thread_n_vals / n_lanes){
										printf("ERROR: *** last round of preprocessing is incorrect ***\n");
										exit(ERROR);
									}
									continue;
								}else{
									int candidate = 0;
									//get candidate to steal
									for( ; candidate < n_lanes; candidate++){
										if(thread_count[candidate] > average){
											break;
										}
									}
									//if padding is needed
									if(candidate == n_lanes){
										thread_valID[j] = -1;
										thread_count[j] = 1;
									}else{
										thread_rowID[j] = candidate;
										thread_valID[j] = thread_valID[candidate];
										thread_count[j] = average;
										thread_rowID[candidate] = candidate;
										thread_valID[candidate] = thread_valID[candidate] + average;
										thread_count[candidate] = thread_count[candidate] - average;
									}
								}
							} //ENDIF4
						} //ENDIF3
					} //ENDFOR2
				} //ENDIF2

				//continue converting
				for(j = 0; j < n_lanes && i < thread_n_vals / n_lanes; j++){
					//if padding exists
					if(-1 == thread_valID[j]){
						cvr->val_ptr[thread_num][j+gather_base] = 0;
						cvr->colidx_ptr[thread_num][j+gather_base] = cvr->colidx_ptr[thread_num][j+gather_base-1];
					}else{
						cvr->val_ptr[thread_num][j+gather_base] = csr->val[thread_valID[j]+thread_start];
						cvr->colidx_ptr[thread_num][j+gather_base] = csr->col_idx[thread_valID[j]+thread_start];
					}
					thread_valID[j]++;
					thread_count[j]--;
				}
				gather_base += n_lanes;
			} //ENDFOR1
//DEBUG: CVR DETAIL
//#pragma omp critical
//print_cvr_detail(cvr, thread_num, thread_nnz, thread_nrow, n_lanes);
//ENDDEBUG: CVR DETAIL
			free(thread_rowID);
			free(thread_valID);
			free(thread_count);
		} //ENDIF0

//printf("thread_%d complete\n", thread_num);
//}
//ENDDEBUG: THREAD

	} //ENDPRAGMA

	printf("OK!\n\n");

	return OK;
}



int spmv(double *y, double *x, cvr_t *cvr, csr_t *csr){

	int nnz_per_thread = cvr->nnz / n_threads;
	int change_thread_nnz = cvr->nnz % n_threads;

	int iteration;
	//FOR1
	for(iteration = 0; iteration < n_iterations; iteration++){
		#pragma omp parallel num_threads(n_threads)
		{
			int thread_num = omp_get_thread_num();

//DEBUG: THREAD
//#pragma omp critical
//{
//printf("spmv thread_%d start\n", thread_num);

			//thread information
			int thread_start, thread_end, thread_nnz;
			int thread_start_row, thread_end_row;
			if(thread_num < change_thread_nnz){
				thread_start = thread_num * nnz_per_thread + thread_num * 1;
				thread_end = (thread_num + 1) * nnz_per_thread + (thread_num + 1) * 1 - 1;
			}else{
				thread_start = thread_num * nnz_per_thread + change_thread_nnz * 1;
				thread_end = (thread_num + 1) * nnz_per_thread + change_thread_nnz * 1 - 1;
			}
			thread_nnz = thread_end - thread_start + 1;
			// IF0
			if(thread_nnz > 0){
				thread_start_row = func_get_row(thread_start, csr);
				thread_end_row = func_get_row(thread_end, csr);

				//store the temporary result of this thread
				double *thread_y = (double *)malloc(cvr->nrow * sizeof(double));
				if(NULL == thread_y){
					printf("ERROR: *** memory overflow in spmv(), thread_%d ***\n", thread_num);
					exit(ERROR);
				}
				memset(thread_y, 0, cvr->nrow * sizeof(double));

				//store the intermediate result
				double *thread_temp = (double *)malloc(n_lanes * sizeof(double));
				if(NULL == thread_temp){
					printf("ERROR: *** memory overflow in spmv(), thread_%d ***\n", thread_num);
					exit(ERROR);
				}
				memset(thread_temp, 0, n_lanes * sizeof(double));

				int rec_idx = 0;
				int offset, writeback;
				int i, j;
				//FOR2
				for(i = 0; i < (thread_nnz + n_lanes - 1) / n_lanes; i++){
//DEBUG: ITERATOR
//printf("iteration %d/%d\n", i, (thread_nnz + n_lanes - 1) / n_lanes);
//ENDDEBUG: ITERATOR
					for(j = 0; j < n_lanes; j++){
						int x_offset = cvr->colidx_ptr[thread_num][i*n_lanes+j];
						thread_temp[j] += cvr->val_ptr[thread_num][i*n_lanes+j] * x[x_offset];
					}
					//corresponding to tracker feeding part
					if(cvr->rec_ptr[thread_num][rec_idx] < cvr->lrrec_ptr[thread_num]){
						//more than one temporary result could be processed here
						while(cvr->rec_ptr[thread_num][rec_idx] / n_lanes == i){
							if(cvr->rec_ptr[thread_num][rec_idx] < cvr->lrrec_ptr[thread_num]){
								offset = cvr->rec_ptr[thread_num][rec_idx++] % n_lanes;
								writeback = cvr->rec_ptr[thread_num][rec_idx++];
								thread_y[writeback] = thread_temp[offset];
								thread_temp[offset] = 0;
							}else{ // in case rec[rec_idx] < lrrec < rec[rec_idx+2]
								offset = cvr->rec_ptr[thread_num][rec_idx++] % n_lanes;
								writeback = cvr->rec_ptr[thread_num][rec_idx++];
								if(-1 != cvr->tail_ptr[thread_num][writeback]){
									thread_y[cvr->tail_ptr[thread_num][writeback]] += thread_temp[offset];
								}
								thread_temp[offset] = 0;
							}
						}
					}else{//corresponding to tracker stealing part
						while(cvr->rec_ptr[thread_num][rec_idx] / n_lanes == i){
							offset = cvr->rec_ptr[thread_num][rec_idx++] % n_lanes;
							writeback = cvr->rec_ptr[thread_num][rec_idx++];
							if(-1 != cvr->tail_ptr[thread_num][writeback]){
								thread_y[cvr->tail_ptr[thread_num][writeback]] += thread_temp[offset];
							}
							thread_temp[offset] = 0;
						}
					}
				} //ENDFOR2

				#pragma omp atomic
				y[thread_start_row] += thread_y[thread_start_row];
				if(thread_start_row != thread_end_row){
					#pragma omp atomic
					y[thread_end_row] += thread_y[thread_end_row];
				}
				
				for(i = thread_start_row + 1; i < thread_end_row; i++){
					y[i] += thread_y[i];
				}
				free(thread_y);
				free(thread_temp);
			} //ENDIF0
//printf("thread_%d complete\n", thread_num);
//}
//ENDDEBUG: THREAD


		} //ENDPRAGMA
	} //ENDFOR1: iteration

	return OK;
}
