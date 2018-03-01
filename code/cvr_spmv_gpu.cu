// GPU version

#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<omp.h>

#include<sys/time.h>

#include<cuda_runtime.h>

#define OK 0
#define ERROR 1

#define CMP_EQUAL 0

#define FIELD_LENGTH 128
#define COO_BASE 0

#define THREADS_PER_WARP 32

#define floatType double


#define CHECK(call){\
	const cudaError_t error = call;\
	if(error != cudaSuccess){\
		printf("Error: %s:%d\n", __FILE__, __LINE__);\
		printf("code: %d, reason: %s\n", error, cudaGetErrorString(error));\
		exit(ERROR);\
	}\
}

#define HOST_CHECK(ptr){\
	if(ptr == NULL){\
		printf("Error: %s:%d\n", __FILE__, __LINE__);\
		printf("Memory overflow!\n");\
		exit(ERROR);\
	}\
}

typedef struct triple{
	int x;
	int y;
	floatType val;
}triple_t;

typedef struct coo{
	triple_t *triple;
	int ncol;
	int nrow;
	int nnz;
}coo_t; // coordinate format

typedef struct csr{
	int ncol;
	int nrow;
	int nnz;
	floatType *val;
	int *col_idx;
	int *row_ptr;
}csr_t; // compressed sparse row format

typedef struct record{
	int pos;
	int wb;
}record_t;

typedef struct cvr{
	int ncol;
	int nrow;
	int nnz;
	floatType *val_ptr;
	int *colidx_ptr;
	record_t *rec_ptr;
	int *lrrec_ptr;
	int *tail_ptr;
}cvr_t; // compressed vactorization-oriented sparse row format


// auxiliary function used in qsort
inline int func_cmp(const void *a, const void *b){
	triple_t *t1 = (triple_t *)a;
	triple_t *t2 = (triple_t *)b;
	if(t1->x != t2->x){
		return t1->x - t2->x;
	}else{
		return t1->y - t2->y;
	}
}

// auxiliary function to get row number
__device__ inline int func_get_row(int valID, csr_t *csr){
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
inline int func_AND(int *val, int n){
	int result = 1, i;
	for(i = 0; i < n; i++){
		result = result && val[i];
	}
	return result;
}

// auxiliary function to get average count
inline int func_average(int *count, int n){
	int sum = 0, i;
	for(i = 0; i < n; i++){
		sum += count[i];
	}
	return (sum + n - 1) / n;
}

// auxiliary function to compare result
inline int func_compare(floatType y, floatType y_verify){
	if(y - y_verify < -0.0001 || y - y_verify > 0.0001){
		return 1;
	}else{
		return 0;
	}
}

inline void func_initialData(floatType *ip, int size){
	time_t t;
	srand((unsigned)time(&t));
	for(int i = 0; i < size; i++)	{
		ip[i] = (floatType)(rand() & 0xff) / 10.0f;
	}
}

// 0-based Matrix Market format -> CSR format
int read_matrix(csr_t *csr, char *filename);
// CSR format -> CVR format
int preprocess(cvr_t *d_cvr, csr_t *d_csr, csr_t *h_csr, int *rowID, int *valID, int *count);
// CVR format SpMV, y = y + M * x
int spmv(floatType *d_y, floatType *d_x, cvr_t *d_cvr, csr_t *d_csr, csr_t *h_csr);

__global__ void preprocess_kernel(cvr_t *cvr, csr_t *csr, int const nnz_per_block, int const change_block_nnz, \
	int *rowID, int *valID, int *count);
__global__ void spmv_kernel();

// however, in this implementation, only one dimension is used for intuition
int gridDim[3] = {1, 1, 1};
int blockDim[3] = {1, 1, 1};
int n_iterations = 1;

int main(int argc, char **argv){

/****  runtime configuration  ****/

	if(argc < 2){
		printf("ERROR: *** wrong parameter format ***\n");
		return ERROR;
	}
	char *filename = argv[1];

//	if(argc > 2){
//		gridDim[0] = atoi(argv[2]);
//		blockDim[0] = atoi(argv[3]);
//		if(5 == argc){
//			n_iterations = atoi(argv[4]);
//		}
//	}

/****  \runtime configuration  ****/


/****  basic runtime information  ****/

	printf("Input file: %s\n", filename);
	printf("Grid dimension: (%d, %d, %d)\n", gridDim[0], gridDim[1], gridDim[2]);
	printf("Block dimension: (%d, %d, %d)\n", blockDim[0], blockDim[1], blockDim[2]);
	printf("Number of iterations: %d\n\n", n_iterations);

/****  \basic runtime information  ****/


/****  prepare host_csr  ****/

	//allocate memory
	csr_t *h_csr = (csr_t *)malloc(sizeof(csr_t));
	HOST_CHECK(h_csr);

	//read matrix to initialize
	if(read_matrix(h_csr, filename)){
		printf("ERROR occured in function read_matrix()\n");
		return ERROR;
	}

/****  \prepare host_csr  ****/


/****  prepare and device_csr  ****/

	csr_t *d_csr = NULL;
	//allocate device global memory
	CHECK(cudaMalloc(&d_csr, sizeof(csr_t)));
	CHECK(cudaMalloc(&d_csr->val, h_csr->nnz * sizeof(floatType)));
	CHECK(cudaMalloc(&d_csr->col_idx, h_csr->nnz * sizeof(int)));
	CHECK(cudaMalloc(&d_csr->row_ptr, (h_csr->nrow + 1) * sizeof(int)));

	//initialize
	CHECK(cudaMemcpy(d_csr, h_csr, 3 * sizeof(int), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_csr->val, h_csr->val, h_csr->nnz * sizeof(floatType), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_csr->col_idx, h_csr->col_idx, h_csr->nnz * sizeof(int), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(d_csr->row_ptr, h_csr->row_ptr, (h_csr->nrow + 1) * sizeof(int), cudaMemcpyHostToDevice));

/****  \prepare device_csr  ****/


/****  prepare device_cvr  ****/

	cvr_t *d_cvr = NULL;
	//cvr structure is dependent on matrix and runtime configuration
	int n_blocks = gridDim[0] * gridDim[1] * gridDim[2];
	int n_block_threads = blockDim[0] * blockDim[1] * blockDim[2];
	int n_block_nnz = h_csr->nnz / n_blocks;
	int change_block_nnz = h_csr->nnz % n_blocks;
	int n_block_vals = (n_block_nnz + 1 + n_block_threads - 1) / n_block_threads * n_block_threads;
	int n_block_recs = n_block_vals + n_block_threads;

	//allocate device global memory
	CHECK(cudaMalloc(&d_cvr, sizeof(cvr_t)));
	CHECK(cudaMalloc(&d_cvr->val_ptr, n_blocks * n_block_vals * sizeof(floatType)));
	CHECK(cudaMalloc(&d_cvr->colidx_ptr, n_blocks * n_block_vals * sizeof(int)));
	CHECK(cudaMalloc(&d_cvr->rec_ptr, n_blocks * n_block_recs * sizeof(record_t)));
	CHECK(cudaMalloc(&d_cvr->lrrec_ptr, n_blocks * sizeof(int)));
	CHECK(cudaMalloc(&d_cvr->tail_ptr, n_blocks * n_block_threads * sizeof(int)));

	//initialize
	CHECK(cudaMemcpy(d_cvr, h_csr, 3 * sizeof(int), cudaMemcpyHostToDevice));

/****  \prepare device_cvr  ****/


/****  prepare host_x, device_x, host_y, device_y and verify_y  ****/

	//allocate memory
	floatType *h_x = (floatType *)malloc(h_csr->ncol * sizeof(floatType));
	HOST_CHECK(h_x);
	floatType *d_x = NULL;
	CHECK(cudaMalloc(&d_x, h_csr->ncol * sizeof(floatType)));
	floatType *h_y = (floatType *)malloc(h_csr->nrow * sizeof(floatType));
	HOST_CHECK(h_y);
	floatType *d_y = NULL;
	CHECK(cudaMalloc(&d_y, h_csr->nrow * sizeof(floatType)));
	floatType *y_verify = (floatType *)malloc(h_csr->nrow * sizeof(floatType));
	HOST_CHECK(y_verify);

	//initialize
	func_initialData(h_x, h_csr->ncol);
	CHECK(cudaMemcpy(d_x, h_x, h_csr->ncol * sizeof(floatType)));
	memset(h_y, 0, h_csr->nrow * sizeof(floatType));
	CHECK(cudaMemset(d_y, 0, h_csr->nrow * sizeof(floatType)));
	memset(y_verify, 0, h_csr->nrow * sizeof(floatType));

/****  \prepare host_x, device_x, host_y, device_y and verify_y  ****/


/****  launch kernels  ****/

	struct timeval tv1, tv2;
	long tv_diff1, tv_diff2;
	gettimeofday(&tv1, NULL);
	// trackers for preprocessing
	int *thread_rowID = NULL, *thread_valID = NULL, *thread_count = NULL;
	CHECK(cudaMalloc(&thread_rowID, n_block_threads * sizeof(int)));
	CHECK(cudaMalloc(&thread_valID, n_block_threads * sizeof(int)));
	CHECK(cudaMalloc(&thread_count, n_block_threads * sizeof(int)));
	// PREPROCESS
	if(preprocess(d_cvr, d_csr, h_csr, thread_rowID, thread_valID, thread_count)){
		printf("ERROR occured in function preprocess()\n");
		return ERROR;
	}
	

	// SPMV KERNEL
	if(spmv(d_y, d_x, d_cvr, d_csr, h_csr)){
		printf("ERROR occured in function spmv()\n");
		return ERROR;
	}
	gettimeofday(&tv2, NULL);
	tv_diff1 = (tv2.tv_sec - tv1.tv_sec) * 1000000 + tv2.tv_usec - tv1.tv_usec;
	printf("cvr time(usec): %ld\n", tv_diff1);

/****  \launch kernels  ****/


/****  copy back  ****/

	CHECK(cudaMemcpy(h_y, d_y, h_csr->nrow * sizeof(floatType), cudaMemcpyDeviceToHost));

/****  \copy back  ****/


/****  free device memory  ****/

	CHECK(cudaFree(thread_rowID));
	CHECK(cudaFree(thread_valID));
	CHECK(cudaFree(thread_count));

	CHECK(cudaFree(d_x));
	CHECK(cudaFree(d_y));

	CHECK(cudaFree(&d_cvr->val_ptr));
	CHECK(cudaFree(&d_cvr->colidx_ptr));
	CHECK(cudaFree(&d_cvr->rec_ptr));
	CHECK(cudaFree(&d_cvr->lrrec_ptr));
	CHECK(cudaFree(&d_cvr->tail_ptr));	
	CHECK(cudaFree(&d_cvr));

	CHECK(cudaFree(&d_csr->val));
	CHECK(cudaFree(&d_csr->col_idx));
	CHECK(cudaFree(&d_csr->row_ptr));
	CHECK(cudaFree(&d_csr));

/****  \free device memory  ****/


/****  compute y_verify using csr spmv  ****/

	gettimeofday(&tv1, NULL);
	for(int iteration = 0; iteration < n_iterations; iteration++){
		#pragma omp parallel for num_threads(omp_get_num_threads())
		floatType sum;
		for(int i = 0; i < h_csr->nrow; i++){
			sum = 0;
			for(int j = h_csr->row_ptr[i]; j < h_csr->row_ptr[i+1]; j++){
				sum += h_csr->val[j] * x[h_csr->col_idx[j]];
			}
			y_verify[i] += sum;
		}
	}
	gettimeofday(&tv2, NULL);
	tv_diff2 = (tv2.tv_sec - tv1.tv_sec) * 1000000 + tv2.tv_usec - tv1.tv_usec;
	printf("csr time(usec): %ld\n", tv_diff2);

/****  \compute y_verify using csr spmv  ****/


/****  check the result  ****/
	
	int count = 0;
	for(int i = 0; i < h_csr->nrow; i++){
		if(func_compare(h_y[i], y_verify[i]) != CMP_EQUAL){
			count++;
			printf("y[%d] should be %lf, but the result is %lf\n", i, y_verify[i], h_y[i]);	
		}
		if(count > 10){
			return 0;
		}
	}
	if(0 == count){
		printf("Correct\n");
	}

/****  \check the result  ****/

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
	HOST_CHECK(coo.triple);

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
			floatType im;
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
			floatType im;
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

	//COO -> CSR
	printf("Coverting to CSR format...\n");

	csr->ncol = coo.ncol;
	csr->nrow = coo.nrow;
	csr->nnz = coo.nnz;
	csr->val = (floatType *)malloc(csr->nnz * sizeof(floatType));
	HOST_CHECK(csr->val);
	csr->col_idx = (int *)malloc(csr->nnz * sizeof(int));
	HOST_CHECK(csr->col_idx);
	csr->row_ptr = (int *)malloc((csr->nrow + 1) * sizeof(int));
	HOST_CHECK(csr->row_ptr);

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

	return OK;
}



int preprocess(cvr_t *d_cvr, csr_t *d_csr, csr_t *h_csr, int *rowID, int *valID, int *count){
	printf("\nPreprocess start.\n");

	dim3 grid(gridDim[0], gridDim[1], gridDim[2]);
	dim3 block(blockDim[0], blockDim[1], blockDim[2]);

	int n_blocks = blockDim[0] * blockDim[1] * blockDim[2];
	int nnz_per_block = h_csr->nnz / n_blocks;
	int change_block_nnz = h_csr->nnz % n_blocks;

	preprocess_kernel<<<grid, block>>>(d_cvr, d_csr, nnz_per_block, change_block_nnz, rowID, valID, count);
	CHECK(cudaGetLastError());
	cudaDeviceSynchronize();

	printf("OK!\n\n");

	return OK;
}



int spmv(floatType *d_y, floatType *d_x, cvr_t *d_cvr, csr_t *d_csr, csr_t *h_csr){
	printf("\nSparse Matrix-Vector multiply start.\n");

	dim3 grid(gridDim[0], gridDim[1], gridDim[2]);
	dim3 block(blockDim[0], blockDim[1], blockDim[2]);

	int iteration;
	//FOR1
	for(iteration = 0; iteration < n_iterations; iteration++){
		spmv_kernel<<<grid, block>>>\
			();
		CHECK(cudaGetLastError());
		cudaDeviceSynchronize();
	} //ENDFOR1: iteration

	printf("OK\n");

	return OK;
}



__global__ void preprocess_kernel(cvr_t *cvr, csr_t *csr, int const nnz_per_block, int const change_block_nnz, \
	int *rowID, int *valID, int *count){
	// general case
	unsigned int block_num = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.y * gridDim.x;
	unsigned int thread_offset = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x;
	unsigned int threads_per_block = blockDim.x * blockDim.y * blockDim.z;
	unsigned int thread_num = block_num * threads_per_block + thread_offset;

	// 1-dimension case
	//unsigned int block_num = blockIdx.x;
	//unsigned int thread_offset = threadIdx.x;
	//unsigned int threads_per_block = blockDim.x;
	//unsigned int thread_num = block_num * threads_per_block + thread_offset;


	unsigned int block_start, block_end, block_nnz;
	unsigned int block_start_row, block_end_row, block_nrow;

	// information about row range and non-zeros in this block
	if(block_num < change_block_nnz){
		block_start = block_num * nnz_per_block + block_num * 1;
		block_end = (block_num + 1) * nnz_per_block + (block_num + 1) * 1 - 1;
	}else{
		block_start = block_num * nnz_per_block + change_block_nnz * 1;
		block_end = (block_num + 1) * nnz_per_block + change_block_nnz * 1 - 1;
	}
	block_nnz = block_end - block_start + 1;

	// IF0: non-empty block
	if(block_nnz > 0){
		block_start_row = func_get_row(block_start, d_csr);
		block_end_row = func_get_row(block_end, d_csr);
		block_nrow = block_end_row - block_start_row + 1;

		int block_n_vals = (block_nnz + threads_per_block - 1) / threads_per_block * threads_per_block;
		int block_n_recs = (block_nrow + threads_per_block * 2 - 1) / threads_per_block * threads_per_block;

		int valID, rowID, count;
		__shared__ int rs = block_start_row;

		// initialize valID, rowID, count for preprocessing
		rowID = rs + thread_offset;
		if(rowID > block_end_row){
			rowID = -1;
			valID = -1;
			count = 0;
		}else{
			valID = csr->row_ptr[rowID];
			count = csr->row_ptr[rowID+1] - csr->row_ptr[rowID];
			if(rowID == block_start_row){
				valID = block_start;
				count = csr->row_ptr[rowID+1] - block_start;
			}
			if(rowID == thread_end_row){
				count = block_end + 1 - valID;
			}
			valID -= block_start;
		}
		if(thread_num == 0){
			rs += threads_per_block;
		}
		__syncthreads();

		//IF1: if the number of rows is less than threads_per_block, initialize tail_ptr
		if(rs > block_end_row){
			cvr->tail_ptr[thread_num] = rowID;
			if(rowID != -1){
				rowID = thread_num;
			}
		}
	} // END IF0


/****************** I am the beautiful split line ******************/
/*
TODO 1: there's one thing remained above: how to deal with rs??
TODO 2: change floatType type

*/

		// IF0: if this thread has at least one number to deal with
		if(thread_nnz > 0){
			int rec_idx = 0, gather_base = 0;
			cvr->lrrec_ptr[thread_num] = -1;
			// FOR1
			for(i = 0; i <= thread_n_vals / n_block_threads; i++){
				// IF2: if some lanes are empty
				if(0 == func_AND(thread_count, n_block_threads)){
					// FOR2: iterate over all lanes, feed or steal
					for(j = 0; j < n_block_threads; j++){
						// IF3: lane[j] is empty
						if(0 == thread_count[j]){
							//EQUAL is included because thread_end_row can be reached
							while(thread_rs <= thread_end_row && csr->row_ptr[thread_rs+1] == csr->row_ptr[thread_rs]){
								thread_rs++;
							}

							//recording
							if(-1 != thread_rowID[j]){
								cvr->rec_ptr[thread_num][rec_idx++] = (i - 1) * n_block_threads + j; //valID which triggers write back
								cvr->rec_ptr[thread_num][rec_idx++] = thread_rowID[j]; //write back position
							}

							//IF4: tracker feeding
							//if next non-empty line exsits, non-empty is guaranteed by previous while loop
							if(thread_rs <= thread_end_row){
								thread_rowID[j] = thread_rs;
								thread_valID[j] = csr->row_ptr[thread_rs] - thread_start;
								thread_count[j] = csr->row_ptr[thread_rs+1] - csr->row_ptr[thread_rs];
								//ELSE1: if the number of rows is more than n_block_threads, initialize tail_ptr
								//the lane deals thread_end_row must reaches here
								if(thread_rs == thread_end_row){
									thread_count[j] = thread_end + 1 - thread_valID[j] - thread_start;
									for(k = 0; k < n_block_threads; k++){
										cvr->tail_ptr[thread_num][k] = thread_rowID[k];
										thread_rowID[k] = k;
									}
								}
								thread_rs++;

							}else{ //ELSE4: tracker stealing, thread_rs is not important since then
								if(-1 == cvr->lrrec_ptr[thread_num]){
									int temp_lrrec = (i - 1) * n_block_threads + j;
									cvr->lrrec_ptr[thread_num] = temp_lrrec < 0 ? 0 : temp_lrrec;
								}

								int average = func_average(thread_count, n_block_threads);
								//if no remainding numbers exist
								if(0 == average){
									if(i != thread_n_vals / n_block_threads){
										printf("ERROR: *** last round of preprocessing is incorrect ***\n");
										exit(ERROR);
									}
									continue;
								}else{
									int candidate = 0;
									//get candidate to steal
									for( ; candidate < n_block_threads; candidate++){
										if(thread_count[candidate] > average){
											break;
										}
									}
									//if padding is needed
									if(candidate == n_block_threads){
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
				for(j = 0; j < n_block_threads && i < thread_n_vals / n_block_threads; j++){
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
				gather_base += n_block_threads;
			} //ENDFOR1
		} //ENDIF0

}

__global__ void spmv_kernel(){
	#pragma omp parallel num_threads(n_blocks)
		{
			int thread_num = omp_get_thread_num();

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
				floatType *thread_y = (floatType *)malloc(cvr->nrow * sizeof(floatType));
				if(NULL == thread_y){
					printf("ERROR: *** memory overflow in spmv(), thread_%d ***\n", thread_num);
					exit(ERROR);
				}
				memset(thread_y, 0, cvr->nrow * sizeof(floatType));

				//store the intermediate result
				floatType *thread_temp = (floatType *)malloc(n_block_threads * sizeof(floatType));
				if(NULL == thread_temp){
					printf("ERROR: *** memory overflow in spmv(), thread_%d ***\n", thread_num);
					exit(ERROR);
				}
				memset(thread_temp, 0, n_block_threads * sizeof(floatType));

				int rec_idx = 0;
				int offset, writeback;
				int i, j;
				//FOR2
				for(i = 0; i < (thread_nnz + n_block_threads - 1) / n_block_threads; i++){
					for(j = 0; j < n_block_threads; j++){
						int x_offset = cvr->colidx_ptr[thread_num][i*n_block_threads+j];
						thread_temp[j] += cvr->val_ptr[thread_num][i*n_block_threads+j] * x[x_offset];
					}
					//corresponding to tracker feeding part
					if(cvr->rec_ptr[thread_num][rec_idx] < cvr->lrrec_ptr[thread_num]){
						//more than one temporary result could be processed here
						while(cvr->rec_ptr[thread_num][rec_idx] / n_block_threads == i){
							if(cvr->rec_ptr[thread_num][rec_idx] < cvr->lrrec_ptr[thread_num]){
								offset = cvr->rec_ptr[thread_num][rec_idx++] % n_block_threads;
								writeback = cvr->rec_ptr[thread_num][rec_idx++];
								thread_y[writeback] = thread_temp[offset];
								thread_temp[offset] = 0;
							}else{ // in case rec[rec_idx] < lrrec < rec[rec_idx+2]
								offset = cvr->rec_ptr[thread_num][rec_idx++] % n_block_threads;
								writeback = cvr->rec_ptr[thread_num][rec_idx++];
								if(-1 != cvr->tail_ptr[thread_num][writeback]){
									thread_y[cvr->tail_ptr[thread_num][writeback]] += thread_temp[offset];
								}
								thread_temp[offset] = 0;
							}
						}
					}else{//corresponding to tracker stealing part
						while(cvr->rec_ptr[thread_num][rec_idx] / n_block_threads == i){
							offset = cvr->rec_ptr[thread_num][rec_idx++] % n_block_threads;
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
			} //ENDIF0

		} //ENDPRAGMA
}

