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

#define floatType float
// if you change float to double
//    (1)%f in read_matrix() and verify answer in main() should be changed to %lf
//    (2)and atomicAdd in spmv_kernel() will be changed 

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
	floatType *val;
	int *colidx;
	record_t *rec;
	int *rec_threshold;
	int *tail;
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
	return ERROR;
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
//		ip[i] = (floatType)(rand() & 0xff) / 10.0f;
        ip[i] = i % 10;
	}
}

// 0-based Matrix Market format -> CSR format
int read_matrix(csr_t *csr, char *filename);
// CSR format -> CVR format
int preprocess(cvr_t *d_cvr, csr_t *d_csr, int n_warps);
// CVR format SpMV, y = y + M * x
int spmv(floatType *d_y, floatType *d_x, cvr_t *d_cvr);

__global__ void preprocess_kernel(cvr_t *cvr, csr_t *csr);
__global__ void spmv_kernel(floatType *y, floatType *x, cvr_t *cvr);

// however, in this implementation, only one dimension is used for intuition
int griddim[3] = {1, 1, 1};
int blockdim[3] = {32, 1, 1};

int n_iterations = 1;

int main(int argc, char **argv){

	/****  runtime configuration  ****/

	if(argc < 2){
		printf("ERROR: *** wrong parameter format ***\n");
		return ERROR;
	}
	char *filename = argv[1];

	if(argc > 2){
		griddim[0] = atoi(argv[2]);
		blockdim[0] = atoi(argv[3]);
		if(5 == argc){
			n_iterations = atoi(argv[4]);
		}
	}

	/****  \runtime configuration  ****/


	/****  basic runtime information  ****/

	printf("Input file: %s\n", filename);
	printf("Grid dimension: (%d, %d, %d)\n", griddim[0], griddim[1], griddim[2]);
	printf("Block dimension: (%d, %d, %d)\n", blockdim[0], blockdim[1], blockdim[2]);
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


	/****  prepare device_csr  ****/
    printf("Preparing device_csr...\n");

	csr_t *d_csr = NULL, temp_csr;
	//allocate device global memory
	CHECK(cudaMalloc(&d_csr, sizeof(csr_t)));

    temp_csr.ncol = h_csr->ncol;
    temp_csr.nrow = h_csr->nrow;
    temp_csr.nnz = h_csr->nnz;
	CHECK(cudaMalloc(&temp_csr.val, h_csr->nnz * sizeof(floatType)));
	CHECK(cudaMalloc(&temp_csr.col_idx, h_csr->nnz * sizeof(int)));
	CHECK(cudaMalloc(&temp_csr.row_ptr, (h_csr->nrow + 1) * sizeof(int)));

	//initialize
	CHECK(cudaMemcpy(temp_csr.val, h_csr->val, h_csr->nnz * sizeof(floatType), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(temp_csr.col_idx, h_csr->col_idx, h_csr->nnz * sizeof(int), cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(temp_csr.row_ptr, h_csr->row_ptr, (h_csr->nrow + 1) * sizeof(int), cudaMemcpyHostToDevice));

	CHECK(cudaMemcpy(d_csr, &temp_csr, sizeof(csr_t), cudaMemcpyHostToDevice));

    printf("OK!\n\n");
	/****  \prepare device_csr  ****/


	/****  prepare device_cvr  ****/
    printf("Preparing device_cvr...\n");

	cvr_t *d_cvr = NULL, temp_cvr;
	//cvr structure is dependent on matrix and runtime configuration
	/*
	**  n_blocks: total number of blocks in this grid
	**  threads_per_block: number of threads in a block
	**  n_threads: total number of threads in this grid
	**  n_warps: total number of warps in this grid
	**  n_warp_nnz: average number of non-zeros dealed by one warp
	**  n_warp_vals: upper bond of number of non-zeros dealed by one warp, aligned
	**  n_warp_recs: upper bond of records needed by one warp, aligned
	*/
	int n_blocks = griddim[0] * griddim[1] * griddim[2];
	int threads_per_block = blockdim[0] * blockdim[1] * blockdim[2];
	int n_threads = n_blocks * threads_per_block;

	int n_warps = (n_threads + THREADS_PER_WARP - 1) / THREADS_PER_WARP;
	int n_warp_nnz = h_csr->nnz / n_warps;
	int n_warp_vals = ((n_warp_nnz + 1) + THREADS_PER_WARP - 1) / THREADS_PER_WARP * THREADS_PER_WARP;
	int n_warp_recs = n_warp_vals + THREADS_PER_WARP;

	//allocate device global memory
	CHECK(cudaMalloc(&d_cvr, sizeof(cvr_t)));

    temp_cvr.ncol = h_csr->ncol;
    temp_cvr.nrow = h_csr->nrow;
    temp_cvr.nnz = h_csr->nnz;
	CHECK(cudaMalloc(&temp_cvr.val, n_warps * n_warp_vals * sizeof(floatType)));
	CHECK(cudaMalloc(&temp_cvr.colidx, n_warps * n_warp_vals * sizeof(int)));
	CHECK(cudaMalloc(&temp_cvr.rec, n_warps * n_warp_recs * sizeof(record_t)));
	CHECK(cudaMalloc(&temp_cvr.rec_threshold, n_warps * sizeof(int)));
	CHECK(cudaMalloc(&temp_cvr.tail, n_threads * sizeof(int)));

	//initialize
	CHECK(cudaMemset(temp_cvr.rec, 0, n_warps * n_warp_recs * sizeof(record_t)));

	CHECK(cudaMemcpy(d_cvr, &temp_cvr, sizeof(cvr_t), cudaMemcpyHostToDevice));

    printf("OK!\n\n");
	/****  \prepare device_cvr  ****/


	/****  prepare host_x, device_x, host_y, device_y and verify_y  ****/
    printf("Preparing vector x and y...\n");

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
	CHECK(cudaMemcpy(d_x, h_x, h_csr->ncol * sizeof(floatType), cudaMemcpyHostToDevice));
	memset(h_y, 0, h_csr->nrow * sizeof(floatType));
	CHECK(cudaMemset(d_y, 0, h_csr->nrow * sizeof(floatType)));
	memset(y_verify, 0, h_csr->nrow * sizeof(floatType));

    printf("OK!\n\n");
	/****  \prepare host_x, device_x, host_y, device_y and verify_y  ****/


	/****  launch kernels  ****/

//	struct timeval tv1, tv2;
//	long tv_diff1, tv_diff2;
//	gettimeofday(&tv1, NULL);

	// PREPROCESS
	if(preprocess(d_cvr, d_csr, n_warps)){
		printf("ERROR occured in function preprocess()\n");
		return ERROR;
	}
	
	// SPMV KERNEL
	if(spmv(d_y, d_x, d_cvr)){
		printf("ERROR occured in function spmv()\n");
		return ERROR;
	}

//	gettimeofday(&tv2, NULL);
//	tv_diff1 = (tv2.tv_sec - tv1.tv_sec) * 1000000 + tv2.tv_usec - tv1.tv_usec;
//	printf("cvr time(usec): %ld\n", tv_diff1);

	/****  \launch kernels  ****/


	/****  copy back  ****/

	CHECK(cudaMemcpy(h_y, d_y, h_csr->nrow * sizeof(floatType), cudaMemcpyDeviceToHost));

	/****  \copy back  ****/


	/****  free device memory  ****/

	CHECK(cudaFree(d_x));
	CHECK(cudaFree(d_y));

	CHECK(cudaFree(temp_cvr.val));
	CHECK(cudaFree(temp_cvr.colidx));
	CHECK(cudaFree(temp_cvr.rec));
	CHECK(cudaFree(temp_cvr.rec_threshold));
	CHECK(cudaFree(temp_cvr.tail));	
	CHECK(cudaFree(d_cvr));

	CHECK(cudaFree(temp_csr.val));
	CHECK(cudaFree(temp_csr.col_idx));
	CHECK(cudaFree(temp_csr.row_ptr));
	CHECK(cudaFree(d_csr));

	/****  \free device memory  ****/


	/****  compute y_verify using csr spmv  ****/

//	gettimeofday(&tv1, NULL);

	for(int iteration = 0; iteration < n_iterations; iteration++){
//		#pragma omp parallel for num_threads(omp_get_num_threads())
		floatType sum;
		for(int i = 0; i < h_csr->nrow; i++){
			sum = 0;
			for(int j = h_csr->row_ptr[i]; j < h_csr->row_ptr[i+1]; j++){
				sum += h_csr->val[j] * h_x[h_csr->col_idx[j]];
			}
			y_verify[i] += sum;
            printf("y[%d]=%f, y_v[%d]=%f\n", i, h_y[i], i, y_verify[i]);
		}
	}

//	gettimeofday(&tv2, NULL);
//	tv_diff2 = (tv2.tv_sec - tv1.tv_sec) * 1000000 + tv2.tv_usec - tv1.tv_usec;
//	printf("csr time(usec): %ld\n", tv_diff2);

	/****  \compute y_verify using csr spmv  ****/


	/****  check the result  ****/

	int count = 0;
	for(int i = 0; i < h_csr->nrow; i++){
		if(func_compare(h_y[i], y_verify[i]) != CMP_EQUAL){
			count++;
			printf("y[%d] should be %f, but the result is %f\n", i, y_verify[i], h_y[i]);	
		}
		if(count > 10){
			break;
		}
	}

	if(0 == count){
		printf("Correct\n");
	}

	/****  \check the result  ****/


	/****  free host memory  ****/

	free(h_x);
	free(h_y);
	free(y_verify);

	free(h_csr->val);
	free(h_csr->col_idx);
	free(h_csr->row_ptr);
	free(h_csr);

	/****  \free host memory  ****/

	return 0;
}


/*
** function: read_matrix()
**     read matrix from MMF file and covert it to csr format
** parameters:
**     csr_t *csr         allocated csr_t pointer
**     char *filename     Matrix Market Format file
*/
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
			floatType im;
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
	printf("OK!\n\n");

	free(coo.triple);

	return OK;
}


/*
** function: preprocess()
**     convert csr format to cvr format
** parameters:
**     cvr_t *d_cvr       allocated cvr_t pointer(device)
**     csr_t *d_csr       initialized csr_t pointer(device)
*/
int preprocess(cvr_t *d_cvr, csr_t *d_csr, int n_warps){
	printf("Preprocess start.\n");

	dim3 grid(griddim[0], griddim[1], griddim[2]);
	dim3 block(blockdim[0], blockdim[1], blockdim[2]);

	preprocess_kernel<<<grid, block, 2*n_warps*sizeof(int)>>>(d_cvr, d_csr);
	CHECK(cudaGetLastError());
	cudaDeviceSynchronize();

	printf("OK!\n\n");

	return OK;
}


/*
** function: spmv()
**     sparse matrix-vector multiplication using cvr format
** parameters:
**     floatType *d_y     allocated pointer(device) to store result y
**     floatType *d_x     initialized pointer(device) to store vector x
**     cvr_t *d_cvr       allocated cvr_t pointer(device)
*/
int spmv(floatType *d_y, floatType *d_x, cvr_t *d_cvr){
	printf("Sparse Matrix-Vector multiply start.\n");

	dim3 grid(griddim[0], griddim[1], griddim[2]);
	dim3 block(blockdim[0], blockdim[1], blockdim[2]);

	int iteration;
	//FOR1
	for(iteration = 0; iteration < n_iterations; iteration++){
		spmv_kernel<<<grid, block>>>(d_y, d_x, d_cvr);
		CHECK(cudaGetLastError());
		cudaDeviceSynchronize();
	} //ENDFOR1: iteration

	printf("OK\n");

	return OK;
}



__global__ void preprocess_kernel(cvr_t *cvr, csr_t *csr){
	extern __shared__ int var_ptr[];
	// general case
	/* 
	** Basic information of block and thread:
	**   block_num:         current block id
	**   thread_offset:     current thread id in this block
	**   threads_per_block: number of threads in a block
	**   thread_num:        current thread id in global vision
	**   n_blocks:          number of blocks in this grid
	**   warp_num:          current warp id in global vision
	**   n_warps:           number of warps in this grid
	**   lane_num:          current thread id in this warp
	*/
	//int block_num = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.y * gridDim.x;
	//int thread_offset = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x;
	//int threads_per_block = blockDim.x * blockDim.y * blockDim.z;
	//int thread_num = block_num * threads_per_block + thread_offset;
	//int n_blocks = gridDim.x * gridDim.y * gridDim.z;

	// 1-dimension case
	int block_num = blockIdx.x;
	int thread_offset = threadIdx.x;
	int threads_per_block = blockDim.x;
	int thread_num = block_num * threads_per_block + thread_offset;
	int n_blocks = gridDim.x;

	int warp_num = thread_num / THREADS_PER_WARP;
	int n_warps = (n_blocks * threads_per_block + THREADS_PER_WARP - 1) / THREADS_PER_WARP;
	int lane_num = thread_num % THREADS_PER_WARP;

	/*
	** Information of non-zeros in a warp
	**   warp_start/warp_end:         first/last non-zero's id in this warp
	**   warp_nnz:                     number of non-zeros in this warp
	**   warp_start_row/warp_end_row: first/last non-zero's row id in this warp
	**   nnz_per_warp:                 average number of non-zeros in a warp
	**   change_warp_nnz:              first n warps have one more non-zeros than others, n=change_warp_nnz
	*/
	int warp_start, warp_end, warp_nnz;
	int warp_start_row, warp_end_row;

	int nnz_per_warp = csr->nnz / n_warps;
	int change_warp_nnz = csr->nnz % n_warps;

	// information about row range and non-zeros in this warp
	if(warp_num < change_warp_nnz){
		warp_start = warp_num * nnz_per_warp + warp_num * 1;
		warp_end = (warp_num + 1) * nnz_per_warp + (warp_num + 1) * 1 - 1;
	}else{
		warp_start = warp_num * nnz_per_warp + change_warp_nnz * 1;
		warp_end = (warp_num + 1) * nnz_per_warp + change_warp_nnz * 1 - 1;
	}
	warp_nnz = warp_end - warp_start + 1;

	// IF0: this warp has at least one non-zero to deal with 
	// ELSE0 is empty
	if(warp_nnz > 0){
		warp_start_row = func_get_row(warp_start, csr);
		warp_end_row = func_get_row(warp_end, csr);

		/*
		**   n_warp_nnz: average number of non-zeros in a warp
		**   n_warp_vals: upperbond of number of values needed to store, related to memory space allocation
		**   n_warp_recs: upperbond of number of records needed to store, related to memory space allocation
		**   warp_n_vals: number of values needed to store, must be less than n_warp_vals
		**   warp_n_recs: number of records needed to store, must be less than n_warp_recs
		*/
		int n_warp_nnz = csr->nnz / n_warps;
		int n_warp_vals = ((n_warp_nnz + 1) + THREADS_PER_WARP - 1) / THREADS_PER_WARP * THREADS_PER_WARP;
		int n_warp_recs = n_warp_vals + THREADS_PER_WARP;

		/*
		** Trackers
		**   valID: id of non-zero that is preprocessing by current thread
		**   rowID: row id of non-zero that is preprocessing by current thread
		**   count: number of non-zeros haven't been preprocessed in this row
		**   recID: record id
		**   cur_row: row id of this row, used to traverse rows in the matrix
		**   cur_rec: current record id
		**   gather_base: calculate store address by gather_base + offset

		
		*/
		int valID, rowID, count, recID, warp_gather_base = warp_num * n_warp_vals;
		__shared__ int *cur_row, *cur_rec;
		// initialize
		if(0 == thread_num){
			cur_row = var_ptr;
			cur_rec = &var_ptr[n_warps];
		}
		__syncthreads();

		if(0 == lane_num){
			cur_row[warp_num] = warp_start_row;
			cur_rec[warp_num] = warp_num * n_warp_recs;
			cvr->rec_threshold[warp_num] = -1;
		}
//		__syncthreads();

		// initialize valID, rowID, count for preprocessing
		rowID = atomicAdd(&cur_row[warp_num], 1);
		// empty rows
		while(rowID <= warp_end_row && csr->row_ptr[rowID+1] == csr->row_ptr[rowID]){
			rowID = atomicAdd(&cur_row[warp_num], 1);
		}

		if(rowID > warp_end_row){
			rowID = -1;
			valID = -1;
			count = 0;
		}else{
			valID = csr->row_ptr[rowID];
			count = csr->row_ptr[rowID+1] - valID;
			if(rowID == warp_start_row){
				count = count + valID - warp_start;
				valID = warp_start;
			}
			if(rowID == warp_end_row){
				count = warp_end + 1 - valID;
			}
//			valID -= warp_start;
		}

		// IF1: if the number of rows is less than THREADS_PER_WARP, initialize tail_ptr
		if(cur_row[warp_num] > warp_end_row){
			cvr->tail[thread_num] = rowID;
			if(rowID != -1){
				rowID = thread_num;
			}
		} // END IF1

		// FOR1: preprocessing loop
		for(int i = 0; i <= n_warp_vals / THREADS_PER_WARP; i++){
			__shared__ int count_and;
			if(0 == lane_num){
				count_and = 1;
			}
			atomicAnd(&count_and, count);
			// IF2: if recording and feeding/stealing is needed
			if(0 == count_and){
				if(0 == count){
					// IF3: recording
					if(-1 != rowID){
						recID = atomicAdd(&cur_rec[warp_num], 1);
						cvr->rec[recID].pos = (i - 1) * THREADS_PER_WARP + lane_num;
						cvr->rec[recID].wb = rowID;
					}// END IF3

					// empty rows
					rowID = atomicAdd(&cur_row[warp_num], 1);
					while(rowID <= warp_end_row && csr->row_ptr[rowID+1] == csr->row_ptr[rowID]){
						rowID = atomicAdd(&cur_row[warp_num], 1);
					}
				}

				// IF4: tracker feeding
				// I'M NOT VERY SURE ABOUT THIS <= 
				if(cur_row[warp_num] <= warp_end_row+THREADS_PER_WARP){
					if(0 == count){
						valID = csr->row_ptr[rowID];
						count = csr->row_ptr[rowID+1] - valID;
						// IF5 & ELSE1
						if(cur_row[warp_num] > warp_end_row){
							cvr->tail[thread_num] = rowID;
							rowID = warp_num;
							// make sure once IF5 is executed, IF4 will never be executed 
							atomicAdd(&cur_row[warp_num], THREADS_PER_WARP);
						}
					}
				}else{ // ELSE4: tracker stealing
					// IF6: set rec_threshold, only executed once
					if(-1 == cvr->rec_threshold[warp_num]){
						__shared__ int temp_rec_threshold;
						switch(lane_num){
							case 31: temp_rec_threshold = 31;
							case 30: temp_rec_threshold = 30;
							case 29: temp_rec_threshold = 29;
							case 28: temp_rec_threshold = 28;
							case 27: temp_rec_threshold = 27;
							case 26: temp_rec_threshold = 26;
							case 25: temp_rec_threshold = 25;
							case 24: temp_rec_threshold = 24;
							case 23: temp_rec_threshold = 23;
							case 22: temp_rec_threshold = 22;
							case 21: temp_rec_threshold = 21;
							case 20: temp_rec_threshold = 20;
							case 19: temp_rec_threshold = 19;
							case 18: temp_rec_threshold = 18;
							case 17: temp_rec_threshold = 17;
							case 16: temp_rec_threshold = 16;
							case 15: temp_rec_threshold = 15;
							case 14: temp_rec_threshold = 14;
							case 13: temp_rec_threshold = 13;
							case 12: temp_rec_threshold = 12;
							case 11: temp_rec_threshold = 11;
							case 10: temp_rec_threshold = 10;
							case  9: temp_rec_threshold =  9;
							case  8: temp_rec_threshold =  8;
							case  7: temp_rec_threshold =  7;
							case  6: temp_rec_threshold =  6;
							case  5: temp_rec_threshold =  5;
							case  4: temp_rec_threshold =  4;
							case  3: temp_rec_threshold =  3;
							case  2: temp_rec_threshold =  2;
							case  1: temp_rec_threshold =  1;
							case  0: temp_rec_threshold =  0;
						}
						if(0 == lane_num){
							temp_rec_threshold += (i - 1) * THREADS_PER_WARP;
							cvr->rec_threshold[warp_num] = temp_rec_threshold < 0 ? 0 : temp_rec_threshold;
						}
					}// END IF6

					__shared__ int average;
					if(0 == lane_num){
						average = 0;
					}
					atomicAdd(&average, count);
					if(0 == lane_num){
						average = (average + THREADS_PER_WARP - 1) / THREADS_PER_WARP;
					}

					if(0 == average){
						if(i != n_warp_vals / THREADS_PER_WARP){
							printf("ERROR: *** last round of preprocessing is incorrect ***\n");
//							return ERROR;
						}
						continue;
					}else{
						__shared__ int candidate;
						if(0 == lane_num){
							candidate = -1;
						}
						if(count > average){
							switch(lane_num){
								case 31: candidate = 31;
								case 30: candidate = 30;
								case 29: candidate = 29;
								case 28: candidate = 28;
								case 27: candidate = 27;
								case 26: candidate = 26;
								case 25: candidate = 25;
								case 24: candidate = 24;
								case 23: candidate = 23;
								case 22: candidate = 22;
								case 21: candidate = 21;
								case 20: candidate = 20;
								case 19: candidate = 19;
								case 18: candidate = 18;
								case 17: candidate = 17;
								case 16: candidate = 16;
								case 15: candidate = 15;
								case 14: candidate = 14;
								case 13: candidate = 13;
								case 12: candidate = 12;
								case 11: candidate = 11;
								case 10: candidate = 10;
								case  9: candidate =  9;
								case  8: candidate =  8;
								case  7: candidate =  7;
								case  6: candidate =  6;
								case  5: candidate =  5;
								case  4: candidate =  4;
								case  3: candidate =  3;
								case  2: candidate =  2;
								case  1: candidate =  1;
								case  0: candidate =  0;
							}
						}

						__shared__ int selected;
						if(0 == count){
							switch(lane_num){
								case 31: selected = 31;
								case 30: selected = 30;
								case 29: selected = 29;
								case 28: selected = 28;
								case 27: selected = 27;
								case 26: selected = 26;
								case 25: selected = 25;
								case 24: selected = 24;
								case 23: selected = 23;
								case 22: selected = 22;
								case 21: selected = 21;
								case 20: selected = 20;
								case 19: selected = 19;
								case 18: selected = 18;
								case 17: selected = 17;
								case 16: selected = 16;
								case 15: selected = 15;
								case 14: selected = 14;
								case 13: selected = 13;
								case 12: selected = 12;
								case 11: selected = 11;
								case 10: selected = 10;
								case  9: selected =  9;
								case  8: selected =  8;
								case  7: selected =  7;
								case  6: selected =  6;
								case  5: selected =  5;
								case  4: selected =  4;
								case  3: selected =  3;
								case  2: selected =  2;
								case  1: selected =  1;
								case  0: selected =  0;
							}
						}

						if(-1 == candidate){
							if(selected == lane_num){
								valID = -1;
								count = 1;
							}
						}else{
							if(selected == lane_num){
								rowID = candidate;
								valID = __shfl(valID, candidate);
								count = average;
							}
							if(candidate == lane_num){
								rowID = candidate;
								valID = valID + average;
								count = count - average;
							}
						}
						

					} // END IF4
				}
			} // END IF2

			int addr = warp_gather_base + lane_num;
			if(-1 == valID){
				cvr->val[addr] = 0;
				cvr->colidx[addr] = 0;
			}else{
				cvr->val[addr] = csr->val[valID];
				cvr->colidx[addr] = csr->col_idx[valID];
			}
			valID++;
			count--;
            warp_gather_base += THREADS_PER_WARP;
			
		} // END FOR1

	} // END IF0

}


__global__ void spmv_kernel(floatType *y, floatType *x, cvr_t *cvr){

	//int block_num = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.y * gridDim.x;
	//int thread_offset = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x;
	//int threads_per_block = blockDim.x * blockDim.y * blockDim.z;
	//int thread_num = block_num * threads_per_block + thread_offset;
	//int n_blocks = gridDim.x * gridDim.y * gridDim.z;

	// 1-dimension case
	int block_num = blockIdx.x;
	int thread_offset = threadIdx.x;
	int threads_per_block = blockDim.x;
	int thread_num = block_num * threads_per_block + thread_offset;
	int n_blocks = gridDim.x;

	int warp_num = thread_num / THREADS_PER_WARP;
	int n_warps = (n_blocks * threads_per_block + THREADS_PER_WARP - 1) / THREADS_PER_WARP;
	int lane_num = thread_num % THREADS_PER_WARP;

	int warp_start, warp_end, warp_nnz;

	int nnz_per_warp = cvr->nnz / n_warps;
	int change_warp_nnz = cvr->nnz % n_warps;

	// information about row range and non-zeros in this warp
	if(warp_num < change_warp_nnz){
		warp_start = warp_num * nnz_per_warp + warp_num * 1;
		warp_end = (warp_num + 1) * nnz_per_warp + (warp_num + 1) * 1 - 1;
	}else{
		warp_start = warp_num * nnz_per_warp + change_warp_nnz * 1;
		warp_end = (warp_num + 1) * nnz_per_warp + change_warp_nnz * 1 - 1;
	}
	warp_nnz = warp_end - warp_start + 1;

	// IF0: this warp has at least one non-zero to deal with 
	// ELSE0 is empty
	if(warp_nnz > 0){
		int n_warp_nnz = cvr->nnz / n_warps;
		int n_warp_vals = ((n_warp_nnz + 1) + THREADS_PER_WARP - 1) / THREADS_PER_WARP * THREADS_PER_WARP;
		int n_warp_recs = n_warp_vals + THREADS_PER_WARP;

		floatType temp_result[THREADS_PER_WARP];
		temp_result[lane_num] = 0;
		int valID = warp_num * n_warp_recs + lane_num;
		int recID = warp_num * n_warp_recs;
		int threshold = cvr->rec_threshold[warp_num];
		for(int i = 0; i < (n_warp_nnz + THREADS_PER_WARP - 1) / THREADS_PER_WARP; i++){
			int x_addr = cvr->colidx[valID];
			temp_result[lane_num] += cvr->val[valID] * x[x_addr];

			int rec_pos = cvr->rec[recID].pos;
			int writeback = cvr->rec[recID].wb;
			int offset = rec_pos % THREADS_PER_WARP;
			if(rec_pos < threshold){
				while(rec_pos / THREADS_PER_WARP == i){
					if(lane_num == offset){
						if(rec_pos < threshold){
							atomicAdd(&y[writeback], temp_result[offset]);
							temp_result[offset] = 0;
						}else{
							writeback = cvr->tail[writeback];
							if(-1 != writeback){
								atomicAdd(&y[writeback], temp_result[offset]);
							}
							temp_result[offset] = 0;
						}
					}
					recID++;
					rec_pos = cvr->rec[recID].pos;
					writeback = cvr->rec[recID].wb;
					offset = rec_pos % THREADS_PER_WARP;
				}
			}else{
				while(rec_pos / THREADS_PER_WARP == i){
					if(lane_num == offset){
						writeback = cvr->tail[writeback];
						if(-1 != writeback){
							atomicAdd(&y[writeback], temp_result[offset]);
						}
						temp_result[offset] = 0;
					}
					recID++;
                    if(recID >= (warp_num + 1) * n_warp_recs){
                        break;
                    }
					rec_pos = cvr->rec[recID].pos;
					writeback = cvr->rec[recID].wb;
					offset = rec_pos % THREADS_PER_WARP;
				}
			}
			valID += THREADS_PER_WARP;
		}

	}// END IF0
}


/*
Possible optimizing method:
1. update y by (1)another kernel (2)shared memory (3)shared memory and nested kernel
2. 

*/




