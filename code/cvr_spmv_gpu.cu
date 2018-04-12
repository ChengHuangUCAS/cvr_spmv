/***************************************************
**
** cvr_spmv_gpu.cu: GPU version of CVR spmv
**
** Compile:
**      $ nvcc cvr_spmv_gpu.cu -o cvr_spmv_gpu -arch=sm_52
** Your CUDA and architecture should at least support warp shuffle and single-precision float atomic add.
**  (e.g. CUDA x.x and compute capability x.x)
**
** run:
**      $ ./cvr_spmv_gpu data.txt [blocks threads [n_iterations]]
** data.txt: matrix market format input file
** default parameters: 1 block, 32 threads per block, 1 iteration
** 
** Default Matrix Market Format store base: 0.
**  If your file is 1-based, please change "#define COO_BASE 0" into 1.
** Default float type: double-precision(double).
**  If you need single-precision(float) type, please comment "#define DOUBLE".
**
****************************************************/

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

//#define TEXTURE

#define DOUBLE

#ifdef DOUBLE
#define floatType double
#else
#define floatType float
#endif

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
    int *colIDx;
    int *row_ptr;
}csr_t; // compressed sparse row format

typedef struct record{
    int pos;
    int wb;
}record_t;

typedef struct cvr{
    int ncol;                   //number of columns
    int nrow;                   //number of rows
    int nnz;                    //number of non-zeros
    floatType *val;             //values stored in cvr-special order
    int *colidx;                //column numbers corresponding to values
    //values in cvr are re-ordered for performance, following elements are used to record how to write to vector y(in spmv: y=Wx)
    record_t *rec;              //records of write-back information
    int *rec_threshold;         //i don't know how to describe this, if you've read the paper, this is lr_rec in the paper
    int *threshold_detail;      //this is new, because threshold is a bit more complicated in GPU version
    int *tail;                  //the last line number(e.g. write-back position, think about it) of each lane(or thread as you like)
    int *warp_start_row;
    int *warp_nnz;
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

// auxiliary function to get row number, binary search
__forceinline__ __device__ int func_get_row(int const valID, csr_t const __restrict__ *csr){
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

// auxiliary function to implement atomic add, GPUs whose compute capability is lower than 6.0 do not support atomic add for double variables
__forceinline__ __device__ floatType floatTypeAtomicAdd(floatType const *address, floatType const val){
    #if __CUDA_ARCH__ >= 600
    return atomicAdd(address, val);
    #else
    #ifdef DOUBLE
    unsigned long long int *address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do{
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    }while(assumed != old);
    return __longlong_as_double(old);
    #else
    return atomicAdd(address, val);
    #endif
    #endif
}

// auxiliary function to compare result
inline int func_compare(floatType y, floatType y_verify){
    if(abs(y-y_verify) > abs(0.01*y_verify)){
        return 1;
    }else{
        return 0;
    }
}

// auxiliary function to initialize vector x(in spmv y=Wx)
inline void func_initialData(floatType *ip, int size){
    time_t t;
    srand((unsigned)time(&t));
    for(int i = 0; i < size; i++)    {
        ip[i] = (floatType)(rand() & 0xff) / 10.0f;
//        ip[i] = i % 10;
    }
}

// COO_BASE-based Matrix Market format -> CSR format
int read_matrix(csr_t *csr, char *filename);
// CSR format -> CVR format
int preprocess(cvr_t *d_cvr, csr_t *d_csr, int n_warps);
// CVR format SpMV, y = y + M * x
int spmv(floatType *d_y, floatType *d_x, cvr_t *d_cvr, csr_t *csr, int threads_per_block);

__global__ void preprocess_kernel(cvr_t * const __restrict__ cvr, csr_t * const __restrict__ csr);
__global__ void spmv_kernel(floatType * const __restrict__ y, floatType * const __restrict__ x, cvr_t * const __restrict__ cvr, csr_t * const __restrict__ csr, const int n_iterations);

#ifdef TEXTURE
#ifdef DOUBLE
texture<int2, 1, cudaReadModeElementType> x_texRef;
#else
texture<floatType, 1, cudaReadModeElementType> x_texRef;
#endif
#endif

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

    //printf("Input file: %s\n", filename);
    //printf("Grid dimension: (%d, %d, %d)\n", griddim[0], griddim[1], griddim[2]);
    //printf("Block dimension: (%d, %d, %d)\n", blockdim[0], blockdim[1], blockdim[2]);
    //printf("Number of iterations: %d\n\n", n_iterations);
    //printf("input:%s, <<<%d,%d>>>, iterations:%d\n", filename, griddim[0], blockdim[0], n_iterations);

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
//    printf("Preparing device_csr...\n");

    csr_t *d_csr = NULL, temp_csr;
    //allocate device global memory
    CHECK(cudaMalloc(&d_csr, sizeof(csr_t)));

    temp_csr.ncol = h_csr->ncol;
    temp_csr.nrow = h_csr->nrow;
    temp_csr.nnz = h_csr->nnz;
    CHECK(cudaMalloc(&temp_csr.val, h_csr->nnz * sizeof(floatType)));
    CHECK(cudaMalloc(&temp_csr.colIDx, h_csr->nnz * sizeof(int)));
    CHECK(cudaMalloc(&temp_csr.row_ptr, (h_csr->nrow + 1) * sizeof(int)));

    //initialize, device addresses like d_csr->val can't be accessed directly
    CHECK(cudaMemcpy(temp_csr.val, h_csr->val, h_csr->nnz * sizeof(floatType), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(temp_csr.colIDx, h_csr->colIDx, h_csr->nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(temp_csr.row_ptr, h_csr->row_ptr, (h_csr->nrow + 1) * sizeof(int), cudaMemcpyHostToDevice));

    CHECK(cudaMemcpy(d_csr, &temp_csr, sizeof(csr_t), cudaMemcpyHostToDevice));

//    printf("OK!\n\n");
    /****  \prepare device_csr  ****/


    /****  prepare host_x, device_x, host_y, device_y and verify_y  ****/
//    printf("Preparing vector x and y...\n");

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

    #ifdef TEXTURE
    //bind x vector to texture
    CHECK(cudaBindTexture(0, x_texRef, d_x));
    #endif

//    printf("OK!\n\n");
    /****  \prepare host_x, device_x, host_y, device_y and verify_y  ****/

    struct timeval tv1, tv2;
    double tv_diff1, tv_diff2;
    gettimeofday(&tv1, NULL);

    /****  prepare device_cvr  ****/
//    printf("Preparing device_cvr...\n");

    cvr_t *d_cvr = NULL, temp_cvr;
    //cvr structure is dependent on matrix and runtime configuration
    /*
    **  n_blocks: total number of blocks in this grid
    **  threads_per_block: number of threads in a block
    **  n_threads: total number of threads in this grid
    **  n_warps: total number of warps in this grid
    **  n_warp_nnz: average number of non-zeros dealed by one warp
    **  n_warp_vals: upper bound of number of non-zeros dealed by one warp, aligned
    **  n_warp_recs: upper bound of records needed by one warp, aligned
    */
    int n_blocks = griddim[0] * griddim[1] * griddim[2];
    int threads_per_block = blockdim[0] * blockdim[1] * blockdim[2];
    int n_threads = n_blocks * threads_per_block;

    int n_warps = (n_threads + THREADS_PER_WARP - 1) / THREADS_PER_WARP;
    int n_warp_nnz = h_csr->nnz / n_warps;
    int n_warp_vals = (n_warp_nnz + 1 + THREADS_PER_WARP - 1) / THREADS_PER_WARP * THREADS_PER_WARP;
    int n_warp_recs = n_warp_vals + THREADS_PER_WARP;

    int warps_per_block = threads_per_block / THREADS_PER_WARP;

    //allocate device global memory
    CHECK(cudaMalloc(&d_cvr, sizeof(cvr_t)));

    temp_cvr.ncol = h_csr->ncol;
    temp_cvr.nrow = h_csr->nrow;
    temp_cvr.nnz = h_csr->nnz;
    CHECK(cudaMalloc(&temp_cvr.val, n_warps * n_warp_vals * sizeof(floatType)));
    CHECK(cudaMalloc(&temp_cvr.colidx, n_warps * n_warp_vals * sizeof(int)));
    CHECK(cudaMalloc(&temp_cvr.rec, n_warps * n_warp_recs * sizeof(record_t)));
    CHECK(cudaMalloc(&temp_cvr.rec_threshold, n_warps * sizeof(int)));
    CHECK(cudaMalloc(&temp_cvr.threshold_detail, n_threads * sizeof(int)));
    CHECK(cudaMalloc(&temp_cvr.tail, n_threads * sizeof(int)));
    CHECK(cudaMalloc(&temp_cvr.warp_start_row, n_warps * sizeof(int)));
    CHECK(cudaMalloc(&temp_cvr.warp_nnz, n_warps * sizeof(int)));

    //initialize
    CHECK(cudaMemset(temp_cvr.rec, 0, n_warps * n_warp_recs * sizeof(record_t)));

    CHECK(cudaMemcpy(d_cvr, &temp_cvr, sizeof(cvr_t), cudaMemcpyHostToDevice));

//    printf("OK!\n\n");
    /****  \prepare device_cvr  ****/


    /****  launch kernels  ****/
    // PREPROCESS
    if(preprocess(d_cvr, d_csr, warps_per_block)){
        printf("ERROR occured in function preprocess()\n");
        return ERROR;
    }

    gettimeofday(&tv2, NULL);
    tv_diff1 = (tv2.tv_sec - tv1.tv_sec) * 1000000 + tv2.tv_usec - tv1.tv_usec;
    //printf("preprocess time: %lfms\n", tv_diff1/1000.0);



    gettimeofday(&tv1, NULL);

    // SPMV KERNEL
    if(spmv(d_y, d_x, d_cvr, d_csr, threads_per_block)){
        printf("ERROR occured in function spmv()\n");
        return ERROR;
    }

    gettimeofday(&tv2, NULL);
    tv_diff2 = (tv2.tv_sec - tv1.tv_sec) * 1000000 + tv2.tv_usec - tv1.tv_usec;
    printf("spmv time: %lfms\n", tv_diff2/n_iterations/1000.0);

    /****  \launch kernels  ****/


    /****  copy back  ****/

    CHECK(cudaMemcpy(h_y, d_y, h_csr->nrow * sizeof(floatType), cudaMemcpyDeviceToHost));

    /****  \copy back  ****/


    /****  free device memory  ****/
    #ifdef TEXTURE
    //unbind x vector to texture
    CHECK(cudaUnbindTexture(x_texRef));
    #endif
    
    CHECK(cudaFree(d_x));
    CHECK(cudaFree(d_y));

    CHECK(cudaFree(temp_cvr.val));
    CHECK(cudaFree(temp_cvr.colidx));
    CHECK(cudaFree(temp_cvr.rec));
    CHECK(cudaFree(temp_cvr.rec_threshold));
    CHECK(cudaFree(temp_cvr.threshold_detail));
    CHECK(cudaFree(temp_cvr.tail));    
    CHECK(cudaFree(d_cvr));

    CHECK(cudaFree(temp_csr.val));
    CHECK(cudaFree(temp_csr.colIDx));
    CHECK(cudaFree(temp_csr.row_ptr));
    CHECK(cudaFree(d_csr));

    /****  \free device memory  ****/


    /****  compute y_verify using csr spmv  ****/

    gettimeofday(&tv1, NULL);

    //for(int iteration = 0; iteration < n_iterations; iteration++){
        #pragma omp parallel for num_threads(24)
        for(int i = 0; i < h_csr->nrow; i++){
            floatType sum = 0;
            for(int j = h_csr->row_ptr[i]; j < h_csr->row_ptr[i+1]; j++){
              #pragma unroll
              for(int iteration = 0; iteration < n_iterations; iteration++){
                sum += h_csr->val[j] * h_x[h_csr->colIDx[j]];
              }
            }
            y_verify[i] += sum;
//            printf("y[%d]=%f, y_v[%d]=%f\n", i, h_y[i], i, y_verify[i]);
        }
    //}

    gettimeofday(&tv2, NULL);
    tv_diff2 = (tv2.tv_sec - tv1.tv_sec) * 1000000 + tv2.tv_usec - tv1.tv_usec;
    //printf("cpu_spmv time: %lfms\n", tv_diff2/1000.0);

    /****  \compute y_verify using csr spmv  ****/


    /****  check the result  ****/

    int count = 0;
    floatType y1 = 0, y2 = 0;
    for(int i = 0; i < h_csr->nrow; i++){
        if(func_compare(h_y[i], y_verify[i]) != CMP_EQUAL){
            y1 += h_y[i];
            y2 += y_verify[i];
            count++;
            if(count <= 10){
                #ifdef DOUBLE
                printf("y[%d] should be %lf, but the result is %lf\n", i, y_verify[i], h_y[i]);
                #else
                printf("y[%d] should be %f, but the result is %f\n", i, y_verify[i], h_y[i]);    
                #endif
            }
        }
//        if(count > 10){
//            break;
//        }
    }

    if(0 == count){
        printf("Correct\n\n");
    }else{
        #ifdef DOUBLE
        printf("count=%d, y_sum=%lf, y_v_sum=%lf\n", count, y1, y2);
        #else 
        printf("count=%d, y_sum=%f, y_v_sum=%f\n", count, y1, y2);
        #endif
    }

    /****  \check the result  ****/


    /****  free host memory  ****/

    free(h_x);
    free(h_y);
    free(y_verify);

    free(h_csr->val);
    free(h_csr->colIDx);
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
    
//    printf("Reading matrix...\n");

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
                #ifdef DOUBLE
                sscanf(buffer, "%d %d %lf %lf", &coo.triple[i].x, &coo.triple[i].y, &coo.triple[i].val, &im);
                #else
                sscanf(buffer, "%d %d %f %f", &coo.triple[i].x, &coo.triple[i].y, &coo.triple[i].val, &im);
                #endif
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
                #ifdef DOUBLE
                sscanf(buffer, "%d %d %lf", &coo.triple[i].x, &coo.triple[i].y, &coo.triple[i].val);
                #else
                sscanf(buffer, "%d %d %f", &coo.triple[i].x, &coo.triple[i].y, &coo.triple[i].val);
                #endif
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
                #ifdef DOUBLE
                sscanf(buffer, "%d %d %lf %lf", &coo.triple[i].x, &coo.triple[i].y, &coo.triple[i].val, &im);
                #else
                sscanf(buffer, "%d %d %f %f", &coo.triple[i].x, &coo.triple[i].y, &coo.triple[i].val, &im);
                #endif
            }
        }else{
            for(i = 0; i < coo.nnz; i++){
                fgets(buffer, sizeof(buffer), fp);
                #ifdef DOUBLE
                sscanf(buffer, "%d %d %lf", &coo.triple[i].x, &coo.triple[i].y, &coo.triple[i].val);
                #else
                sscanf(buffer, "%d %d %f", &coo.triple[i].x, &coo.triple[i].y, &coo.triple[i].val);
                #endif
            }
        }
    }
    fclose(fp);

    if(i > coo.nnz){
        printf("ERROR: *** too many matrix elements occered ***\n");
        return ERROR;
    }
//    printf("\nMatrix is in coordinate format now\n");

    //printf("Matrix Information:\n");
    //printf("Number of rows      : %d\n", coo.nrow);
    //printf("Number of columns   : %d\n", coo.ncol);
    //printf("Number of non-zeros : %d\n\n", coo.nnz);
    //printf("matrix:%dx%d, %d non-zeros\n", coo.nrow, coo.ncol, coo.nnz);

    //COO -> CSR
//    printf("Coverting to CSR format...\n");

    csr->ncol = coo.ncol;
    csr->nrow = coo.nrow;
    csr->nnz = coo.nnz;
    csr->val = (floatType *)malloc(csr->nnz * sizeof(floatType));
    HOST_CHECK(csr->val);
    csr->colIDx = (int *)malloc(csr->nnz * sizeof(int));
    HOST_CHECK(csr->colIDx);
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
        csr->colIDx[i] = coo.triple[i].y - COO_BASE;
    }
    while(r < csr->nrow){
        csr->row_ptr[++r] = i;
    }
//    printf("OK!\n\n");

    free(coo.triple);

    return OK;
}


/*
** function: preprocess()
**     convert csr format to cvr format
** parameters:
**     cvr_t *d_cvr       allocated cvr_t pointer(device)
**     csr_t *d_csr       initialized csr_t pointer(device)
**     int n_warps        number of warps, used to allocate shared memory
*/
int preprocess(cvr_t *d_cvr, csr_t *d_csr, int warps_per_block){
//    printf("Preprocess start.\n");

    dim3 grid(griddim[0], griddim[1], griddim[2]);
    dim3 block(blockdim[0], blockdim[1], blockdim[2]);

    //int exe_config[2];
    //cudaOccupancyMaxPotentialBlockSize(&exe_config[0], &exe_config[1], (void *)preprocess_kernel, 6*warps_per_block*sizeof(int));
    //printf("runtime API suggests: %d blocks per grid, %d threads per block for preprocess . size of shared memory:%d\n", exe_config[0], exe_config[1], 6*warps_per_block*sizeof(int));
    
    preprocess_kernel<<<grid, block, 6*warps_per_block*sizeof(int)>>>(d_cvr, d_csr);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

//    printf("OK!\n\n");

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
int spmv(floatType *d_y, floatType *d_x, cvr_t *d_cvr, csr_t *d_csr, int threads_per_block){
//    printf("Sparse Matrix-Vector multiply start.\n");

    dim3 grid(griddim[0], griddim[1], griddim[2]);
    dim3 block(blockdim[0], blockdim[1], blockdim[2]);

//    int exe_config[2];
//    cudaOccupancyMaxPotentialBlockSize(&exe_config[0], &exe_config[1], (void *)spmv_kernel, threads_per_block*sizeof(floatType));
//    printf("runtime API suggests: %d blocks per grid, %d threads per block for spmv. size of shared memory:%d\n", exe_config[0], exe_config[1], threads_per_block*sizeof(floatType));
    
    int iteration;
    //FOR1
    //#pragma unroll
    for(iteration = 0; iteration < n_iterations; iteration++){
        spmv_kernel<<<grid, block>>>(d_y, d_x, d_cvr, d_csr, n_iterations);
        CHECK(cudaGetLastError());
//        CHECK(cudaDeviceSynchronize());
    } //ENDFOR1: iteration
    CHECK(cudaDeviceSynchronize());
//    printf("OK!\n");

    return OK;
}



__global__ void preprocess_kernel(cvr_t * const __restrict__ cvr, csr_t * const __restrict__ csr){
    extern __shared__ int var_ptr[];
    /* 
    ** Basic information of block and thread:
    **   block_num:         current block id
    **   thread_offset:     current thread id in this block
    **   threads_per_block: number of threads in a block
    **   threadID:        current thread id in global vision
    **   n_blocks:          number of blocks in a grid
    **   warpID:          current warp id in global vision
    **   warp_offset:       current warp id in this block
    **   n_warps:           number of warps in a grid
    **   laneID:          current thread id in this warp
    */
    // general case
    //int block_num = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.y * gridDim.x;
    //int thread_offset = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.y * blockDim.x;
    //int threads_per_block = blockDim.x * blockDim.y * blockDim.z;
    //int threadID = block_num * threads_per_block + thread_offset;
    //int n_blocks = gridDim.x * gridDim.y * gridDim.z;

    // 1-dimension case
    int block_num = blockIdx.x;
    int thread_offset = threadIdx.x;
    int threads_per_block = blockDim.x;
    int threadID = block_num * threads_per_block + thread_offset;
    int n_blocks = gridDim.x;

    int warpID = threadID / THREADS_PER_WARP;
    int warp_offset = thread_offset / THREADS_PER_WARP;
    int n_warps = (n_blocks * threads_per_block + THREADS_PER_WARP - 1) / THREADS_PER_WARP;
    int warps_per_block = threads_per_block / THREADS_PER_WARP;
    int laneID = threadID % THREADS_PER_WARP;

    /*
    ** Information of non-zeros in a warp
    **   warp_start/warp_end:         first/last non-zero's id in this warp
    **   warp_nnz:                     number of non-zeros in this warp
    **   warp_start_row/warp_end_row: first/last non-zero's row id in this warp
    **   n_warp_nnz:                 average number of non-zeros in a warp
    **   change_warp_nnz:              first change_warp_nnz warps have one more non-zeros than others
    */
    int warp_start, warp_end, warp_nnz;
    int warp_start_row, warp_end_row;

    int n_warp_nnz = csr->nnz / n_warps;
    int change_warp_nnz = csr->nnz % n_warps;

    // information about row range and non-zeros in this warp
    if(warpID < change_warp_nnz){
        warp_start = warpID * n_warp_nnz + warpID * 1;
        warp_end = (warpID + 1) * n_warp_nnz + (warpID + 1) * 1 - 1;
    }else{
        warp_start = warpID * n_warp_nnz + change_warp_nnz * 1;
        warp_end = (warpID + 1) * n_warp_nnz + change_warp_nnz * 1 - 1;
    }
    warp_nnz = warp_end - warp_start + 1;

    // IF0: this warp has at least one non-zero to deal with 
    // ELSE0 is empty
    if(warp_nnz > 0){
        warp_start_row = func_get_row(warp_start, csr);
        warp_end_row = func_get_row(warp_end, csr);

        /*
        **   n_warp_vals: upperbound of number of values needed to store, related to memory space allocation
        **   n_warp_recs: upperbound of number of records needed to store, related to memory space allocation
        */
        //int n_warp_vals = ((n_warp_nnz + 1) + THREADS_PER_WARP - 1) / THREADS_PER_WARP * THREADS_PER_WARP;
        int n_warp_vals = (n_warp_nnz + 1 + THREADS_PER_WARP - 1) / THREADS_PER_WARP * THREADS_PER_WARP;
        int n_warp_recs = n_warp_vals + THREADS_PER_WARP;
        int n_steps = (warp_nnz + THREADS_PER_WARP - 1) / THREADS_PER_WARP; 

        /*
        ** Trackers
        **   valID: track current non-zero
        **   rowID: track current row
        **   count: track number of non-zeros unprocessed in current row
        **   recID: track current record number
        ** Other registers:
        **   warp_gather_base: used to reorder matrix values
        **   shfl_temp: used to help warp shuffle
        ** Shared memory arrays:
        **   cur_row: used to traverse rows in each warp
        **   cur_rec: used to traverse records in each warp
        **   count_and: used for reduce-and
        **   average, candidate, selected: used for tracker stealing
        */
        int valID, rowID, count, recID, warp_gather_base = warpID * n_warp_vals, shfl_temp;
        //volatile __shared__ int *cur_row, *cur_rec, *count_and, *average, *candidate, *selected;
        __shared__ int *cur_row, *cur_rec, *count_and, *average, *candidate, *selected;

        // initialize
        if(0 == thread_offset){
            cur_row = var_ptr;
            cur_rec = &var_ptr[warps_per_block];
            count_and = &var_ptr[2*warps_per_block];
            average = &var_ptr[3*warps_per_block];
            candidate = &var_ptr[4*warps_per_block];
            selected = &var_ptr[5*warps_per_block];
        }

        //__threadfence_block();
        __syncthreads();

        if(0 == laneID){
            cur_row[warp_offset] = warp_start_row;
            cur_rec[warp_offset] = warpID * n_warp_recs;
            count_and[warp_offset] = 1;
            average[warp_offset] = 0;
            candidate[warp_offset] = -1;
            selected[warp_offset] = -1;
            cvr->rec_threshold[warpID] = -1;
            cvr->warp_start_row[warpID] = warp_start_row;
            cvr->warp_nnz[warpID] = warp_nnz;
        }
        cvr->threshold_detail[threadID] = 1; // initially, no threads can write directly to rec.wb in threshold loop

        // initialize valID, rowID, count for preprocessing
        rowID = atomicAdd(&cur_row[warp_offset], 1);
        // empty rows
        while(rowID <= warp_end_row && csr->row_ptr[rowID+1] == csr->row_ptr[rowID]){
            rowID = atomicAdd(&cur_row[warp_offset], 1);
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
        }

        // IF1: if the number of rows is less than THREADS_PER_WARP, initialize tail_ptr
        if(cur_row[warp_offset] > warp_end_row){
            //if(count == 1 && rowID <= warp_end_row){
            //    cvr->threshold_detail[threadID] = 0; // these threads can write to rec.wb directly in threshold loop
            //}
            cvr->tail[threadID] = rowID;
            if(rowID != -1){
                rowID = threadID;
            }
            if(0 == laneID){
                cur_row[warp_offset] += THREADS_PER_WARP; // ensure IF4 and IF5(ELSE1) will never be executed
                cvr->rec_threshold[warpID] = 0;
            }
        } // END IF1

        // FOR1: preprocessing loop
        for(int i = 0; i <= n_steps; i++){

           if(0 == laneID){
                count_and[warp_offset] = 1;
            }
            atomicAnd(&count_and[warp_offset], count>0);
            // IF2: if count in some lane(s) = 0, recording and feeding/stealing is needed
            if(0 == count_and[warp_offset]){
                if(0 == count){
                    // IF3: recording
                    if(-1 != valID){
                        recID = atomicAdd(&cur_rec[warp_offset], 1);
                        cvr->rec[recID].pos = (i - 1) * THREADS_PER_WARP + laneID;
                        cvr->rec[recID].wb = rowID;
                    }// END IF3

                    // omit empty rows and get a new row
                    rowID = atomicAdd(&cur_row[warp_offset], 1);
                    while(rowID <= warp_end_row && csr->row_ptr[rowID+1] == csr->row_ptr[rowID]){
                        rowID = atomicAdd(&cur_row[warp_offset], 1);
                    }
                }
                // WHILE1: feeding/stealing one by one
                while(0 == count_and[warp_offset]){

                    // IF4: tracker feeding
                    if(cur_row[warp_offset] <= warp_end_row+THREADS_PER_WARP){
                        if(0 == count && rowID <= warp_end_row){
                            valID = csr->row_ptr[rowID];
                            count = csr->row_ptr[rowID+1] - valID;
                            if(warp_end_row == rowID){
                                count = warp_end - valID + 1;
                            }
                        }
                        // IF5 & ELSE1
                        if(cur_row[warp_offset] > warp_end_row){
                            cvr->tail[threadID] = rowID;
                            if(count == 0 && rowID <= warp_end_row){
                                cvr->threshold_detail[threadID] = 0; // these threads can write to rec.wb directly in threshold loop
                            }
                            rowID = threadID;
                        }
                    }// END IF4

                    // re-calculate count_and after possible tracker feeding
                    if(0 == laneID){
                        count_and[warp_offset] = 1;
                    }
                    atomicAnd(&count_and[warp_offset], count>0);

                    if(cur_row[warp_offset] > warp_end_row){
                        // IF6: set rec_threshold, only executed once
                        if(-1 == cvr->rec_threshold[warpID]){
                            
                            if(0 == laneID){
                                // make sure once IF6 is executed, IF4 will never be executed 
                                cur_row[warp_offset] += THREADS_PER_WARP;
                                cvr->rec_threshold[warpID] = i;
                            }
                        }// END IF6
                    }

                    // IF7: tracker stealing
                    if(0 == count_and[warp_offset] && cur_row[warp_offset] > warp_end_row){
                        // calculate average count
                        if(0 == laneID){
                            average[warp_offset] = 0;
                        }
                        atomicAdd(&average[warp_offset], count);
                        if(0 == laneID){
                            average[warp_offset] = (average[warp_offset] + THREADS_PER_WARP - 1) / THREADS_PER_WARP;
                        }   
    
                        // IF8: stealing
                        if(0 == average[warp_offset]){
                            if(i != n_warp_vals / THREADS_PER_WARP){
                                //printf("ERROR: *** last round of preprocessing is incorrect ***\n");
                            }
                            break;
                        }else{
                            // find candidate to steal
                            if(0 == laneID){
                                candidate[warp_offset] = -1;
                            }
                            if(count > average[warp_offset]){
                                switch(laneID){
                                    case 31: candidate[warp_offset] = 31; break;
                                    case 30: candidate[warp_offset] = 30; break;
                                    case 29: candidate[warp_offset] = 29; break;
                                    case 28: candidate[warp_offset] = 28; break;
                                    case 27: candidate[warp_offset] = 27; break;
                                    case 26: candidate[warp_offset] = 26; break;
                                    case 25: candidate[warp_offset] = 25; break;
                                    case 24: candidate[warp_offset] = 24; break;
                                    case 23: candidate[warp_offset] = 23; break;
                                    case 22: candidate[warp_offset] = 22; break;
                                    case 21: candidate[warp_offset] = 21; break;
                                    case 20: candidate[warp_offset] = 20; break;
                                    case 19: candidate[warp_offset] = 19; break;
                                    case 18: candidate[warp_offset] = 18; break;
                                    case 17: candidate[warp_offset] = 17; break;
                                    case 16: candidate[warp_offset] = 16; break;
                                    case 15: candidate[warp_offset] = 15; break;
                                    case 14: candidate[warp_offset] = 14; break;
                                    case 13: candidate[warp_offset] = 13; break;
                                    case 12: candidate[warp_offset] = 12; break;
                                    case 11: candidate[warp_offset] = 11; break;
                                    case 10: candidate[warp_offset] = 10; break;
                                    case  9: candidate[warp_offset] =  9; break;
                                    case  8: candidate[warp_offset] =  8; break;
                                    case  7: candidate[warp_offset] =  7; break;
                                    case  6: candidate[warp_offset] =  6; break;
                                    case  5: candidate[warp_offset] =  5; break;
                                    case  4: candidate[warp_offset] =  4; break;
                                    case  3: candidate[warp_offset] =  3; break;
                                    case  2: candidate[warp_offset] =  2; break;
                                    case  1: candidate[warp_offset] =  1; break;
                                    case  0: candidate[warp_offset] =  0;
                                }
                            }

                            // select one lane that need to steal
                            if(0 == count){
                                switch(laneID){
                                    case 31: selected[warp_offset] = 31; break;
                                    case 30: selected[warp_offset] = 30; break;
                                    case 29: selected[warp_offset] = 29; break;
                                    case 28: selected[warp_offset] = 28; break;
                                    case 27: selected[warp_offset] = 27; break;
                                    case 26: selected[warp_offset] = 26; break;
                                    case 25: selected[warp_offset] = 25; break;
                                    case 24: selected[warp_offset] = 24; break;
                                    case 23: selected[warp_offset] = 23; break;
                                    case 22: selected[warp_offset] = 22; break;
                                    case 21: selected[warp_offset] = 21; break;
                                    case 20: selected[warp_offset] = 20; break;
                                    case 19: selected[warp_offset] = 19; break;
                                    case 18: selected[warp_offset] = 18; break;
                                    case 17: selected[warp_offset] = 17; break;
                                    case 16: selected[warp_offset] = 16; break;
                                    case 15: selected[warp_offset] = 15; break;
                                    case 14: selected[warp_offset] = 14; break;
                                    case 13: selected[warp_offset] = 13; break;
                                    case 12: selected[warp_offset] = 12; break;
                                    case 11: selected[warp_offset] = 11; break;
                                    case 10: selected[warp_offset] = 10; break;
                                    case  9: selected[warp_offset] =  9; break;
                                    case  8: selected[warp_offset] =  8; break;
                                    case  7: selected[warp_offset] =  7; break;
                                    case  6: selected[warp_offset] =  6; break;
                                    case  5: selected[warp_offset] =  5; break;
                                    case  4: selected[warp_offset] =  4; break;
                                    case  3: selected[warp_offset] =  3; break;
                                    case  2: selected[warp_offset] =  2; break;
                                    case  1: selected[warp_offset] =  1; break;
                                    case  0: selected[warp_offset] =  0;
                                }
                            }
    
                            // if no candidate, padding
                            if(-1 == candidate[warp_offset]){
                                if(selected[warp_offset] == laneID){
                                    valID = -1;
                                    count = 1;
                                }
                            }else{
                                shfl_temp = __shfl(valID, candidate[warp_offset]);
                                if(selected[warp_offset] == laneID){
                                    rowID = candidate[warp_offset] + warpID * THREADS_PER_WARP;
                                    valID = shfl_temp;
                                    count = average[warp_offset];
                                    selected[warp_offset] = -1;
                                }
                                if(candidate[warp_offset] == laneID){
                                    rowID = candidate[warp_offset] + warpID * THREADS_PER_WARP;
                                    valID = valID + average[warp_offset];
                                    count = count - average[warp_offset];
                                    candidate[warp_offset] = -1;
                                }
                            }
                        

                        } // END IF8
                    } // END IF7

                    // re-calculate count_and, if = 1, jump out of while loop
                    if(0 == laneID){
                        count_and[warp_offset] = 1;
                    }
                    atomicAnd(&count_and[warp_offset], count>0);
                } // END WHILE1
            } // END IF2 

            // in the last round of for loop, the only thing need to do is recording
            if(warp_gather_base >= (warpID + 1) * n_warp_vals){
                continue;
            }
            int addr = warp_gather_base + laneID;
            if(-1 == valID){
                cvr->val[addr] = 0;
                cvr->colidx[addr] = 0;
            }else{
                cvr->val[addr] = csr->val[valID];
                cvr->colidx[addr] = csr->colIDx[valID];
                valID++;
            }
            count--;
            warp_gather_base += THREADS_PER_WARP;
            
        } // END FOR1

    } // END IF0

}


__global__ void spmv_kernel(floatType * const __restrict__ y, floatType * const __restrict__ x, cvr_t * const __restrict__ cvr, csr_t * const __restrict__ csr, const int n_iterations){
    //extern __shared__ floatType shared_y[];

    // these variables are the same as preprocess_kernel
    // warp_offset is useless here because it's used to access shared memory
    int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    int warpID = threadID / THREADS_PER_WARP;
    //int warp_offset = threadIdx.x / THREADS_PER_WARP;
    int n_warps = (gridDim.x * blockDim.x + THREADS_PER_WARP - 1) / THREADS_PER_WARP;
    int laneID = threadID % THREADS_PER_WARP;

    int warp_start_row = cvr->warp_start_row[warpID];

    //int n_warp_vals = ((n_warp_nnz + 1) + THREADS_PER_WARP - 1) / THREADS_PER_WARP * THREADS_PER_WARP;
    int n_warp_vals = (cvr->nnz / n_warps + 1 + THREADS_PER_WARP - 1) / THREADS_PER_WARP * THREADS_PER_WARP;
    int n_warp_recs = n_warp_vals + THREADS_PER_WARP;
    int n_steps = (cvr->warp_nnz[warpID] + THREADS_PER_WARP - 1) / THREADS_PER_WARP;

    //floatType *shared_y = &shared_var[warp_offset * n_warp_recs];
    //int init_smem = laneID;
    //while(init_smem < n_warp_recs){
    //    shared_y[init_smem] = 0;
    //    init_smem += THREADS_PER_WARP;
    //}
    //__syncthreads();
        
    /*
    ** temp_result: temp result of current lane, write to y[] after finishing one row
    ** valID: offset of cvr->val and cvr->colidx
    ** recID: offset of cvr->rec, there is only one recID in one warp
    ** threshold: store cvr->rec_threshold of current warp
    */
    floatType temp_result = 0;
    int valID = warpID * n_warp_vals + laneID;
    int recID = warpID * n_warp_recs;
    int threshold = cvr->rec_threshold[warpID];
    int x_addr, rec_pos, writeback, rec_flag, threshold_flag;


    // FOR0
    #pragma unroll
    for(int i = 0; i < n_steps; i++){
        //
        //** x_addr: offset of vector x
        //** rec_pos: store cvr->rec.pos, used to calculate offset
        //** writeback: store cvr->rec.wb, address to write back
        //** offset: lane number of current record
        //
        x_addr = cvr->colidx[valID];

        // ******** this is the core multiplication!!!!!!!!! ********
 
        #ifdef TEXTURE
            
        #ifdef DOUBLE
        int2 x_trans = tex1Dfetch(x_texRef, x_addr);
        floatType x_val = __hiloint2double(x_trans.y, x_trans.x);
        #else
        floatType x_val = tex1Dfetch(x_texRef, x_addr);
        #endif

        temp_result += cvr->val[valID] * x_val;
            
        #else

        //temp_result += __ldg(&cvr->val[valID]) * __ldg(&x[x_addr]);
        temp_result += cvr->val[valID] * x[x_addr];
            
        #endif

        rec_pos = cvr->rec[recID].pos;
        rec_flag = 0;
        while(rec_pos / THREADS_PER_WARP == i){
            if(rec_pos % THREADS_PER_WARP == laneID){
                rec_flag = 1;
                writeback = cvr->rec[recID].wb;
            }
            recID++;
            if(recID >= (warpID + 1) * n_warp_recs){
                break;
            }
            rec_pos = cvr->rec[recID].pos;
        }
            
        // corresponding to tracker feeding stage
        if(1 == rec_flag){
            if(i < threshold){
                if(writeback == warp_start_row){
                    floatTypeAtomicAdd(&y[writeback], temp_result);
                }else{
                    y[writeback] += temp_result;
                }
                    
            }else if(i == threshold){
                threshold_flag = cvr->threshold_detail[threadID];
                if(0 == threshold_flag){
                    if(writeback == warp_start_row){
                        floatTypeAtomicAdd(&y[writeback], temp_result);
                    }else{
                        y[writeback] += temp_result;
                    }
                }else{
                    writeback = cvr->tail[writeback];
                    if(-1 != writeback){
                        floatTypeAtomicAdd(&y[writeback], temp_result);
                    }
                }
            }else{
                writeback = cvr->tail[writeback];
                if(-1 != writeback){
                    floatTypeAtomicAdd(&y[writeback], temp_result);
                }
            }
            temp_result = 0;
            rec_flag = 0;
        }

        valID += THREADS_PER_WARP;
    } // END FOR0


  //if(0 == laneID){
  //  floatTypeAtomicAdd(&y[warp_start_row], shared_y[0]);
  //  floatTypeAtomicAdd(&y[warp_end_row], shared_y[warp_end_row-warp_start_row]);
  //  #pragma unroll
  //  for(int wb = warp_start_row + 1; wb < warp_end_row; wb++){
  //      y[wb] += shared_y[wb-warp_start_row];
  //  }
  //}
}





