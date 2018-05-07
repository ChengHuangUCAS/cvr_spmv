/***************************************************
**
** cvr_spmv_gpu.cu: GPU version of CVR spmv
**
** run:
**      $ make
**      $ ./cvr_spmv_gpu data.txt [#blocks #threads] [#n_iterations]
** data.txt: matrix market format input file
** default parameters: # of blocks and threads per block: autoselect, 1000 iteration
** default compute capability 5.2 (Maxwell)
** 
** Default Matrix Market Format store base: 1.
**  If your file is 0-based, please change "#define COO_BASE 1" into 0.
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

#include<math.h>

#define OK 0
#define ERROR -1

#define CMP_EQUAL 0

#define FIELD_LENGTH 128
#define COO_BASE 1

#define OMP_THREADS 12

#define THREADS_PER_WARP 32

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
    int *col_idx;
    int *row_ptr;
}csr_t; // compressed sparse row format

typedef struct record{
    int pos;
    unsigned mask;
    int wb[THREADS_PER_WARP];
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
__forceinline__ __device__ int func_get_row(int const valID, csr_t const csr){
    int start = 0, end = csr.nrow;
    int mid = (start + end) / 2;
    while(start <= end){
        if(csr.row_ptr[mid] > valID){
            end = mid - 1;
        }else if(mid < csr.nrow && csr.row_ptr[mid+1] <= valID){
            start = mid + 1;
        }else{
            while(mid < csr.nrow && csr.row_ptr[mid] == csr.row_ptr[mid+1]){
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
__forceinline__ __device__ floatType func_floatTypeAtomicAdd(floatType const *address, floatType const val){
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

// auxiliary function for reducing AND
__forceinline__ __device__ bool func_reduceAnd(bool count){
    bool count_res = count;
    int step = THREADS_PER_WARP / 2;
    #pragma unroll
    for(int i = 0; i < 5; i++){
        count_res &= __shfl_down(count_res, step);
        step /= 2;
    }
    count_res = __shfl(count_res, 0);
    return count_res;
}

// auxiliary function for reducing OR
__forceinline__ __device__ unsigned func_reduceBitOr(bool src){
    unsigned bitmap = src << (threadIdx.x % THREADS_PER_WARP);
    int step = THREADS_PER_WARP / 2;
    #pragma unroll
    for(int i = 0; i < 5; i++){
        bitmap |= __shfl_down(bitmap, step);
        step /= 2;
    }
    bitmap = __shfl(bitmap, 0);
    return bitmap;
}

// auxiliary function for reducing average
__forceinline__ __device__ int func_reduceAvg(int num){
    int avg_res = num;
    int step = THREADS_PER_WARP / 2;
    #pragma unroll
    for(int i = 0; i < 5; i++){
        avg_res += __shfl_down(avg_res, step);
        step /= 2;
    }
    avg_res = (__shfl(avg_res, 0) + THREADS_PER_WARP - 1) / THREADS_PER_WARP;
    return avg_res;
}

// auxiliary function for reducing selection
__forceinline__ __device__ int func_reduceSel(bool sel){
    unsigned sel_bitmap = func_reduceBitOr(sel);
    #pragma unroll
    for(int i = 0; i < THREADS_PER_WARP; i++){
        if(sel_bitmap & 1){
            return i;
        }else{
            sel_bitmap >>= 1;
        }
    }
    return -1;
}

// auxiliary function to compare result
inline int func_compare(floatType y, floatType y_verify){
    if(abs(y-y_verify) > 0.00001){
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
        //ip[i] = (floatType)(rand() & 0xff) / 10.0f;
        //ip[i] = i % 10;
        ip[i] = 1;
    }
}

// COO_BASE-based Matrix Market format -> CSR format
int read_matrix(csr_t *csr, char *filename);
// CSR format -> CVR format
int preprocess();
// CVR format SpMV, y = y + M * x
int spmv(floatType *d_y, floatType *d_x);

__global__ void preprocess_kernel();
__global__ void spmv_kernel(floatType * const __restrict__ y, floatType * const __restrict__ x);


// In this implementation, only one dimension is used for intuition
int grid_dim = 32;
int block_dim = 64;

int n_iterations = 1000;

__constant__ csr_t const_csr;
__constant__ cvr_t const_cvr;

int main(int argc, char **argv){

    /****  runtime configuration  ****/

    if(argc < 2){
        printf("ERROR: *** wrong parameter format ***\n");
        return ERROR;
    }
    char *filename = argv[1];

    /****  \runtime configuration  ****/


    /****  prepare host_csr  ****/

    //allocate memory
    csr_t *h_csr = (csr_t *)malloc(sizeof(csr_t));
    HOST_CHECK(h_csr);

    //read matrix to initialize
    if(read_matrix(h_csr, filename)){
        printf("ERROR occured in function read_matrix()\n");
        return ERROR;
    }

    printf("Matrix: (%d, %d), %d non-zeros.\n", h_csr->nrow, h_csr->ncol, h_csr->nnz);
    
    /****  \prepare host_csr  ****/

    int total_threads = floor(0.0591 * h_csr->nnz + 116038);
    int floor1 = floor(24.094 * pow(h_csr->nrow, 0.2423));
    block_dim = min(1024, max(64, floor1 - floor1 % 32));
    grid_dim = min(1024, max(32, total_threads / block_dim + total_threads / block_dim % 2));

    if(argc == 3){
        n_iterations = atoi(argv[2]);
    }else if(argc == 4){
        grid_dim = atoi(argv[2]);
        block_dim = atoi(argv[3]);
    }else if(argc == 5){
        grid_dim = atoi(argv[2]);
        block_dim = atoi(argv[3]);
        n_iterations = atoi(argv[4]);
    }

    printf("Matrix file:%s. Execution config:<<<%d,%d>>>. Iterations:%d.\n", filename, grid_dim, block_dim, n_iterations);


    /****  prepare device_csr  ****/

    csr_t temp_csr;
    //allocate device global memory
    temp_csr.ncol = h_csr->ncol;
    temp_csr.nrow = h_csr->nrow;
    temp_csr.nnz = h_csr->nnz;
    CHECK(cudaMalloc(&temp_csr.val, h_csr->nnz * sizeof(floatType)));
    CHECK(cudaMalloc(&temp_csr.col_idx, h_csr->nnz * sizeof(int)));
    CHECK(cudaMalloc(&temp_csr.row_ptr, (h_csr->nrow + 1) * sizeof(int)));

    //initialize, device addresses like d_csr->val can't be accessed directly
    CHECK(cudaMemcpy(temp_csr.val, h_csr->val, h_csr->nnz * sizeof(floatType), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(temp_csr.col_idx, h_csr->col_idx, h_csr->nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(temp_csr.row_ptr, h_csr->row_ptr, (h_csr->nrow + 1) * sizeof(int), cudaMemcpyHostToDevice));

    CHECK(cudaMemcpyToSymbol(const_csr, &temp_csr, sizeof(csr_t)));

    /****  \prepare device_csr  ****/


    /****  prepare host_x, device_x, host_y, device_y and verify_y  ****/

    //allocate memory
    floatType *h_x, *h_y, *y_verify, *d_x, *d_y;
    h_x = (floatType *)malloc(h_csr->ncol * sizeof(floatType));
    h_y = (floatType *)malloc(h_csr->nrow * sizeof(floatType));
    y_verify = (floatType *)malloc(h_csr->nrow * sizeof(floatType));
    HOST_CHECK(h_x);
    HOST_CHECK(h_y);
    HOST_CHECK(y_verify);
    CHECK(cudaMalloc(&d_x, h_csr->ncol * sizeof(floatType)));
    CHECK(cudaMalloc(&d_y, h_csr->nrow * sizeof(floatType)));

    //initialize
    func_initialData(h_x, h_csr->ncol);
    memset(h_y, 0, h_csr->nrow * sizeof(floatType));
    memset(y_verify, 0, h_csr->nrow * sizeof(floatType));
    CHECK(cudaMemcpy(d_x, h_x, h_csr->ncol * sizeof(floatType), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(d_y, 0, h_csr->nrow * sizeof(floatType)));

    /****  \prepare host_x, device_x, host_y, device_y and verify_y  ****/



        cvr_t temp_cvr;
    //cvr structure is dependent on matrix and runtime configuration
    int n_threads = grid_dim * block_dim;
    int n_warps = n_threads / THREADS_PER_WARP;

    // average number of non-zeros per warp
    int n_warp_nnz = h_csr->nnz / n_warps;
    // upperbound of needed loop iterations to finish preprocess/multiplication
    int ub_steps = (n_warp_nnz + 1 + THREADS_PER_WARP - 1) / THREADS_PER_WARP;
    int ub_warp_vals = ub_steps * THREADS_PER_WARP;
    int ub_warp_recs = ub_steps + 1;

    //allocate device global memory
    temp_cvr.ncol = h_csr->ncol;
    temp_cvr.nrow = h_csr->nrow;
    temp_cvr.nnz = h_csr->nnz;
    CHECK(cudaMalloc(&temp_cvr.val, n_warps * ub_warp_vals * sizeof(floatType)));
    CHECK(cudaMalloc(&temp_cvr.colidx, n_warps * ub_warp_vals * sizeof(int)));
    CHECK(cudaMalloc(&temp_cvr.rec, n_warps * ub_warp_recs * sizeof(record_t)));
    CHECK(cudaMalloc(&temp_cvr.rec_threshold, n_warps * sizeof(int)));
    CHECK(cudaMalloc(&temp_cvr.threshold_detail, n_warps * sizeof(int)));
    CHECK(cudaMalloc(&temp_cvr.tail, n_threads * sizeof(int)));
    CHECK(cudaMalloc(&temp_cvr.warp_nnz, n_warps * sizeof(int)));

    //initialize
    CHECK(cudaMemset(temp_cvr.tail, -1, n_threads * sizeof(int)));
    CHECK(cudaMemcpyToSymbol(const_cvr, &temp_cvr, sizeof(cvr_t)));

    // warming up
    if(preprocess()){
        printf("ERROR occured while warming up\n");
        return ERROR;
    }

    /****  preprocess time  ****/
    struct timeval tv1, tv2;
    double tv_diff1, tv_diff2;
    gettimeofday(&tv1, NULL);


    // PREPROCESS KERNEL
    if(preprocess()){
        printf("ERROR occured in function preprocess()\n");
        return ERROR;
    }

    gettimeofday(&tv2, NULL);
    tv_diff1 = (tv2.tv_sec - tv1.tv_sec) * 1000000 + tv2.tv_usec - tv1.tv_usec;
    printf("preprocess time: %lfms\n", tv_diff1/1000.0);

    /****  \preprocess time  ****/


    /****  spmv time  ****/

    gettimeofday(&tv1, NULL);
    // SPMV KERNEL, deciding which branch to take here for performance considering
    if(spmv(d_y, d_x)){
        printf("ERROR occured in function spmv()\n");
        return ERROR;
    }
    gettimeofday(&tv2, NULL);
    tv_diff2 = (tv2.tv_sec - tv1.tv_sec) * 1000000 + tv2.tv_usec - tv1.tv_usec;
    printf("spmv time: %lfms\n", tv_diff2/1000.0/n_iterations);

    /****  \spmv time  ****/


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
    CHECK(cudaFree(temp_cvr.threshold_detail));
    CHECK(cudaFree(temp_cvr.tail));
    CHECK(cudaFree(temp_cvr.warp_nnz));

    CHECK(cudaFree(temp_csr.val));
    CHECK(cudaFree(temp_csr.col_idx));
    CHECK(cudaFree(temp_csr.row_ptr));

    /****  \free device memory  ****/


    /****  compute y_verify using csr spmv  ****/

    gettimeofday(&tv1, NULL);

    for(int iteration = 0; iteration < n_iterations; iteration++){
        #pragma omp parallel for num_threads(OMP_THREADS)
        for(int i = 0; i < h_csr->nrow; i++){
            floatType sum = 0;
            for(int j = h_csr->row_ptr[i]; j < h_csr->row_ptr[i+1]; j++){
                sum += h_csr->val[j] * h_x[h_csr->col_idx[j]];
            }
            y_verify[i] += sum;
        }
    }

    gettimeofday(&tv2, NULL);
    tv_diff2 = (tv2.tv_sec - tv1.tv_sec) * 1000000 + tv2.tv_usec - tv1.tv_usec;
    printf("cpu_spmv time: %lfms\n", tv_diff2/1000.0);

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
    free(h_csr->col_idx);
    free(h_csr->row_ptr);
    free(h_csr);

    /****  \free host memory  ****/

    return 0;
}


/*
** function: read_matrix()
** programmer: Lukasz Wesolowski
** creation: July 2, 2010
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
    coo.nnz = i;

    //COO -> CSR
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

    free(coo.triple);

    return OK;
}


/*
** function: preprocess()
**     convert csr format to cvr format
*/
int preprocess(){

    int warps_per_block = block_dim / THREADS_PER_WARP;

    //shared memory allocate for cur_row[] and reg_flag[]
    preprocess_kernel<<<grid_dim, block_dim, 2 * warps_per_block * sizeof(int)>>>();
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());


    return OK;
}


/*
** function: spmv()
**     sparse matrix-vector multiplication using cvr format
** parameters:
**     floatType *d_y     allocated pointer(device) to store result y
**     floatType *d_x     initialized pointer(device) to store vector x
*/
int spmv(floatType *d_y, floatType *d_x){

    int iteration;
    for(iteration = 0; iteration < n_iterations; iteration++){
        //shared memory allocate for temp result (reducing global write)
        spmv_kernel<<<grid_dim, block_dim, block_dim * sizeof(floatType)>>>(d_y, d_x);
        CHECK(cudaGetLastError());
    }
    CHECK(cudaDeviceSynchronize());

    return OK;
}


__global__ void preprocess_kernel(){
    extern __shared__ int var_ptr[];

    int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    // current warp id in global vision
    int warpID = threadID / THREADS_PER_WARP;
    int laneID = threadID % THREADS_PER_WARP;
    // current warp id in this block
    int warp_offset = threadIdx.x / THREADS_PER_WARP;
    int n_warps = (gridDim.x * blockDim.x + THREADS_PER_WARP - 1) / THREADS_PER_WARP;

    // use register to store csr members for pointer reuse
    csr_t reg_csr;
    reg_csr.ncol = const_csr.ncol;
    reg_csr.nrow = const_csr.nrow;
    reg_csr.nnz = const_csr.nnz;
    reg_csr.val = const_csr.val;
    reg_csr.col_idx = const_csr.col_idx;
    reg_csr.row_ptr = const_csr.row_ptr;

    // use register to store cvr members
    cvr_t reg_cvr;
    reg_cvr.ncol = const_cvr.ncol;
    reg_cvr.nrow = const_cvr.nrow;
    reg_cvr.nnz = const_cvr.nnz;
    reg_cvr.val = const_cvr.val;
    reg_cvr.colidx = const_cvr.colidx;
    reg_cvr.rec = const_cvr.rec;
    reg_cvr.rec_threshold = const_cvr.rec_threshold;
    reg_cvr.threshold_detail = const_cvr.threshold_detail;
    reg_cvr.tail = const_cvr.tail;
    reg_cvr.warp_nnz = const_cvr.warp_nnz;

    // non-zero id
    int warp_start, warp_end;
    int warp_nnz;
    int warp_start_row, warp_end_row;

    // average number of non-zeros in a warp
    int n_warp_nnz = reg_csr.nnz / n_warps;
    // a few warps have one more non-zero to deal with
    int change_warp_nnz = reg_csr.nnz % n_warps;

    // information about row range and non-zeros in this warp
    if(warpID < change_warp_nnz){
        warp_start = warpID * n_warp_nnz + warpID * 1;
        warp_end = (warpID + 1) * n_warp_nnz + (warpID + 1) * 1 - 1;
    }else{
        warp_start = warpID * n_warp_nnz + change_warp_nnz * 1;
        warp_end = (warpID + 1) * n_warp_nnz + change_warp_nnz * 1 - 1;
    }
    warp_nnz = warp_end - warp_start + 1;

    warp_start_row = func_get_row(warp_start, reg_csr);
    warp_end_row = func_get_row(warp_end, reg_csr);

    // upperbound of needed loop iterations to finish preprocess/multiplication
    int ub_steps = (n_warp_nnz + 1 + THREADS_PER_WARP - 1) / THREADS_PER_WARP; 
    int ub_warp_vals = ub_steps * THREADS_PER_WARP;
    int ub_warp_recs = ub_steps + 1;
    // actual number of iterations
    int n_steps = (warp_nnz + THREADS_PER_WARP - 1) / THREADS_PER_WARP;


    // track non-zero/row id in csr
    int valID, rowID, count, candi_valID;
    // record write-back information
    int recID = warpID * ub_warp_recs;
    // reduction and/add/select in a warp
    int count_res, average_res, candidate_res, stealer_res;
    // base address to write in cvr
    int warp_gather_base = warpID * ub_warp_vals;
    __shared__ int *cur_row, *rec_flag;
    bool rec_bit = 0;

    // initialize registers and shared arrays
    if(0 == threadIdx.x){
        cur_row = var_ptr;
        rec_flag = &var_ptr[blockDim.x / THREADS_PER_WARP];
    }
    __syncthreads();

    if(0 == laneID){
        cur_row[warp_offset] = warp_start_row;
        rec_flag[warp_offset] = 0;
        reg_cvr.rec_threshold[warpID] = -1;
        reg_cvr.threshold_detail[warpID] = 0xffffffff;// initially, no threads can write directly to rec.wb in threshold loop
        reg_cvr.warp_nnz[warpID] = warp_nnz;
    }

    // initialize valID, rowID, count for preprocessing
    rowID = atomicAdd(&cur_row[warp_offset], 1);
    // empty rows
    while(rowID < warp_end_row && reg_csr.row_ptr[rowID+1] == reg_csr.row_ptr[rowID]){
        rowID = atomicAdd(&cur_row[warp_offset], 1);
    }

    if(rowID > warp_end_row){
        rowID = -1;
        valID = -1;
        count = 0;
    }else{
        valID = reg_csr.row_ptr[rowID];
        count = reg_csr.row_ptr[rowID+1] - valID;
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
        if(rowID <= warp_end_row){
            reg_cvr.tail[threadID] = rowID;
        }
        if(rowID != -1){
            rowID = threadID;
        }
        if(0 == laneID){
            cur_row[warp_offset] += THREADS_PER_WARP; // ensure IF4 and IF5(ELSE1) will never be executed
            reg_cvr.rec_threshold[warpID] = 0;
        }
    } // END IF1

    // FOR1: preprocessing loop
    for(int i = 0; i <= n_steps; i++){
        // reduce AND
        count_res = func_reduceAnd(count > 0);

        // IF2: if count in some lane(s) = 0, recording and feeding/stealing is needed
        if(0 == count_res){
            if(0 == count){
                // IF3: recording
                if(-1 != valID){
                    reg_cvr.rec[recID].pos = i - 1;
                    rec_bit = 1;
                    reg_cvr.rec[recID].wb[laneID] = rowID;
                    rec_flag[warp_offset] = 1;
                }// END IF3

                // omit empty rows and get a new row
                rowID = atomicAdd(&cur_row[warp_offset], 1);
                while(rowID < warp_end_row && reg_csr.row_ptr[rowID+1] == reg_csr.row_ptr[rowID]){
                    rowID = atomicAdd(&cur_row[warp_offset], 1);
                }
            }

            // WHILE1: feeding/stealing one by one
            while(0 == count_res){

                // IF4: tracker feeding
                if(cur_row[warp_offset] <= warp_end_row+THREADS_PER_WARP){
                    if(0 == count && rowID <= warp_end_row){
                        valID = reg_csr.row_ptr[rowID];
                        count = reg_csr.row_ptr[rowID+1] - valID;
                        if(warp_end_row == rowID){
                            count = warp_end - valID + 1;
                        }
                    }

                    // IF5 & ELSE1
                    if(cur_row[warp_offset] > warp_end_row){
                        bool detail_bit = 0;
                        if(rowID <= warp_end_row){
                            reg_cvr.tail[threadID] = rowID;
                        }
                        if(count == 0 && rowID <= warp_end_row){
                            detail_bit = 1; // these threads can write to rec.wb directly in threshold loop
                        }
                        reg_cvr.threshold_detail[warpID] ^= func_reduceBitOr(detail_bit);
                        rowID = threadID;
                    }
                }// END IF4

                // IF6: set rec_threshold, only executed once
                if(-1 == reg_cvr.rec_threshold[warpID] && cur_row[warp_offset] > warp_end_row){
                    if(0 == laneID){
                        // make sure once IF6 is executed, IF4 will never be executed 
                        cur_row[warp_offset] += THREADS_PER_WARP;
                        reg_cvr.rec_threshold[warpID] = i;
                    }
                }// END IF6

                // re-calculate count_and after possible tracker feeding
                count_res = func_reduceAnd(count > 0);

                // IF7: tracker stealing
                if(0 == count_res && cur_row[warp_offset] > warp_end_row){
                    // calculate average count
                    average_res = func_reduceAvg(count);

                    // find candidate to steal
                    candidate_res = func_reduceSel(count > average_res);

                    // select one lane that need to steal
                    stealer_res = func_reduceSel(count == 0);
    
                    // IF8: if no candidate, padding
                    if(-1 == candidate_res){
                        if(stealer_res == laneID){
                            valID = -1;
                            count = 1;
                        }
                    }else{ // ELSE9, stealing
                        candi_valID = __shfl(valID, candidate_res);
                        if(stealer_res == laneID){
                            rowID = candidate_res + warpID * THREADS_PER_WARP;
                            valID = candi_valID;
                            count = average_res;
                            stealer_res = -1;
                        }
                        if(candidate_res == laneID){
                            rowID = candidate_res + warpID * THREADS_PER_WARP;
                            valID = valID + average_res;
                            count = count - average_res;
                            candidate_res = -1;
                        }
                    } // END IF8
                } // END IF7

                // re-calculate count_and, if = 1, jump out of while loop
                count_res = func_reduceAnd(count > 0);
            } // END WHILE1

            if(1 == rec_flag[warp_offset]){
                reg_cvr.rec[recID].mask = func_reduceBitOr(rec_bit);
                recID++;
                rec_flag[warp_offset] = 0;
                rec_bit = 0;
            }
        } // END IF2 

        // in the last round of for loop, the only thing need to do is recording
        if(i == n_steps){
            continue;
        }
        int addr = warp_gather_base + laneID;
        if(-1 == valID){
            reg_cvr.val[addr] = 0;
            reg_cvr.colidx[addr] = 0;
        }else{
            reg_cvr.val[addr] = reg_csr.val[valID];
            reg_cvr.colidx[addr] = reg_csr.col_idx[valID];
            valID++;
        }
        count--;
        warp_gather_base += THREADS_PER_WARP;
            
    } // END FOR1

}


__global__ void spmv_kernel(floatType * const __restrict__ y, floatType * const __restrict__ x){
    extern __shared__ floatType shared_y[];

    int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    int warpID = threadID / THREADS_PER_WARP;
    int n_warps = (gridDim.x * blockDim.x + THREADS_PER_WARP - 1) / THREADS_PER_WARP;
    int laneID = threadID % THREADS_PER_WARP;
    unsigned lane_mask = 1 << laneID;

    shared_y[threadIdx.x] = 0;

    // use register to store cvr members
    cvr_t reg_cvr;
    reg_cvr.ncol = const_cvr.ncol;
    reg_cvr.nrow = const_cvr.nrow;
    reg_cvr.nnz = const_cvr.nnz;
    reg_cvr.val = const_cvr.val;
    reg_cvr.colidx = const_cvr.colidx;
    reg_cvr.rec = const_cvr.rec;
    reg_cvr.rec_threshold = const_cvr.rec_threshold;
    reg_cvr.threshold_detail = const_cvr.threshold_detail;
    reg_cvr.tail = const_cvr.tail;
    reg_cvr.warp_nnz = const_cvr.warp_nnz;

    // upperbound of needed loop iterations to finish preprocess/multiplication
    int ub_steps = (reg_cvr.nnz / n_warps + 1 + THREADS_PER_WARP - 1) / THREADS_PER_WARP;
    int ub_warp_vals = ub_steps * THREADS_PER_WARP;
    int ub_warp_recs = ub_steps + 1;
    // actual number of iteration loops
    int n_steps = (reg_cvr.warp_nnz[warpID] + THREADS_PER_WARP - 1) / THREADS_PER_WARP;

    floatType temp_result = 0;
    int valID = warpID * ub_warp_vals + laneID;
    int recID = warpID * ub_warp_recs;
    int threshold = reg_cvr.rec_threshold[warpID];
    int x_addr, writeback, writeback2 = -1;
    record_t *rec;
 
    // FOR0
    for(int i = 0; i < n_steps; i++){
 
        x_addr = reg_cvr.colidx[valID];
        // ******** this is the core multiplication!!!!!!!!! ********
        temp_result += reg_cvr.val[valID] * x[x_addr];
            
        rec = &reg_cvr.rec[recID];
        if(rec->pos == i){
            if(0 != (rec->mask & lane_mask)){
                writeback = rec->wb[laneID];
                if((i < threshold) || (i == threshold && ((reg_cvr.threshold_detail[warpID] & lane_mask) == 0))){
                    func_floatTypeAtomicAdd(&y[writeback], temp_result);
                }else{
                    func_floatTypeAtomicAdd(&shared_y[writeback%blockDim.x], temp_result);
 
                }
                temp_result = 0;
            }
            recID++;
        }

        valID += THREADS_PER_WARP;
    } // END FOR0

    writeback2 = reg_cvr.tail[threadID];
    
    if(writeback2 != -1){
        func_floatTypeAtomicAdd(&y[writeback2], shared_y[threadIdx.x]);
 
    }

}

