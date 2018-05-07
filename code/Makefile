
#cvr_spmv_gpu : cvr_spmv_gpu.cu
#	nvcc cvr_spmv_gpu.cu -o cvr_spmv_gpu -O3 -m64 -arch=sm_52 -Xcompiler -fopenmp -lgomp

cvr_spmv_gpu_auto : cvr_spmv_gpu_auto.cu
	nvcc cvr_spmv_gpu_auto.cu -o cvr_spmv_gpu_auto -O3 -m64 -arch=sm_52 -Xcompiler -fopenmp -lgomp

debug :
	nvcc cvr_spmv_gpu.cu -o cvr_spmv_gpu -O3 -m64 -arch=sm_52 -Xcompiler -fopenmp -lgomp -g -G
#-gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_52,code=compute_52	

keep :
	cp ../cvr_spmv_gpu.cu ./
	nvcc cvr_spmv_gpu.cu -o cvr_spmv_gpu -O3 -m64 -arch=sm_52 -Xcompiler -fopenmp -lgomp -keep
	cp cvr_spmv_gpu.ptx ../

reg :
	nvcc cvr_spmv_gpu.cu -o cvr_spmv_gpu -O3 -m64 -arch=sm_52 -Xcompiler -fopenmp -lgomp -Xptxas -v

