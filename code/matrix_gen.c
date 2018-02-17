//matrix_gen.c
//0-based matrix market format generator
#include<stdio.h>
#include<stdlib.h>
#include<time.h>

typedef struct{
	int x, y;
	float val;
}triple_t;

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

int func_fliter(triple_t *t, int *n){
	int i, count = *n;
	int gap = 0;
	for(i = 0; i + gap < count; ){
		if(t[i].x == t[i+gap+1].x && t[i].y == t[i+gap+1].y){
			gap++;
			(*n)--;
		}else{
			t[i+1].x = t[i+gap+1].x;
			t[i+1].y = t[i+gap+1].y;
			t[i+1].val = t[i+gap+1].val;
			i++;
		}
	}
	return 0;
}

int main(int argc, char **argv){
	int row = 10, col = 10, nnz = 20;
	if(argc < 2){
		printf("ERROR: filename is needed in parameter list\n");
		return 0;
	}else if(5 == argc){
		row = atoi(argv[2]);
		col = atoi(argv[3]);
		nnz = atoi(argv[4]);
	}

	char *filename = argv[1];
	FILE *fp = fopen(filename, "w");
	fprintf(fp, "%%%%MatrixMarket matrix coordinate real general\n");
	fprintf(fp, "%%============\n%%unimportant comment\n%%============\n");

	triple_t *triple = (triple_t *)malloc(nnz * sizeof(triple_t));
	int i;
	srand((unsigned)time(NULL));
	for(i = 0; i < nnz; i++){
		triple[i].x = rand() % row;
		triple[i].y = rand() % col;
		triple[i].val = rand() * 1.0 / RAND_MAX;
	}

	qsort(triple, nnz, sizeof(triple_t), func_cmp);
	func_fliter(triple, &nnz);

	fprintf(fp, "  %d  %d  %d\n", row, col, nnz);
	for(i = 0; i < nnz; i++){
		fprintf(fp, " %d %d %f  \n", triple[i].x, triple[i].y, triple[i].val);
	}

	fclose(fp);

	return 0;
}
