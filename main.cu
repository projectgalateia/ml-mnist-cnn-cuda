#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "mnist.h"

#include <cuda.h>
#include <cstdio>

static mnist_data *train_set, *test_set;
static unsigned int train_cnt, test_cnt;

static void learn();
static void clear();
static unsigned int classify(double data[28][28]);

static void loaddata()
{
	mnist_load("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte",
		&train_set, &train_cnt);
	mnist_load("data/t10k-images-idx3-ubyte", "data/t10k-labels-idx1-ubyte",
		&test_set, &test_cnt);
}

static void test()
{
	int error = 0;

	for (int i = 0; i < test_cnt; ++i) {
		if (classify(test_set[i].data) != test_set[i].label) {
			++error;
		}
	}

	fprintf(stdout, "Error Rate: %.2lf%%\n",
		double(error) / double(test_cnt) * 100.0);
}

int main(int argc, const  char **argv)
{
	srand(time(NULL));

	if (cuInit(0) != CUDA_SUCCESS) {
		fprintf(stderr, "cuInit failed\n");
		return 1;
	}

	loaddata();
	learn();
	test();
	clear();

	return 0;
}

///////////////////////////////////////////////////////////////////////////////

#include <cstdlib>
#include <vector>
#include <memory>
#include <cublas_v2.h>

const static float dt = 1.0E-01f;
const static float threshold = 1.0E-02f;

struct Layer {
	float *output;
	float *preact;

	float *bias;
	float *weight;

	float *d_output;
	float *d_preact;
	float *d_weight;

	const int M, N, O;

	Layer(int M, int N, int O)
		: M(M), N(N), O(O)
	{
		float h_bias[N];
		float h_weight[N][M];

		output = NULL;
		preact = NULL;
		bias   = NULL;
		weight = NULL;

		for (int i = 0; i < N; ++i) {
			h_bias[i] = 0.5f - float(rand()) / float(RAND_MAX);
			/*h_bias[i] = 0.0f;*/

			for (int j = 0; j < M; ++j) {
				h_weight[i][j] = 0.5f - float(rand()) / float(RAND_MAX);
				/*h_weight[i][j] = 0.05f;*/
			}
		}

		cudaMalloc(&output, sizeof(float) * O);
		cudaMalloc(&preact, sizeof(float) * O);

		cudaMalloc(&bias, sizeof(float) * N);

		cudaMalloc(&weight, sizeof(float) * M * N);

		cudaMalloc(&d_output, sizeof(float) * O);
		cudaMalloc(&d_preact, sizeof(float) * O);
		cudaMalloc(&d_weight, sizeof(float) * M * N);

		cudaMemcpy(bias, h_bias, sizeof(float) * N, cudaMemcpyHostToDevice);

		cudaMemcpy(weight, h_weight, sizeof(float) * M * N, cudaMemcpyHostToDevice);
	}

	~Layer()
	{
		cudaFree(output);
		cudaFree(preact);

		cudaFree(bias);

		cudaFree(weight);

		cudaFree(d_output);
		cudaFree(d_preact);
		cudaFree(d_weight);
	}

	void setOutput(float *data)
	{
		cudaMemcpy(output, data, sizeof(float) * O, cudaMemcpyHostToDevice);
	}

	void clear()
	{
		cudaMemset(output, 0x00, sizeof(float) * O);
		cudaMemset(preact, 0x00, sizeof(float) * O);
	}

	void bp_clear()
	{
		cudaMemset(d_output, 0x00, sizeof(float) * O);
		cudaMemset(d_preact, 0x00, sizeof(float) * O);
		cudaMemset(d_weight, 0x00, sizeof(float) * M * N);
	}
};

static cublasHandle_t blas;
static Layer l_input(0, 0, 28*28), l_c1(5*5, 6, 24*24*6), l_s1(4*4, 1, 6*6*6), l_f(6*6*6, 10, 10);

__device__ float step_function(float v)
{
	return 1 / (1 + exp(-v));
}

__global__ void apply_step_function(float *input, float *output, const int N)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		output[idx] = step_function(input[idx]);
	}
}

__global__ void makeError(float *err, float *output, unsigned int Y, const int N)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		err[idx] = ((Y == idx ? 1.0f : 0.0f) - output[idx]);
	}
}

__global__ void apply_grad(float *output, float *grad, const int N)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		output[idx] += dt * grad[idx];
	}
}

__global__ void preact_c1(float input[28][28], float preact[6][24][24], float weight[6][5][5])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 5*5*6*24*24;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 5);
		const int i2 = ((idx /= 5	) % 5);
		const int i3 = ((idx /= 5	) % 6);
		const int i4 = ((idx /= 6	) % 24);
		const int i5 = ((idx /= 24	) % 24);

		atomicAdd(&preact[i3][i4][i5], weight[i3][i1][i2] * input[i4 + i1][i5 + i2]);
	}
}

__global__ void bias_c1(float preact[6][24][24], float bias[6])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 6*24*24;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 6);
		const int i2 = ((idx /= 6	) % 24);
		const int i3 = ((idx /= 24	) % 24);

		preact[i1][i2][i3] += bias[i1];
	}
}

__global__ void preact_s1(float input[6][24][24], float preact[6][6][6], float weight[1][4][4])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 4*4*6*6*6;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 4);
		const int i2 = ((idx /= 4	) % 4);
		const int i3 = ((idx /= 4	) % 6);
		const int i4 = ((idx /= 6	) % 6);
		const int i5 = ((idx /= 6	) % 6);

		atomicAdd(&preact[i3][i4][i5], weight[0][i1][i2] * input[i3][i4 * 4 + i1][i5 * 4 + i2]);
	}
}

__global__ void bias_s1(float preact[6][6][6], float bias[1])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 6*6*6;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 6);
		const int i2 = ((idx /= 6	) % 6);
		const int i3 = ((idx /= 6	) % 6);

		preact[i1][i2][i3] += bias[0];
	}
}

__global__ void preact_f(float input[6][6][6], float preact[10], float weight[10][6][6][6])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 10*6*6*6;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 10);
		const int i2 = ((idx /= 10	) % 6);
		const int i3 = ((idx /= 6	) % 6);
		const int i4 = ((idx /= 6	) % 6);

		atomicAdd(&preact[i1], weight[i1][i2][i3][i4] * input[i2][i3][i4]);
	}
}

__global__ void bias_f(float preact[10], float bias[10])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 10;

	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		preact[idx] += bias[idx];
	}
}

__global__ void bp_weight_f(float d_weight[10][6][6][6], float d_preact[10], float p_output[6][6][6])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 10*6*6*6;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 10);
		const int i2 = ((idx /= 10	) % 6);
		const int i3 = ((idx /= 6	) % 6);
		const int i4 = ((idx /= 6	) % 6);

		d_weight[i1][i2][i3][i4] = d_preact[i1] * p_output[i2][i3][i4];
	}
}

__global__ void bp_bias_f(float bias[10], float d_preact[10])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 10;

	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		bias[idx] += dt * d_preact[idx];
	}
}

__global__ void bp_output_s1(float d_output[6][6][6], float n_weight[10][6][6][6], float nd_preact[10])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 10*6*6*6;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 10);
		const int i2 = ((idx /= 10	) % 6);
		const int i3 = ((idx /= 6	) % 6);
		const int i4 = ((idx /= 6	) % 6);

		atomicAdd(&d_output[i2][i3][i4], n_weight[i1][i2][i3][i4] * nd_preact[i1]);
	}
}

__global__ void bp_preact_s1(float d_preact[6][6][6], float d_output[6][6][6], float preact[6][6][6])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 6*6*6;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 6);
		const int i2 = ((idx /= 6	) % 6);
		const int i3 = ((idx /= 6	) % 6);

		const float o = step_function(preact[i1][i2][i3]);

		d_preact[i1][i2][i3] = d_output[i1][i2][i3] * o * (1 - o);
	}
}

__global__ void bp_weight_s1(float d_weight[1][4][4], float d_preact[6][6][6], float p_output[6][24][24])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 1*4*4*6*6*6;
	const float d = pow(6.0f, 3.0f);

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 1);
		const int i2 = ((idx /= 1	) % 4);
		const int i3 = ((idx /= 4	) % 4);
		const int i4 = ((idx /= 4	) % 6);
		const int i5 = ((idx /= 6	) % 6);
		const int i6 = ((idx /= 6	) % 6);

		atomicAdd(&d_weight[i1][i2][i3], d_preact[i4][i5][i6] * p_output[i4][i5 * 4 + i2][i6 * 4 + i3]);
	}
}

__global__ void bp_bias_s1(float bias[1], float d_preact[6][6][6])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 6*6*6;
	const float d = pow(6.0f, 3.0f);

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 6);
		const int i2 = ((idx /= 6	) % 6);
		const int i3 = ((idx /= 6	) % 6);

		atomicAdd(&bias[0], dt * d_preact[i1][i2][i3] / d);
	}
}

__global__ void bp_output_c1(float d_output[6][24][24], float n_weight[1][4][4], float nd_preact[6][6][6])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 1*4*4*6*6*6;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 1);
		const int i2 = ((idx /= 1	) % 4);
		const int i3 = ((idx /= 4	) % 4);
		const int i4 = ((idx /= 4	) % 6);
		const int i5 = ((idx /= 6	) % 6);
		const int i6 = ((idx /= 6	) % 6);

		atomicAdd(&d_output[i4][i5 * 4 + i2][i6 * 4 + i3], n_weight[i1][i2][i3] * nd_preact[i4][i5][i6]);
	}
}

__global__ void bp_preact_c1(float d_preact[6][24][24], float d_output[6][24][24], float preact[6][24][24])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 6*24*24;

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 6);
		const int i2 = ((idx /= 6	) % 24);
		const int i3 = ((idx /= 24	) % 24);

		const float o = step_function(preact[i1][i2][i3]);

		d_preact[i1][i2][i3] = d_output[i1][i2][i3] * o * (1 - o);
	}
}

__global__ void bp_weight_c1(float d_weight[6][5][5], float d_preact[6][24][24], float p_output[28][28])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 6*5*5*24*24;
	const float d = pow(24.0f, 2.0f);

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 6);
		const int i2 = ((idx /= 6	) % 5);
		const int i3 = ((idx /= 5	) % 5);
		const int i4 = ((idx /= 5	) % 24);
		const int i5 = ((idx /= 24	) % 24);

		atomicAdd(&d_weight[i1][i2][i3], d_preact[i1][i4][i5] * p_output[i4 + i2][i5 + i3] / d);
	}
}

__global__ void bp_bias_c1(float bias[6], float d_preact[6][24][24])
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 6*24*24;
	const float d = pow(24.0f, 2.0f);

	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) {
		int idx = n;
		const int i1 = ((idx /= 1	) % 6);
		const int i2 = ((idx /= 6	) % 24);
		const int i3 = ((idx /= 24	) % 24);

		atomicAdd(&bias[i1], dt * d_preact[i1][i2][i3] / d);
	}
}

static void propagate(double data[28][28])
{
	float input[28][28];

	for (int i = 0; i < 28; ++i) {
		for (int j = 0; j < 28; ++j) {
			input[i][j] = data[i][j];
		}
	}

	l_input.clear();
	l_c1.clear();
	l_s1.clear();
	l_f.clear();

	l_input.setOutput((float *)input);
	
	preact_c1<<<64, 64>>>((float (*)[28])l_input.output, (float (*)[24][24])l_c1.preact, (float (*)[5][5])l_c1.weight);
	bias_c1<<<64, 64>>>((float (*)[24][24])l_c1.preact, l_c1.bias);
	apply_step_function<<<64, 64>>>(l_c1.preact, l_c1.output, l_c1.O);

	preact_s1<<<64, 64>>>((float (*)[24][24])l_c1.output, (float (*)[6][6])l_s1.preact, (float (*)[4][4])l_s1.weight);
	bias_s1<<<64, 64>>>((float (*)[6][6])l_s1.preact, l_s1.bias);
	apply_step_function<<<64, 64>>>(l_s1.preact, l_s1.output, l_s1.O);

	preact_f<<<64, 64>>>((float (*)[6][6])l_s1.output, l_f.preact, (float (*)[6][6][6])l_f.weight);
	bias_f<<<64, 64>>>(l_f.preact, l_f.bias);
	apply_step_function<<<64, 64>>>(l_f.preact, l_f.output, l_f.O);
}

static void learn()
{
	cublasCreate(&blas);

	float err;
	int iter = 240;

	while (iter < 0 || iter-- > 0) {
		err = 0.0f;
		
		for (int i = 0; i < train_cnt; ++i) {
			float tmp;

			propagate(train_set[i].data);

			makeError<<<10, 1>>>(l_f.d_preact, l_f.output, train_set[i].label, 10);

			cublasSnrm2(blas, 10, l_f.d_preact, 1, &tmp);

			err += tmp;
		}

		err /= train_cnt;
		fprintf(stdout, "error: %e\n", err);

		if (err < threshold)
			break;

		for (int i = 0; i < train_cnt; ++i) {
			propagate(train_set[i].data);

			l_f.bp_clear();
			l_s1.bp_clear();
			l_c1.bp_clear();

			makeError<<<10, 1>>>(l_f.d_preact, l_f.output, train_set[i].label, 10);

			bp_weight_f<<<64, 64>>>((float (*)[6][6][6])l_f.d_weight, l_f.d_preact, (float (*)[6][6])l_s1.output);
			bp_bias_f<<<64, 64>>>(l_f.bias, l_f.d_preact);

			bp_output_s1<<<64, 64>>>((float (*)[6][6])l_s1.d_output, (float (*)[6][6][6])l_f.weight, l_f.d_preact);
			bp_preact_s1<<<64, 64>>>((float (*)[6][6])l_s1.d_preact, (float (*)[6][6])l_s1.d_output, (float (*)[6][6])l_s1.preact);
			bp_weight_s1<<<64, 64>>>((float (*)[4][4])l_s1.d_weight, (float (*)[6][6])l_s1.d_preact, (float (*)[24][24])l_c1.output);
			bp_bias_s1<<<64, 64>>>(l_s1.bias, (float (*)[6][6])l_s1.d_preact);

			bp_output_c1<<<64, 64>>>((float (*)[24][24])l_c1.d_output, (float (*)[4][4])l_s1.weight, (float (*)[6][6])l_s1.d_preact);
			bp_preact_c1<<<64, 64>>>((float (*)[24][24])l_c1.d_preact, (float (*)[24][24])l_c1.d_output, (float (*)[24][24])l_c1.preact);
			bp_weight_c1<<<64, 64>>>((float (*)[5][5])l_c1.d_weight, (float (*)[24][24])l_c1.d_preact, (float (*)[28])l_input.output);
			bp_bias_c1<<<64, 64>>>(l_c1.bias, (float (*)[24][24])l_c1.d_preact);


			apply_grad<<<64, 64>>>(l_f.weight, l_f.d_weight, l_f.M * l_f.N);
			apply_grad<<<64, 64>>>(l_s1.weight, l_s1.d_weight, l_s1.M * l_s1.N);
			apply_grad<<<64, 64>>>(l_c1.weight, l_c1.d_weight, l_c1.M * l_c1.N);
		}
	}
}

static unsigned int classify(double data[28][28])
{
	float res[10];

	propagate(data);

	unsigned int max = 0;

	cudaMemcpy(res, l_f.output, sizeof(float) * 10, cudaMemcpyDeviceToHost);

	for (int i = 1; i < 10; ++i) {
		if (res[max] < res[i]) {
			max = i;
		}
	}

	return max;
}

static void clear()
{
}

