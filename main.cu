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
const static float threshold = 1.0E-01f;

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

		cudaMalloc(&d_output, sizeof(float) * N);
		cudaMalloc(&d_preact, sizeof(float) * N);
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
};

static cublasHandle_t blas;

__device__ float step_function(float v)
{
	return 1 / (1 + exp(-v));
}

// 28x28 -> 24x24x6
__global__ void preact_c1(float *data, float *weight, float *result)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;
	const int M = 24*24;
	const int N = M*6;

	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		const int o = idx / M;
		const int y = (idx % M) / 24;
		const int x = (idx % M) % 24;

		result[idx] = 0;

		for (int i = 0; i < 5; ++i) {
			for (int j = 0; j < 5; ++j) {
				const int p = (y + i) * 28 + (x + j);
				const int w = o * 25 + i * 5 + j;

				result[idx] += weight[w] * data[p];
			}
		}
	}
}

__global__ void output_c1(float *data, float *bias, float *result)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;
	const int M = 24*24;
	const int N = M*6;

	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		const int o = idx / M;

		result[idx] = step_function(data[idx] + bias[o]);
	}
}

// 24x24x6 -> 6x6x6
__global__ void preact_s1(float *data, float *weight, float *result)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;
	const int M = 6*6;
	const int N = M*6;

	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		const int o = idx / M;
		const int y = (idx % M) / 6;
		const int x = (idx % M) % 6;

		result[idx] = 0;

		for (int i = 0; i < 4; ++i) {
			for (int j = 0; j < 4; ++j) {
				const int p = o*24*24 + (y * 4 + i) * 24 + (x * 4 + j);
				const int w = i * 4 + j;

				result[idx] += weight[w] * data[p];
			}
		}
	}
}

__global__ void output_s1(float *data, float *bias, float *result)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;
	const int M = 6*6;
	const int N = M*6;

	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		result[idx] = step_function(data[idx] + bias[0]);
	}
}

// 6x6x6 -> 10
__global__ void preact_f(float *data, float *weight, float *result) {
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;
	const int N = 10;

	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		result[idx] = 0;

		for (int i = 0; i < 6*6*6; ++i) {
			result[idx] += weight[idx * 6*6*6 + i] * data[i];
		}
	}
}

__global__ void output_f(float *data, float *bias, float *result) {
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;
	const int N = 10;

	for (int idx = N * pos / size; idx < N * (pos+1) / size; ++idx) {
		result[idx] = step_function(data[idx] + bias[idx]);
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

__global__ void bp_bias(float *bias, float *d_preact, const int L)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	for (int idx = L * pos / size; idx < L * (pos+1) / size; ++idx) {
		bias[idx] += dt * d_preact[idx];
	}
}

__global__ void bp_weight(float *d_weight, float *d_preact, float *p_output, const int M, const int N)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int L = M * N;

	for (int idx = L * pos / size; idx < L * (pos+1) / size; ++idx) {
		const int a = idx / M;
		const int b = idx % M;

		d_weight[idx] = d_preact[a] * p_output[b];
	}
}

__global__ void bp_preact(float *d_preact, float *d_output, float *preact, const int L)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	for (int idx = L * pos / size; idx < L * (pos+1) / size; ++idx) {
		const float o = step_function(preact[idx]);

		d_preact[idx] = o * (1 - o) * d_output[idx];
	}
}

__global__ void bp_output_s1(float *d_output, float *n_weight, float *nd_preact)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int L = 6*6*6;

	for (int idx = L * pos / size; idx < L * (pos+1) / size; ++idx) {
		d_output[idx] = 0.0f;

		for (int i = 0; i < 10; ++i) {
			const int w = idx * 10 + i;

			d_output[idx] += n_weight[w] * nd_preact[i];
		}

		d_output[idx] /= 10.0f;
	}
}

__global__ void bp_weight_s1(float *d_weight, float *d_preact, float *p_output)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int N = 4*4;
	const int K = 6*6*4*4;

	for (int idx = K * pos / size; idx < K * (pos+1) / size; ++idx) {
		float res = 0.0f;

		const int KN = idx / N;
		const int KP = idx % N;

		const int n_y = KN / 6;
		const int n_x = KN % 6;

		const int p_y = KP / 4;
		const int p_x = KP % 4;

		const int nidx = n_y * 4 + n_x;

		for (int i = 0; i < 6; ++i) {
			const int pidx = i * 6*6 + (n_y * 4 + p_y) * 24 + (n_x * 4 + p_x);

			res += d_preact[0] * p_output[pidx];
		}

		atomicAdd(&d_weight[nidx], res / (6*6));
	}
}

__global__ void bp_outupt_c1(float *d_output, float *n_weight, float *nd_preact)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int L = 5*5*24*24;

	const int N = 5*5;
	const int M = 24*24;

	for (int idx = L * pos / size; idx < L * (pos+1) / size; ++idx) {
		const int KN = idx / M;
		const int KP = idx % M;

		const int n_y = KN / 5;
		const int n_x = KN % 5;
		const int p_y = KP / 24;
		const int p_x = KP % 24;

		float res = 0.0f;

		for (int i  = 0; i < 6; ++i) {
		}
	}
}

__global__ void update_var(float *target, float *grad, const int L)
{
	const int pos = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	for (int idx = L * pos / size; idx < L * (pos+1) / size; ++idx) {
		target[idx] += dt * grad[idx];
	}
}

static Layer l_input(0, 0, 28*28), l_c1(5*5, 6, 24*24*6), l_s1(4*4, 1, 6*6*6), l_f(6*6*6, 10, 10);

static void propagate(double data[28][28])
{
	float input[28*28];

	for (int i = 0; i < 28; ++i) {
		for (int j = 0; j < 28; ++j) {
			const int idx = i * 28 + j;

			input[idx] = data[i][j];
		}
	}

	l_input.setOutput(input);

	preact_c1<<<64,32>>>(l_input.output, l_c1.weight, l_c1.preact);
	output_c1<<<64,32>>>(l_c1.preact, l_c1.bias, l_c1.output);

	preact_s1<<<64,32>>>(l_c1.output, l_s1.weight, l_s1.preact);
	output_s1<<<64,32>>>(l_s1.preact, l_s1.bias, l_s1.output);

	preact_f<<<64,32>>>(l_s1.output, l_f.weight, l_f.preact);
	output_f<<<64,32>>>(l_f.preact, l_f.bias, l_f.output);
}

static void learn()
{
	cublasCreate(&blas);

	float err;

	while (true) {
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

			makeError<<<10, 1>>>(l_f.d_preact, l_f.output, train_set[i].label, 10);

			bp_weight<<<64, 32>>>(l_f.d_weight, l_f.d_preact, l_s1.output, l_f.M, l_f.N);
			bp_bias<<<64, 32>>>(l_f.bias, l_f.d_preact, l_f.N);


			bp_output_s1<<<64, 32>>>(l_s1.d_output, l_f.weight, l_f.d_preact);
			bp_preact<<<64, 32>>>(l_s1.d_preact, l_s1.d_output, l_s1.preact, l_s1.N);
			cudaMemset(l_s1.d_weight, 0x0, sizeof(float) * l_s1.M * l_s1.N); bp_weight_s1<<<64, 32>>>(l_s1.d_weight, l_s1.d_preact, l_c1.output);
			bp_bias<<<64, 32>>>(l_s1.bias, l_s1.d_preact, l_f.N);

			/*float test[l_s1.N][l_s1.M];*/

			/*cudaMemcpy(test, l_s1.d_weight, sizeof(test), cudaMemcpyDeviceToHost);*/

			/*for (int j = 0; j < l_s1.N; ++j) {*/
				/*for (int k = 0; k < l_s1.M; ++k) {*/
					/*printf("|%+2d", (int)(test[j][k] * 10.0f));*/
				/*}*/
				/*printf("|\n");*/
			/*}*/
			/*printf("\n");*/
			/*printf("%d %d\n", l_s1.M, l_s1.N);*/
			/*exit(0);*/

			update_var<<<64, 32>>>(l_f.weight, l_f.d_weight, l_f.M * l_f.N);
			update_var<<<64, 32>>>(l_s1.weight, l_s1.d_weight, l_s1.M * l_s1.N);

			/*for (int j = layers.size() - 1; j > 0; --j) {*/
				/*backpropagate(layers[j - 1], layers[j]);*/
			/*}*/
			/*exit(0);*/
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

