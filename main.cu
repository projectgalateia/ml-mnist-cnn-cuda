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
const static float threshold = 1.3E-01f;

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
static Layer l_input(0, 0, 28*28), l_c1(5*5, 6, 6*24*24), l_s1(2*2, 1, 6*12*12),
	     l_c2(5*5, 6, 36*8*8), l_s2(2*2, 1, 36*4*4), l_f(36*4*4, 10, 10);

#define FOR6(I1, I2, I3, I4, I5, I6, M) \
	const int pos = blockIdx.x * blockDim.x + threadIdx.x; \
	const int size = blockDim.x * gridDim.x; \
	\
	const int N = I1*I2*I3*I4*I5*I6; \
	\
	for (int n = N * pos / size; n < N * (pos+1) / size; ++n) { \
		int idx = n; \
		const int i1 = ((idx /= 1	) % I1); \
		const int i2 = ((idx /= I1	) % I2); \
		const int i3 = ((idx /= I2	) % I3); \
		const int i4 = ((idx /= I3	) % I4); \
		const int i5 = ((idx /= I4	) % I5); \
		const int i6 = ((idx /= I5	) % I6); \
		\
		M; \
	} \

#define FOR5(I1, I2, I3, I4, I5, M)	FOR6(I1, I2, I3, I4, I5, 1, M)
#define FOR4(I1, I2, I3, I4, M)		FOR6(I1, I2, I3, I4, 1, 1, M)
#define FOR3(I1, I2, I3, M)		FOR6(I1, I2, I3, 1, 1, 1, M)
#define FOR2(I1, I2, M)			FOR6(I1, I2, 1, 1, 1, 1, M)
#define FOR1(I1, M)			FOR6(I1, 1, 1, 1, 1, 1, M)

#define CAST0(I, v)	((float (*)[I])v)
#define CAST1(I, v)	((float (*)[I][I])v)
#define CAST2(V, I, v)	((float (*)[V][I][I])v)

#define PREACT_C(I1, I2, I3, I4, input, preact, weight) \
	preact_c<I1, I2, I3, I4><<<64, 64>>>((float (*)[I3+I1-1][I3+I1-1])input, (float (*)[I3][I3])preact, (float (*)[I1][I1])weight)

#define BIAS_C(I1, I2, preact, bias) \
	bias_c<I1, I2><<<64, 64>>>((float (*)[I2][I2])preact, bias);

#define PREACT_S(I1, I2, I3, input, preact, weight) \
	preact_s<I1, I2, I3><<<64, 64>>>((float (*)[I1*I3][I1*I3])input, (float (*)[I3][I3])preact, (float (*)[I1])weight)

#define BIAS_S(I1, I2, preact, bias) \
	bias_s<I1, I2><<<64, 64>>>((float (*)[I2][I2])preact, bias)

#define BP_OUTPUT_S(I1, I2, I3, I4, d_output, n_weight, nd_preact) \
	bp_output_s<I1, I2, I3, I4><<<64, 64>>>(CAST1(I2+I3-1, d_output), CAST1(I2, n_weight), CAST1(I3, nd_preact))

#define BP_PREACT_S(I1, I2, d_preact, d_output, preact) \
	bp_preact_s<I1, I2><<<64, 64>>>((float (*)[I2][I2])d_preact, (float (*)[I2][I2])d_output, (float (*)[I2][I2])preact)

#define BP_WEIGHT_S(I1, I2, I3, d_weight, d_preact, p_output) \
	bp_weight_s<I1, I2, I3><<<64, 64>>>(CAST0(I1, d_preact), CAST1(I3, d_preact), CAST1(I1*I3, p_output))

#define BP_BIAS_S(I1, I2, bias, d_preact) \
	bp_bias_s<I1, I2><<<64, 64>>>(bias, CAST1(I2, d_preact))


#define BP_OUTPUT_C(I1, I2, I3, d_output, n_weight, nd_preact) \
	bp_output_c<I1, I2, I3><<<64, 64>>>(CAST1(I1*I3, d_output), CAST0(I1, n_weight), CAST1(I3, nd_preact))

#define BP_PREACT_C(I1, I2, d_preact, d_output, preact) \
	bp_preact_c<I1, I2><<<64, 64>>>(CAST1(I2, d_preact), CAST1(I2, d_output), CAST1(I2, preact))

#define BP_WEIGHT_C(I1, I2, I3, I4, d_weight, d_preact, p_output) \
	bp_weight_c<I1, I2, I3, I4><<<64, 64>>>(CAST1(I2, d_weight), CAST1(I3, d_preact), CAST1(I3 + I2 - 1, p_output))

#define BP_BIAS_C(I1, I2, I3, bias, d_preact) \
	bp_bias_c<I1, I2, I3><<<64, 64>>>(bias, CAST1(I2, d_preact))

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

template<int I1, int I2, int I3, int I4>
__global__ void preact_c(float input[I4][I3+I1-1][I3+I1-1], float preact[I2*I4][I3][I3], float weight[I2][I1][I1])
{
	FOR6(I1, I1, I2, I3, I3, I4, atomicAdd(&preact[i6 * I2 + i3][i4][i5], weight[i3][i1][i2] * input[i6][i4 + i1][i5 + i2]));
}

template<int I1, int I2>
__global__ void bias_c(float preact[I1][I2][I2], float bias[I1])
{
	FOR3(I1, I2, I2, preact[i1][i2][i3] += bias[i1]);
}

template<int I1, int I2, int I3>
__global__ void preact_s(float input[I2][I1*I3][I1*I3], float preact[I2][I3][I3], float weight[I1][I1])
{
	FOR5(I1, I1, I2, I3, I3, atomicAdd(&preact[i3][i4][i5], weight[i1][i2] * input[i3][i4 * I1 + i1][i5 * I1 + i2]));
}

template<int I1, int I2>
__global__ void bias_s(float preact[I1][I2][I2], float bias[1])
{
	FOR3(I1, I2, I2, preact[i1][i2][i3] += bias[0]);
}

__global__ void preact_f(float input[36][4][4], float preact[10], float weight[10][36][4][4])
{
	FOR4(10, 36, 4, 4, atomicAdd(&preact[i1], weight[i1][i2][i3][i4] * input[i2][i3][i4]));
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

__global__ void bp_weight_f(float d_weight[10][36][4][4], float d_preact[10], float p_output[36][4][4])
{
	FOR4(10, 36, 4, 4, d_weight[i1][i2][i3][i4] = d_preact[i1] * p_output[i2][i3][i4]);
}

__global__ void bp_bias_f(float bias[10], float d_preact[10])
{
	FOR1(10, bias[i1] += dt * d_preact[i1]);
}

__global__ void bp_output_s2(float d_output[36][4][4], float n_weight[10][36][4][4], float nd_preact[10])
{
	FOR4(10, 36, 4, 4, atomicAdd(&d_output[i2][i3][i4], n_weight[i1][i2][i3][i4] * nd_preact[i1]));
}

template<int I1, int I2, int I3, int I4>
__global__ void bp_output_s(float d_output[I1][I2+I3-1][I2+I3-1], float n_weight[I4][I2][I2], float nd_preact[I1*I4][I3][I3])
{
	FOR6(I1, I2, I2, I3, I3, I4,
		atomicAdd(&d_output[i1][i2+i4-1][i3+i5-1], n_weight[i6][i2][i3] * nd_preact[i6 * I1 + i1][i4][i5])
	);
}

template<int I1, int I2>
__global__ void bp_preact_s(float d_preact[I1][I2][I2], float d_output[I1][I2][I2], float preact[I1][I2][I2])
{
	FOR3(I1, I2, I2, 
		const float o = step_function(preact[i1][i2][i3]);

		d_preact[i1][i2][i3] = d_output[i1][i2][i3] * o * (1 - o);
	);
}

template<int I1, int I2, int I3>
__global__ void bp_weight_s(float d_weight[I1][I1], float d_preact[I2][I3][I3], float p_output[I2][I1*I3][I1*I3])
{
	const float d = I2 * I3 * I3;

	FOR5(I1, I1, I2, I3, I3,
		atomicAdd(&d_weight[i1][i2], d_preact[i3][i4][i5] * p_output[i3][i4 * I1 + i1][i5 * I1 + i2]) / d);
}

template<int I1, int I2>
__global__ void bp_bias_s(float bias[1], float d_preact[I1][I2][I2])
{
	const float d = I1 * I2 * I2;

	FOR3(I1, I2, I2, atomicAdd(&bias[0], dt * d_preact[i1][i2][i3] / d));
}

template<int I1, int I2, int I3>
__global__ void bp_output_c(float d_output[I2][I1*I3][I1*I3], float n_weight[I1][I1], float nd_preact[I2][I3][I3])
{
	FOR5(I1, I1, I2, I3, I3,
		atomicAdd(&d_output[i3][i4 * I1 + i1][i5 * I1 + i2], n_weight[i1][i2] * nd_preact[i3][i4][i5]));
}

template<int I1, int I2>
__global__ void bp_preact_c(float d_preact[I1][I2][I2], float d_output[I1][I2][I2], float preact[I1][I2][I2])
{
	FOR3(I1, I2, I2, 
		const float o = step_function(preact[i1][i2][i3]);

		d_preact[i1][i2][i3] = d_output[i1][i2][i3] * o * (1 - o);
	);
}

template<int I1, int I2, int I3, int I4>
__global__ void bp_weight_c(float d_weight[I1][I2][I2], float d_preact[I1*I4][I3][I3],
	float p_output[I4][I3 + I2 - 1][I3 + I2 - 1])
{
	const float d = I3 * I3 * I4;

	FOR6(I1, I2, I2, I3, I3, I4,
		atomicAdd(&d_weight[i1][i2][i3], d_preact[i6 * I1 + i1][i4][i5] * p_output[i6][i4 + i2][i5 + i3] / d));
}

template<int I1, int I2, int I3>
__global__ void bp_bias_c(float bias[I1], float d_preact[I1 * I3][I2][I2])
{
	const float d = I2 * I2 * I3;

	FOR4(I1, I2, I2, I3, atomicAdd(&bias[i1], dt * d_preact[i1 * I3 + i4][i2][i3] / d));
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
	l_c2.clear();
	l_s2.clear();
	l_f.clear();

	l_input.setOutput((float *)input);
	
	PREACT_C(5, 6, 24, 1, l_input.output, l_c1.preact, l_c1.weight);
	BIAS_C(6, 24, l_c1.preact, l_c1.bias);
	apply_step_function<<<64, 64>>>(l_c1.preact, l_c1.output, l_c1.O);

	PREACT_S(2, 6, 12, l_c1.output, l_s1.preact, l_s1.weight);
	BIAS_S(6, 6, l_s1.preact, l_s1.bias);
	apply_step_function<<<64, 64>>>(l_s1.preact, l_s1.output, l_s1.O);
	
	PREACT_C(5, 6, 8, 6, l_s1.output, l_c2.preact, l_c2.weight);
	BIAS_C(36, 8, l_c2.preact, l_c2.bias);
	apply_step_function<<<64, 64>>>(l_c2.preact, l_c2.output, l_c2.O);

	PREACT_S(2, 6*6, 4, l_c2.output, l_s2.preact, l_s2.weight);
	BIAS_S(36, 4, l_s2.preact, l_s2.bias);
	apply_step_function<<<64, 64>>>(l_s2.preact, l_s2.output, l_s2.O);

	preact_f<<<64, 64>>>((float (*)[4][4])l_s2.output, l_f.preact, (float (*)[36][4][4])l_f.weight);
	bias_f<<<64, 64>>>(l_f.preact, l_f.bias);
	apply_step_function<<<64, 64>>>(l_f.preact, l_f.output, l_f.O);
}

static void learn()
{
	cublasCreate(&blas);

	float err;
	int iter = 3;

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
			l_s2.bp_clear();
			l_c2.bp_clear();

			makeError<<<10, 1>>>(l_f.d_preact, l_f.output, train_set[i].label, 10);

			bp_weight_f<<<64, 64>>>((float (*)[36][4][4])l_f.d_weight, l_f.d_preact, (float (*)[4][4])l_s2.output);
			bp_bias_f<<<64, 64>>>(l_f.bias, l_f.d_preact);

			bp_output_s2<<<64, 64>>>((float (*)[4][4])l_s2.d_output, (float (*)[36][4][4])l_f.weight, l_f.d_preact);
			BP_PREACT_S(36, 4, l_s2.d_preact, l_s2.d_output, l_s2.preact);
			BP_WEIGHT_S(2, 36, 4, l_s2.d_weight, l_s2.d_preact, l_c2.output);
			BP_BIAS_S(36, 4, l_s2.bias, l_s2.d_preact);

			BP_OUTPUT_C(2, 36, 4, l_c2.d_output, l_s2.weight, l_s2.d_preact);
			BP_PREACT_C(36, 8, l_c2.d_preact, l_c2.d_output, l_c2.preact);
			BP_WEIGHT_C(6, 5, 8, 6, l_c2.d_weight, l_c2.d_preact, l_s2.output);
			BP_BIAS_C(6, 8, 6, l_c2.bias, l_c2.preact);

			BP_OUTPUT_S(6, 12, 5, 6, l_s1.d_output, l_c2.weight, l_c2.d_preact);
			BP_PREACT_S(6, 12, l_s1.d_preact, l_s1.d_output, l_s1.preact);
			BP_WEIGHT_S(2, 6, 12, l_s1.d_weight, l_s1.d_preact, l_c1.output);
			BP_BIAS_S(6, 12, l_s1.bias, l_s1.d_preact);

			BP_OUTPUT_C(2, 6, 12, l_c1.d_output, l_s1.weight, l_s1.d_preact);
			BP_PREACT_C(6, 24, l_c1.d_preact, l_c1.d_output, l_c1.preact);
			BP_WEIGHT_C(6, 5, 23, 1, l_c1.d_weight, l_c1.d_preact, l_input.output);
			BP_BIAS_C(6, 24, 1, l_c1.bias, l_c1.preact);

			apply_grad<<<64, 64>>>(l_f.weight, l_f.d_weight, l_f.M * l_f.N);
			apply_grad<<<64, 64>>>(l_s2.weight, l_s2.d_weight, l_s2.M * l_s2.N);
			apply_grad<<<64, 64>>>(l_c2.weight, l_c2.d_weight, l_c2.M * l_c2.N);
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

