#include "error.cuh"
#include <stdio.h>

#define USE_DP 1

#ifdef USE_DP
typedef double real; // 使用双精度浮点数结果正确
#else
typedef float real;
// 使用单精度浮点数结果错误，sum = 33554432.000000, 单精度浮点数只有6、7位精确的有效数字,
// 因此sum的值累加到3000多万后再将它和1 .23相加, 其值就不再增加了
// （小数被大数 "吃掉了", 但大数并没有变化

#endif

const int NUM_REPEATS = 20;
void timing(const real *x, const int N);
real reduce(const real *x, const int N);

// 折半规约法
int main(void)
{
    const int N = 100000000;
    const int M = sizeof(real) * N;
    real *x = (real *)malloc(M);
    for (int n = 0; n < N; ++n)
    {
        x[n] = 1.23;
    }

    timing(x, N);

    free(x);
    return 0;
}

void timing(const real *x, const int N)
{
    real sum = 0;

    for (int repeat = 0; repeat < NUM_REPEATS; ++repeat)
    {
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        CHECK(cudaEventRecord(start));
        cudaEventQuery(start);

        sum = reduce(x, N);

        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        float elapsed_time;
        CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
        printf("Time = %g ms.\n", elapsed_time);

        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }

    printf("sum = %f.\n", sum);
}

real reduce(const real *x, const int N)
{
    real sum = 0.0;
    for (int n = 0; n < N; ++n)
    {
        sum += x[n];
    }
    return sum;
}
