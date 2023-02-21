#include <immintrin.h>
#include <stdio.h>
 
int main(int argc, char* argv[])
{
 
    __m256i first = _mm256_set_epi64x(10, 20, 30, 40);
    __m256i second = _mm256_set_epi64x(5, 5, 5, 5);
    __m256i result = _mm256_add_epi64(first, second);
 
    long int* values = (long int*) &result;
    printf("==%ld \n", sizeof(long int));
    for (int i = 0;i < 4; i++)
    {
        printf("%ld ", values[i]);
    }
 
    return 0;
}