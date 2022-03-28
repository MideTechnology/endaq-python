//
// Created by cflanigan on 3/9/2022.
//

#include<math.h>
#include<omp.h>
#include<stdlib.h>
#include<stdio.h>

#include "pvss.h"


struct accumReturn accumulate(float* arr, long arrLen, struct sos filt) {

    struct accumReturn out;

    double s1, s2;
    s1 = 0;
    s2 = 0;

    double b0, b1, b2, a1, a2;
    b0 = filt.b0;
    b1 = filt.b1;
    b2 = filt.b2;
    a1 = filt.a1;
    a2 = filt.a2;

    double xn, yn;
    double min, max;
    min = INFINITY;
    max = -INFINITY;

    for (long i = 0; i < arrLen; i++) {
        xn = arr[i];

        yn = xn*b0 + s1;

        s1 = xn*b1 - yn*a1 + s2;

        s2 = xn*b2 - yn*a2;

        if (yn > max) {
            max = yn;
        }
        if (yn < min) {
            min = yn;
        }

    }

    out.max = max;
    out.min = min;

    return out;

}


struct accumReturn* accumulatePar(float* arr, long arrLen, struct sos* filtBank, const long bankLen) {

    struct accumReturn* out = (struct accumReturn*) malloc(sizeof(struct accumReturn)*bankLen);
    long i;

    #pragma omp parallel shared(arr, arrLen, filtBank, out, bankLen) private(i)
    {
        #pragma omp for
        for (i = 0; i < bankLen; i++) {
            out[i] = accumulate(arr, arrLen, filtBank[i]);
        }
    }


    return out;

}


