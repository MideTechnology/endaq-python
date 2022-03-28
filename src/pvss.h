//
// Created by cflanigan on 3/9/2022.
//

#ifndef ENDAQ_PYTHON_PVSS_H
#define ENDAQ_PYTHON_PVSS_H

struct sos {
    double a1;
    double a2;
    double b0;
    double b1;
    double b2;
};

struct accumReturn {
    double max;
    double min;
};

struct accumReturn accumulate(float*, long, struct sos);

struct accumReturn* accumulatePar(float*, long, struct sos*, const long);

#endif //ENDAQ_PYTHON_PVSS_H
