//
// Created by cflanigan on 3/25/2022.
//

#include <gtest/gtest.h>

#include "../src/pvss.c"

TEST(PVSSTest, BasicAssertions) {

    const long arrLen = 5;
    float arr[arrLen] = {1, 2, 3, 4, 5};
    struct sos filt = {0, 0, 1, 0, 0};

    struct accumReturn expect = {5, 1};
    struct accumReturn actual = accumulate(arr, arrLen, filt);

    EXPECT_FLOAT_EQ(actual.max, expect.max);
    EXPECT_FLOAT_EQ(actual.min, expect.min);

}

TEST(ImpulseIdentity, BasicAssertions) {

    const long arrLen = 11;
    float arr[arrLen] = {0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    struct sos filt = {0, 0, 1, 20, -1000};

    struct accumReturn expect = {20, -1000};
    struct accumReturn actual = accumulate(arr, arrLen, filt);

    EXPECT_FLOAT_EQ(actual.max, expect.max);
    EXPECT_FLOAT_EQ(actual.min, expect.min);

}

TEST(ImpulseDecayedSine, BasicAssertions) {

    const long arrLen = 20;
    float arr[arrLen] = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    struct sos filt = {-1.711901729331276, 0.810000000000000, 0.278115294937453, 0, 0};

    struct accumReturn expect = {0.6239882, -0.2175712};
    struct accumReturn actual = accumulate(arr, arrLen, filt);

    EXPECT_FLOAT_EQ(actual.max, expect.max);
    EXPECT_FLOAT_EQ(actual.min, expect.min);

}
