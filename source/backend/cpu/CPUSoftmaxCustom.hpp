//
//  CPUSoftmax.hpp
//  MNN
//
//  Created by MNN on 2018/07/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CPUSoftmaxCustom_hpp
#define CPUSoftmaxCustom_hpp

#include "core/Execution.hpp"

namespace MNN {
class CPUSoftmaxCustom : public Execution {
public:
    CPUSoftmaxCustom(Backend *b, int axis);
    virtual ~CPUSoftmaxCustom() = default;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    int _softmaxCommon(const float *srcData, float *dstData, int inside, int outside, int channel, float *maxValue,
                       float *sumValue, int threadNum);
    int _softmax1(const float *srcData, float *dstData, int outside, int channel, int threadNum);

    int mAxis;
    Tensor mStorage;
    Tensor mMaxValue;
    Tensor mSumValue;
    bool mNeedUnpackC4;
};
} // namespace MNN

#endif /* CPUSoftmaxCustom_hpp */
