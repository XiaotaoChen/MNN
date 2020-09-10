//
//  CPUSoftmax.cpp
//  MNN
//
//  Created by MNN on 2018/07/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <assert.h>
#include "backend/cpu/CPUSoftmaxCustom.hpp"
#include <math.h>
#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"
#include "core/Concurrency.h"
#include "core/Macro.h"
#include "core/TensorUtils.hpp"
namespace MNN {

int CPUSoftmaxCustom::_softmax1(const float *srcData, float *dstData, int outside, int channel, int threadNum) {
    // Max and sub
    MNN_CONCURRENCY_BEGIN(tId, threadNum)
    {
        const float *srcY = srcData + tId * channel;
        float *dstY       = dstData + tId * channel;
        for (int y = (int)tId; y < outside; y += threadNum, srcY += channel * threadNum, dstY += channel * threadNum) {
            float maxValue = srcY[0];
            {
                int c = 1;
                for (; c < channel; ++c) {
                    float value = srcY[c];
                    if (value > maxValue)
                        maxValue = value;
                }
            }

            for (int c = 0; c < channel; ++c) {
                dstY[c] = -srcY[c] + maxValue;
            }
        }
    }
    MNN_CONCURRENCY_END();
    
    //Exp
    auto schedule = ((CPUBackend*)backend())->multiThreadDivide(channel * outside);
    int sizeDivide = schedule.first;
    int scheduleNumber = schedule.second;

    MNN_CONCURRENCY_BEGIN(tId, scheduleNumber) {
        int start = sizeDivide * (int)tId;
        int realSize = sizeDivide;
        if (tId == scheduleNumber -1 ) {
            realSize = channel * outside - start;
        }
        if (realSize > 0) {
            MNNExp(dstData + start, dstData + start, realSize);
        }
    }
    MNN_CONCURRENCY_END();

    // Sum and div
    MNN_CONCURRENCY_BEGIN(tId, threadNum);
    {
        float *dstY       = dstData + tId * channel;
        for (int y = (int)tId; y < outside; y += threadNum, dstY += channel * threadNum) {
            // sum
            float sumValue = 0;

            for (int c = 0; c < channel; ++c) {
                sumValue += dstY[c];
            }

            // div
            {
                int c = 0;
                for (; c < channel; ++c) {
                    dstY[c] /= sumValue;
                }
            }
        }
    }
    MNN_CONCURRENCY_END();

    return 0;
}
int CPUSoftmaxCustom::_softmaxCommon(const float *srcData, float *dstData, int inside, int outside, int channel,
                               float *maxValue, float *sumValue, int threadNum) {

    if (inside != 1) {
        MNN_ERROR("%s:%d [CPUSoftmaxCustom.cpp] require inside to be 1, actually is %d\n", __FILE__, __LINE__, inside);
        assert(false);
    }
    // MNN_PRINT("[CPUSoftmaxCustom.cpp] inside: %d, outside: %d, channel:%d\n", inside, outside, channel);
    return _softmax1(srcData, dstData, outside, channel, threadNum);
}

ErrorCode CPUSoftmaxCustom::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto input           = inputs[0];
    const int dimensions = input->buffer().dimensions;

    const auto layout = TensorUtils::getDescribe(input)->dimensionFormat;
    mNeedUnpackC4     = layout == MNN_DATA_FORMAT_NC4HW4;

    if (mNeedUnpackC4) {
        int totalSize = 1;
        for (int i = 1; i < dimensions; ++i) {
            totalSize *= input->length(i);
        }
        mStorage.buffer().dim[0].extent = input->length(0);
        mStorage.buffer().dim[1].extent = totalSize;
        TensorUtils::getDescribe(&mStorage)->dimensionFormat = MNN_DATA_FORMAT_NHWC;
        mStorage.buffer().dimensions    = 2;
        mStorage.buffer().type          = input->getType();
        backend()->onAcquireBuffer(&mStorage, Backend::DYNAMIC);
    }

    int inside = 1;
    int dims   = input->buffer().dimensions;
    for (int i = mAxis + 1; i < dims; ++i) {
        inside *= input->length(i);
    }

    if (inside != 1) { // not run _softmax1, we need maxValue Tensor and sumValue Tensor.
        int threadNum = ((CPUBackend *)backend())->threadNumber();

        mMaxValue.buffer().dim[0].extent = inside * threadNum;
        mMaxValue.buffer().dimensions    = 1;
        mMaxValue.setType(DataType_DT_FLOAT);
        backend()->onAcquireBuffer(&mMaxValue, Backend::DYNAMIC);

        mSumValue.buffer().dim[0].extent = inside * threadNum;
        mSumValue.buffer().dimensions    = 1;
        mSumValue.setType(DataType_DT_FLOAT);
        backend()->onAcquireBuffer(&mSumValue, Backend::DYNAMIC);

        backend()->onReleaseBuffer(&mMaxValue, Backend::DYNAMIC);
        backend()->onReleaseBuffer(&mSumValue, Backend::DYNAMIC);
    }

    if (mNeedUnpackC4) {
        backend()->onReleaseBuffer(&mStorage, Backend::DYNAMIC);
    }

    return NO_ERROR;
}

ErrorCode CPUSoftmaxCustom::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    MNN_ASSERT(1 == inputs.size());
    MNN_ASSERT(1 == outputs.size());
    auto inputTensor        = inputs[0];
    auto outputTensor       = outputs[0];
    const auto inputDataPtr = inputTensor->host<float>();
    auto outputDataPtr      = outputTensor->host<float>();
    const int batch         = inputTensor->batch();
    const auto dims         = inputTensor->buffer().dimensions;

    float *tempData = nullptr;
    if (mNeedUnpackC4) {
        tempData = mStorage.host<float>();
    }

    int areaInput = 1;
    for (int i = 2; i < dims; ++i) {
        areaInput *= inputTensor->length(i);
    }
    int inside  = 1;
    int outside = 1;
    int channel = 1;
    for (int i = 0; i < mAxis; ++i) {
        outside *= inputTensor->length(i);
    }
    channel = inputTensor->length(mAxis);
    for (int i = mAxis + 1; i < dims; ++i) {
        inside *= inputTensor->length(i);
    }

    int threadNum = ((CPUBackend *)backend())->threadNumber();
    if (!mNeedUnpackC4) {
        _softmaxCommon(inputDataPtr, outputDataPtr, inside, outside, channel, mMaxValue.host<float>(),
                   mSumValue.host<float>(), threadNum);
        return NO_ERROR;
    }
    auto outputSize = outputTensor->elementSize();
    int batchSize = outputSize / batch;
    for (int batchIndex = 0; batchIndex < batch; ++batchIndex) {
        auto inputData  = inputDataPtr + batchIndex * batchSize;
        MNNUnpackC4(outputDataPtr + batchIndex * mStorage.length(1), inputData, areaInput, inputTensor->channel());
    }
    _softmaxCommon(outputDataPtr, tempData, inside, outside, channel, mMaxValue.host<float>(), mSumValue.host<float>(), threadNum);
    for (int batchIndex = 0; batchIndex < batch; ++batchIndex) {
        auto outputData = outputDataPtr + batchIndex * batchSize;
        auto tempPtr = tempData + batchIndex * mStorage.length(1);
        MNNPackC4(outputData, tempPtr, areaInput, outputTensor->channel());
    }
    return NO_ERROR;
}

CPUSoftmaxCustom::CPUSoftmaxCustom(Backend *b, int axis) : MNN::Execution(b), mAxis(axis), mStorage(2), mNeedUnpackC4(false) {
    // nothing to do
}

class CPUSoftmaxCustomCreator : public CPUBackend::Creator {
public:
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs,
                                const MNN::Op *op, Backend *backend) const override {
        auto axis = op->main_as_CustomAxis()->custom();
        if (axis < 0) {
            axis = inputs[0]->dimensions() + axis;
        }
        return new CPUSoftmaxCustom(backend, axis);
    }
};

REGISTER_CPU_OP_CREATOR(CPUSoftmaxCustomCreator, OpType_SoftmaxCustom);

} // namespace MNN
