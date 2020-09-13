//
//  ReluOnnx.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <stdio.h>
#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(ReluCustomOnnx);

MNN::OpType ReluCustomOnnx::opType() {
    return MNN::OpType_ReLUCustom;
}
MNN::OpParameter ReluCustomOnnx::type() {
    return MNN::OpParameter_ReLUCustom;
}

void ReluCustomOnnx::run(MNN::OpT* dstOp, const onnx::NodeProto* onnxNode,
                   std::vector<const onnx::TensorProto*> initializers) {
    auto relu = new MNN::ReLUCustomT;
    relu->threshold = .0f;
    dstOp->main.value = relu;
}

REGISTER_CONVERTER(ReluCustomOnnx, Relu);
