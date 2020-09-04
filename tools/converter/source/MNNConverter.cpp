//
//  MNNConverter.cpp
//  MNNConverter
//
//  Created by MNN on 2019/01/31.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "cli.hpp"

#include "MNN_generated.h"
#include "PostConverter.hpp"
#include "addBizCode.hpp"
#include "caffeConverter.hpp"
#include "liteConverter.hpp"
#include "onnxConverter.hpp"
#include "tensorflowConverter.hpp"
#include "writeFb.hpp"

int main(int argc, char *argv[]) {
    modelConfig modelPath;

    // parser command line arg
    try {
        Cli::initializeMNNConvertArgs(modelPath, argc, argv);
        Cli::printProjectBanner();

        std::cout << "Start to Convert Other Model Format To MNN Model..." << std::endl;
        std::unique_ptr<MNN::NetT> netT = std::unique_ptr<MNN::NetT>(new MNN::NetT());
        if (modelPath.model == modelConfig::CAFFE) {
            caffe2MNNNet(modelPath.prototxtFile, modelPath.modelFile, modelPath.bizCode, netT);
        } else if (modelPath.model == modelConfig::TENSORFLOW) {
            tensorflow2MNNNet(modelPath.modelFile, modelPath.bizCode, netT);
        } else if (modelPath.model == modelConfig::MNN) {
            addBizCode(modelPath.modelFile, modelPath.bizCode, netT);
        } else if (modelPath.model == modelConfig::ONNX) {
            onnx2MNNNet(modelPath.modelFile, modelPath.bizCode, netT);
        } else if (modelPath.model == modelConfig::TFLITE) {
            tflite2MNNNet(modelPath.modelFile, modelPath.bizCode, netT);
        } else {
            std::cout << "Not Support Model Type" << std::endl;
        }

        std::cout << "source MNN tensorName size: " << netT->tensorName.size() << std::endl;
        // for (int i=0; i<netT->tensorName.size(); i++) {
        //     std::cout << i << ": " << netT->tensorName[i] << std::endl;
        // }
        std::cout << "oplist size: " << netT->oplists.size() << std::endl;
        // for (int i=0; i<netT->oplists.size(); i++) {
        //     std::cout << i << ": " << MNN::EnumNameOpType(netT->oplists[i]->type) << std::endl;
        // }


        if (modelPath.model != modelConfig::MNN) {
            std::cout << "Start to Optimize the MNN Net..." << std::endl;
            std::unique_ptr<MNN::NetT> newNet = optimizeNet(netT, modelPath.forTraining);

            std::cout << "optimized MNN tensorName size: " << newNet->tensorName.size() << std::endl;
            // for (int i=0; i<newNet->tensorName.size(); i++) {
            //     std::cout << i << ": " << newNet->tensorName[i] << std::endl;
            // }
            std::cout << "oplist size: " << newNet->oplists.size() << std::endl;
            // for (int i=0; i<newNet->oplists.size(); i++) {
            //     std::cout << i << ": " << MNN::EnumNameOpType(newNet->oplists[i]->type) << std::endl;
            // }


            writeFb(newNet, modelPath.MNNModel, modelPath.benchmarkModel, modelPath.saveHalfFloat);
        } else {
            writeFb(netT, modelPath.MNNModel, modelPath.benchmarkModel, modelPath.saveHalfFloat);
        }
    } catch (const cxxopts::OptionException &e) {
        std::cerr << "Error while parsing options! " << std::endl;
        std::cerr << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }
    std::cout << "Converted Done!" << std::endl;

    return 0;
}
