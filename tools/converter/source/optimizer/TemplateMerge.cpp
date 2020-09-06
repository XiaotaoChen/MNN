//
//  TemplateMerge.cpp
//  MNNConverter
//
//  Created by MNN on 2019/09/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include <iostream>

#include "TemplateMerge.hpp"
#include <set>
namespace MNN {
namespace Express {
bool TemplateMerge::onExecute(const std::vector<VARP>& outputs, std::shared_ptr<Parameters> parameters) {

    std::cout << "[TemplateMerge.cpp] mTemplates size: " << mTemplates.size() << std::endl;
    for (auto& iter : mTemplates) {
        std::cout << iter.first << " ";
    }
    std::cout << std::endl;


    bool hasChange = false;
    do {
        hasChange = false;
        for (auto& iter : mTemplates) {
            // if (iter.first != "TensorConverterMerge" && iter.first != "OnnxExtraManager" && iter.first != "TensorConverterSameMerge" && \
            //     iter.first != "TurnCompabilityOpAsNC4HW4" && iter.first != "TurnBinaryToElementwise") {
            //     std::cout << iter.first << " to continue..." << std::endl;
            //     continue;
            // }

            std::set<EXPRP> invalidVARP;
            auto execute = Variable::getExecuteOrder(outputs);

            // std::cout << "template: " << iter.first << " sequence size: " << execute.size() << std::endl;
            // int index = 0;
            // for (const auto& item: execute) {
            //     std::cout << index++ << ": " << item->name() << std::endl;
            // }

            for (auto var : execute) {
                if (var->get() == nullptr) {
                    continue;
                }
                if (invalidVARP.find(var) != invalidVARP.end()) {
                    continue;
                }
                if (iter.second.first(var)) {
                    auto res = iter.second.second(var);
                    hasChange = hasChange || res;
                } else {
                    invalidVARP.insert(var);
                }
            }
        }

        // std::cout << "*********** hasChange: " << std::boolalpha << hasChange << " *****************" << std::endl;

    } while (hasChange);
    return true;
}
TemplateMerge& TemplateMerge::getInstance(const std::string& pass) {
    static std::map<std::string, TemplateMerge> gMerge;
    if (gMerge.find(pass) == gMerge.end()) {
        gMerge.insert(std::make_pair(pass, TemplateMerge()));
    }
    auto iter = gMerge.find(pass);
    return iter->second;
}

void TemplateMerge::insertTemplate(std::string key, std::function<bool(EXPRP)> compare,
                                   std::function<bool(EXPRP)> transform) {
    mTemplates.insert(std::make_pair(key, std::make_pair(compare, transform)));
}
} // namespace Express
} // namespace MNN
