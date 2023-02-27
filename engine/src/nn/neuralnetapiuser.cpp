/*
  CrazyAra, a deep learning chess variant engine
  Copyright (C) 2018       Johannes Czech, Moritz Willig, Alena Beyer
  Copyright (C) 2019-2020  Johannes Czech

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

/*
 * @file: neuralnetapiuser.h
 * Created on 06.10.2020
 * @author: queensgambit
 */

#include "neuralnetapiuser.h"
#include "stateobj.h"
#ifdef TENSORRT
#include "NvInfer.h"
#include <cuda_runtime_api.h>
#include "common.h"
#endif

NeuralNetAPIUser::NeuralNetAPIUser(NeuralNetAPI *net):
    net(net),
    auxiliaryOutputs(nullptr)
{
    // allocate memory for all predictions and results
#ifdef TENSORRT
//MR #ifdef FALSE
#ifdef DYNAMIC_NN_ARCH
    //info_string("//MR: before CHECK(cudaMallocHost((void**) &inputPlanes, net->get_batch_size() * net->get_nb_input_values_total() * sizeof(float)))");
    CHECK(cudaMallocHost((void**) &inputPlanes, net->get_batch_size() * net->get_nb_input_values_total() * sizeof(float)));
#else
     CHECK(cudaMallocHost((void**) &inputPlanes, net->get_batch_size() * StateConstants::NB_VALUES_TOTAL() * sizeof(float)));
#endif
    //info_string("//MR: before CHECK(cudaMallocHost((void**) &valueOutputs, net->get_batch_size() * sizeof(float)))");
    CHECK(cudaMallocHost((void**) &valueOutputs, net->get_batch_size() * sizeof(float)));
    CHECK(cudaMallocHost((void**) &probOutputs, net->get_batch_size() * net->get_nb_policy_values() * sizeof(float)));
    if (net->has_auxiliary_outputs()) {
        CHECK(cudaMallocHost((void**) &auxiliaryOutputs, net->get_batch_size() * net->get_nb_auxiliary_outputs() * sizeof(float)));
    }
#else
    //info_string("//MR: before inputPlanes = new float[net->get_batch_size() * net->get_nb_input_values_total()]");
    inputPlanes = new float[net->get_batch_size() * net->get_nb_input_values_total()];
    //info_string("//MR: before valueOutputs = new float[net->get_batch_size()]");
    valueOutputs = new float[net->get_batch_size()];
    //info_string("//MR: before probOutputs = new float[net->get_batch_size() * net->get_nb_policy_values()]");
    probOutputs = new float[net->get_batch_size() * net->get_nb_policy_values()];
    //info_string("//MR: after probOutputs = new float[net->get_batch_size() * net->get_nb_policy_values()]");
#ifdef DYNAMIC_NN_ARCH
    if (net->has_auxiliary_outputs()) {
        auxiliaryOutputs = new float[net->get_batch_size() * net->get_nb_auxiliary_outputs()];
    }
#else
    if (StateConstants::NB_AUXILIARY_OUTPUTS()) {
         auxiliaryOutputs = new float[net->get_batch_size() * StateConstants::NB_AUXILIARY_OUTPUTS()];
    }
#endif
#endif
}

NeuralNetAPIUser::~NeuralNetAPIUser()
{
#ifdef TENSORRT
//MR #ifdef FALSE
    //info_string("//MR: free inputPlanes");
    CHECK(cudaFreeHost(inputPlanes));
    //info_string("//MR: free valueOutputs");
    CHECK(cudaFreeHost(valueOutputs));
    //info_string("//MR: free propOutputs");
    CHECK(cudaFreeHost(probOutputs));
#ifdef DYNAMIC_NN_ARCH
    if (net->has_auxiliary_outputs()) {
#else
    if (StateConstants::NB_AUXILIARY_OUTPUTS()) {
#endif
        //info_string("//MR: free auxOutputs");
        CHECK(cudaFreeHost(auxiliaryOutputs));
    }
#else
    //info_string("//MR: before delete [] inputPlanes");
    delete [] inputPlanes;
    //info_string("//MR: before delete [] valueOutputs");
    delete [] valueOutputs;
    //info_string("//MR: before delete [] probOutputs");
    delete [] probOutputs;
    //info_string("//MR: after delete [] probOutputs");
#ifdef DYNAMIC_NN_ARCH
    if (net->has_auxiliary_outputs()) {
#else
    if (StateConstants::NB_AUXILIARY_OUTPUTS()) {
#endif
        delete [] auxiliaryOutputs;
    }
#endif
}
