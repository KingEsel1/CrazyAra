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
 * @file: searchthread.cpp
 * Created on 23.05.2019
 * @author: queensgambit
 */

#include "searchthread.h"
#ifdef TENSORRT
#include "NvInfer.h"
#include <cuda_runtime_api.h>
#include "common.h"
#endif

#include <stdlib.h>
#include <climits>
#include "util/blazeutil.h"


size_t SearchThread::get_max_depth() const
{
    return depthMax;
}

SearchThread::SearchThread(NeuralNetAPI* netBatch, const SearchSettings* searchSettings, MapWithMutex* mapWithMutex) :
    NeuralNetAPIUser(netBatch),
    rootNode(nullptr), rootState(nullptr), newState(nullptr),  // will be be set via setter methods
    newNodes(make_unique<FixedVector<Node*>>(searchSettings->batchSize)),
    newNodeSideToMove(make_unique<FixedVector<SideToMove>>(searchSettings->batchSize)),
    transpositionValues(make_unique<FixedVector<float>>(searchSettings->batchSize * 2)),
    isRunning(true), mapWithMutex(mapWithMutex), searchSettings(searchSettings),
    tbHits(0), depthSum(0), depthMax(0), visitsPreSearch(0), 
#ifdef MCTS_SINGLE_PLAYER
    terminalNodeCache(1),
#else
    terminalNodeCache(searchSettings->batchSize * 2),
#endif
    reachedTablebases(false),
    factPlanes(nullptr) //MR
{
    searchLimits = nullptr;  // will be set by set_search_limits() every time before go()
    trajectoryBuffer.reserve(DEPTH_INIT);
    actionsBuffer.reserve(DEPTH_INIT);
}

void SearchThread::set_root_node(Node *value)
{
    rootNode = value;
    visitsPreSearch = rootNode->get_visits();
}

void SearchThread::set_search_limits(SearchLimits *s)
{
    searchLimits = s;
}

bool SearchThread::is_running() const
{
    return isRunning;
}

void SearchThread::set_is_running(bool value)
{
    isRunning = value;
}

void SearchThread::set_reached_tablebases(bool value)
{
    reachedTablebases = value;
}

//MR
void SearchThread::set_fact_planes(float* value)
{
    factPlanes = value;
}

Node* SearchThread::add_new_node_to_tree(StateObj* newState, Node* parentNode, ChildIdx childIdx, NodeBackup& nodeBackup)
{
    bool transposition;
    Node* newNode = parentNode->add_new_node_to_tree(mapWithMutex, newState, childIdx, searchSettings, transposition);
    if (transposition) {
        const float qValue =  parentNode->get_child_node(childIdx)->get_value();
        //MR
        const float noveltyScore = parentNode->get_child_node(childIdx)->get_novelty_score();
        transpositionValues->add_element(qValue);
        //MR
        transpositionNoveltyScores->add_element(noveltyScore);
        //info_string("//MR: transposition node! mit qValue = " + to_string(qValue) + " und noveltyScore = " + to_string(noveltyScore));
        nodeBackup = NODE_TRANSPOSITION;
        return newNode;
    }
    nodeBackup = NODE_NEW_NODE;
    return newNode;
}

void SearchThread::stop()
{
    isRunning = false;
}

Node *SearchThread::get_root_node() const
{
    return rootNode;
}

SearchLimits *SearchThread::get_search_limits() const
{
    return searchLimits;
}

void random_playout(Node* currentNode, ChildIdx& childIdx)
{
    if (currentNode->is_fully_expanded()) {
        const size_t idx = rand() % currentNode->get_number_child_nodes();
        if (currentNode->get_child_node(idx) == nullptr || !currentNode->get_child_node(idx)->is_playout_node()) {
            childIdx = idx;
            return;
        }
        if (currentNode->get_child_node(idx)->get_node_type() == UNSOLVED) {
            childIdx = idx;
            return;
        }
        childIdx = uint16_t(-1);
    }
    else {
        childIdx = min(size_t(currentNode->get_no_visit_idx()), currentNode->get_number_child_nodes()-1);
        currentNode->increment_no_visit_idx();
        return;
    }
}

Node* SearchThread::get_starting_node(Node* currentNode, NodeDescription& description, ChildIdx& childIdx)
{
    size_t depth = get_random_depth();
    for (uint curDepth = 0; curDepth < depth; ++curDepth) {
        currentNode->lock();
        childIdx = get_best_action_index(currentNode, true, 0, 0);
        Node* nextNode = currentNode->get_child_node(childIdx);
        if (nextNode == nullptr || !nextNode->is_playout_node() || nextNode->get_visits() < searchSettings->epsilonGreedyCounter || nextNode->get_node_type() != UNSOLVED) {
            currentNode->unlock();
            break;
        }
        currentNode->unlock();
        actionsBuffer.emplace_back(currentNode->get_action(childIdx));
        currentNode = nextNode;
        ++description.depth;
    }
    return currentNode;
}

Node* SearchThread::get_new_child_to_evaluate(NodeDescription& description)
{
    //info_string("//MR: get_new_child_to_evaluate() -> erstelle eine neue Trajektorie");
    description.depth = 0;
    Node* currentNode = rootNode;
    Node* nextNode;

    ChildIdx childIdx = uint16_t(-1);
    if (searchSettings->epsilonGreedyCounter && rootNode->is_playout_node() && rand() % searchSettings->epsilonGreedyCounter == 0) {
        currentNode = get_starting_node(currentNode, description, childIdx);
        currentNode->lock();
        random_playout(currentNode, childIdx);
        currentNode->unlock();
    }
    else if (searchSettings->epsilonChecksCounter && rootNode->is_playout_node() && rand() % searchSettings->epsilonChecksCounter == 0) {
        currentNode = get_starting_node(currentNode, description, childIdx);
        currentNode->lock();
        childIdx = select_enhanced_move(currentNode);
        if (childIdx ==  uint16_t(-1)) {
            random_playout(currentNode, childIdx);
        }
        currentNode->unlock();
    }

    while (true) {
        currentNode->lock();
        if (childIdx == uint16_t(-1)) {
            //info_string("//MR: select_child_node(searchSettings)");
            childIdx = currentNode->select_child_node(searchSettings);
            //info_string("//MR: childIdx aus select_child_node(searchSettings) ist " + to_string(childIdx) + "\n");
        }
        //info_string("//MR apply_virtual_loss_to_child()");
        currentNode->apply_virtual_loss_to_child(childIdx, searchSettings->virtualLoss);
        trajectoryBuffer.emplace_back(NodeAndIdx(currentNode, childIdx));

        nextNode = currentNode->get_child_node(childIdx);
        description.depth++;
        if (nextNode == nullptr) {
            //info_string("//MR: nextNode ist null -> fuege Knoten mit nullptr und initWerten hinzu ...");
#ifdef MCTS_STORE_STATES
            StateObj* newState = currentNode->get_state()->clone();
#else
            newState = unique_ptr<StateObj>(rootState->clone());
            assert(actionsBuffer.size() == description.depth-1);
            for (Action action : actionsBuffer) {
                newState->do_action(action);
            }
#endif
            newState->do_action(currentNode->get_action(childIdx));
            currentNode->increment_no_visit_idx();
#ifdef MCTS_STORE_STATES
            nextNode = add_new_node_to_tree(newState, currentNode, childIdx, description.type);
#else
            //info_string("//MR: ... und haenge den Knoten an childIndex = " + to_string(childIdx) + " an den Baum (nullptr ersetzen durch Knoten)");
            nextNode = add_new_node_to_tree(newState.get(), currentNode, childIdx, description.type);
#endif
            currentNode->unlock();

            if (description.type == NODE_NEW_NODE) {
#ifdef SEARCH_UCT
                Node* nextNode = currentNode->get_child_node(childIdx);
                nextNode->set_value(newState->random_rollout());
                nextNode->enable_has_nn_results();
                if (searchSettings->useTranspositionTable && !nextNode->is_terminal()) {
                    mapWithMutex->mtx.lock();
                    mapWithMutex->hashTable.insert({nextNode->hash_key(), nextNode});
                    mapWithMutex->mtx.unlock();
                }
#else
                // fill a new board in the input_planes vector
                // we shift the index by nbNNInputValues each time
                newState->get_state_planes(true, inputPlanes + newNodes->size() * net->get_nb_input_values_total(), net->get_version());
                // save a reference newly created list in the temporary list for node creation
                // it will later be updated with the evaluation of the NN
                newNodeSideToMove->add_element(newState->side_to_move());
#endif
            }
            return nextNode;
        }
        if (nextNode->is_terminal()) {
            //info_string("//MR: nextNode->is_terminal()");
            description.type = NODE_TERMINAL;
            currentNode->unlock();
            return nextNode;
        }
        if (!nextNode->has_nn_results()) {
            //info_string("//MR: !nextNode->has_nn_results()");
            description.type = NODE_COLLISION;
            currentNode->unlock();
            return nextNode;
        }
        if (nextNode->is_transposition()) {
            //info_string("//MR: nextNode->is_transposition()");
            nextNode->lock();
            const uint_fast32_t transposVisits = currentNode->get_real_visits(childIdx);
            const double transposQValue = -currentNode->get_q_sum(childIdx, searchSettings->virtualLoss) / transposVisits;

            if (nextNode->is_transposition_return(transposQValue)) {
                const float qValue = get_transposition_q_value(transposVisits, transposQValue, nextNode->get_value());
                //MR Der NoveltyScore muss nicht wie die qValues gecheckt werden, da kein virtualLoss angewendet wird... kann einfach übernommen werden
                //MR aber nochmal fragen! -> mit dem check ist die Funktion  get_transposition_q_value(transposVisits, transposQValue, nextNode->get_value()) gemeint
                const float noveltyScore = nextNode->get_novelty_score();
                nextNode->unlock();
                description.type = NODE_TRANSPOSITION;
                transpositionValues->add_element(qValue);
                //MR
                transpositionNoveltyScores->add_element(noveltyScore);
                currentNode->unlock();
                return nextNode;
            }
            nextNode->unlock();
        }
        currentNode->unlock();
#ifndef MCTS_STORE_STATES
        actionsBuffer.emplace_back(currentNode->get_action(childIdx));
#endif
        currentNode = nextNode;
        childIdx = uint16_t(-1);
    }
}

void SearchThread::set_root_state(StateObj* value)
{
    rootState = value;
}

size_t SearchThread::get_tb_hits() const
{
    return tbHits;
}

void SearchThread::reset_stats()
{
    tbHits = 0;
    depthMax = 0;
    depthSum = 0;
}

//MR add inputPlanes to params
void fill_nn_results(size_t batchIdx, bool isPolicyMap, const float* valueOutputs, const float* probOutputs, const float* auxiliaryOutputs, Node *node, size_t& tbHits, bool mirrorPolicy, const SearchSettings* searchSettings, bool isRootNodeTB, const float* inputPlanes, float* factPlanes, int number_input_total)
{
    //info_string("//MR: fill_nn_results(...) to newNode(s)");
    node->set_probabilities_for_moves(get_policy_data_batch(batchIdx, probOutputs, isPolicyMap), mirrorPolicy);
    node_post_process_policy(node, searchSettings->nodePolicyTemperature, searchSettings);
    node_assign_value(node, valueOutputs, tbHits, batchIdx, isRootNodeTB);
    //MR
    node_assign_novelty_score(node, valueOutputs, batchIdx, searchSettings, inputPlanes, factPlanes, number_input_total);
#ifdef MCTS_STORE_STATES
    node->set_auxiliary_outputs(get_auxiliary_data_batch(batchIdx, auxiliaryOutputs));
#endif
    node->enable_has_nn_results();
}

void SearchThread::set_nn_results_to_child_nodes()
{
    size_t batchIdx = 0;
    for (auto node: *newNodes) {
        if (!node->is_terminal()) {
            fill_nn_results(batchIdx, net->is_policy_map(), valueOutputs, probOutputs, auxiliaryOutputs, node,
                            tbHits, rootState->mirror_policy(newNodeSideToMove->get_element(batchIdx)),
                            searchSettings, rootNode->is_tablebase(), inputPlanes, factPlanes, net->get_nb_input_values_total());
        }
        ++batchIdx;
    }
}

void SearchThread::backup_value_outputs()
{
    //info_string("//MR: jetzt kommt backup_values fuer newTrajectories mit Size: " + to_string(newTrajectories.size()));
    backup_values(*newNodes, newTrajectories);
    newNodeSideToMove->reset_idx();
    //MR add noveltyScore to params
    //info_string("//MR: jetzt kommt backup_values fuer transpositionTrajectories mit Size: " + to_string(transpositionTrajectories.size()));
    backup_values(transpositionValues.get(), transpositionTrajectories, transpositionNoveltyScores.get());
}

void SearchThread::backup_collisions() {
    for (size_t idx = 0; idx < collisionTrajectories.size(); ++idx) {
        backup_collision(searchSettings->virtualLoss, collisionTrajectories[idx]);
    }
    collisionTrajectories.clear();
}

bool SearchThread::nodes_limits_ok()
{
    return (searchLimits->nodes == 0 || (rootNode->get_node_count() < searchLimits->nodes)) &&
            (searchLimits->simulations == 0 || (rootNode->get_visits() < searchLimits->simulations)) &&
            (searchLimits->nodesLimit == 0 || (rootNode->get_node_count() < searchLimits->nodesLimit));
}

bool SearchThread::is_root_node_unsolved()
{
#ifdef MCTS_TB_SUPPORT
    return is_unsolved_or_tablebase(rootNode->get_node_type());
#else
    return rootNode->get_node_type() == UNSOLVED;
#endif
}

size_t SearchThread::get_avg_depth()
{
    return size_t(double(depthSum) / (rootNode->get_visits() - visitsPreSearch) + 0.5);
}

void SearchThread::create_mini_batch()
{
    // select nodes to add to the mini-batch
    NodeDescription description;
    size_t numTerminalNodes = 0;

    while (!newNodes->is_full() &&
           collisionTrajectories.size() != searchSettings->batchSize &&
           !transpositionValues->is_full() &&
           numTerminalNodes < terminalNodeCache) {

        trajectoryBuffer.clear();
        actionsBuffer.clear();
        //info_string("//MR: create_mini_batch() VOR get_new_child_to_evaluate(description)\n");
        Node* newNode = get_new_child_to_evaluate(description);
        depthSum += description.depth;
        depthMax = max(depthMax, description.depth);

        if(description.type == NODE_TERMINAL) {
            ++numTerminalNodes;
            //MR add noveltyScore to params
            backup_value<true>(newNode->get_value(), searchSettings->virtualLoss, trajectoryBuffer, searchSettings->mctsSolver, newNode->get_novelty_score());
            
        }
        else if (description.type == NODE_COLLISION) {
            // store a pointer to the collision node in order to revert the virtual loss of the forward propagation
            collisionTrajectories.emplace_back(trajectoryBuffer);
        }
        else if (description.type == NODE_TRANSPOSITION) {
            transpositionTrajectories.emplace_back(trajectoryBuffer);
        }
        else {  // NODE_NEW_NODE
            newNodes->add_element(newNode);
            newTrajectories.emplace_back(trajectoryBuffer);
        }
    }
}

void SearchThread::thread_iteration()
{
    create_mini_batch();
#ifndef SEARCH_UCT
    if (newNodes->size() != 0) {
        //info_string("//MR: PREDICT FOR NEWNODES!!! in thread_iteration()");
        net->predict(inputPlanes, valueOutputs, probOutputs, auxiliaryOutputs);
        set_nn_results_to_child_nodes();
    }
#endif
    //info_string("//MR: \n\n ---------------> BACKPROP!!! in thread_iteration()");
    backup_value_outputs();
    //info_string("//MR: \n jetzt werden die Kollisionen aufgehoben!");
    backup_collisions();
}

void run_search_thread(SearchThread *t)
{
    t->set_is_running(true);
    t->reset_stats();
    while(t->is_running() && t->nodes_limits_ok() && t->is_root_node_unsolved()) {
        //info_string("//MR: run_search_thread()");
        t->thread_iteration();
    }
    t->set_is_running(false);
    //info_string("//MR: ein thread ist fertig!!! run_search_thread()");
}

void SearchThread::backup_values(FixedVector<Node*>& nodes, vector<Trajectory>& trajectories) {
    for (size_t idx = 0; idx < nodes.size(); ++idx) {
        Node* node = nodes.get_element(idx);
#ifdef MCTS_TB_SUPPORT
        const bool solveForTerminal = searchSettings->mctsSolver && node->is_tablebase();
        //MR add node->get_novelty_score() to params
        backup_value<false>(node->get_value(), searchSettings->virtualLoss, trajectories[idx], solveForTerminal, node->get_novelty_score());
#else
        //MR add node->get_novelty_score() to params
        backup_value<false>(node->get_value(), searchSettings->virtualLoss, trajectories[idx], false, node->get_novelty_score());
#endif
    }
    nodes.reset_idx();
    trajectories.clear();
}

void SearchThread::backup_values(FixedVector<float>* values, vector<Trajectory>& trajectories, FixedVector<float>* noveltyScores) {
    for (size_t idx = 0; idx < values->size(); ++idx) {                                              //MR
        const float value = values->get_element(idx);
        //MR
        const float noveltyScore = noveltyScores->get_element(idx);
        backup_value<true>(value, searchSettings->virtualLoss, trajectories[idx], false, noveltyScore);
    }
    values->reset_idx();
    trajectories.clear();
}

ChildIdx SearchThread::select_enhanced_move(Node* currentNode) const {
    if (currentNode->is_playout_node() && !currentNode->was_inspected() && !currentNode->is_terminal()) {

        // iterate over the current state
        unique_ptr<StateObj> pos = unique_ptr<StateObj>(rootState->clone());
        for (Action action : actionsBuffer) {
            pos->do_action(action);
        }

        // make sure a check has been explored at least once
        for (size_t childIdx = currentNode->get_no_visit_idx(); childIdx < currentNode->get_number_child_nodes(); ++childIdx) {
            if (pos->gives_check(currentNode->get_action(childIdx))) {
                for (size_t idx = currentNode->get_no_visit_idx(); idx < childIdx+1; ++idx) {
                    currentNode->increment_no_visit_idx();
                }
                return childIdx;
            }
        }
        // a full loop has been done
        currentNode->set_as_inspected();
    }
    return uint16_t(-1);
}

void node_assign_value(Node *node, const float* valueOutputs, size_t& tbHits, size_t batchIdx, bool isRootNodeTB)
{
#ifdef MCTS_TB_SUPPORT
    if (node->is_tablebase()) {
        ++tbHits;
        // TODO: Improvement the value assignment for table bases
        if (node->get_value() != 0 && isRootNodeTB) {
            // use the average of the TB entry and NN eval for non-draws
            node->set_value((valueOutputs[batchIdx] + node->get_value()) * 0.5f);
        }
        return;
    }
#endif
    //info_string("//MR: newNode node_assign_value(...) mit value: " + to_string(valueOutputs[batchIdx]));
    node->set_value(valueOutputs[batchIdx]);
}

void node_assign_novelty_score(Node* node, const float* valueOutputs, size_t batchIdx, const SearchSettings* searchSettings, const float* inputPlanes, float* factPlanes, int numberInputTotal)
{
    //MR calculate novelty score here!
    bool isNovel = false;
    int numberOfNovelFacts = 0;
    int numberOfNovelPocketPieces = 0;
    int row = 0;
    int col = 0;
    int chanel = 0;
    int index = 0;

    //MR Die factPlanes fehlen hier noch, in denen die neuen Values gespeichert werden!!! siehe python Version

    // 8 * 8 = 64 squares * 12 piecetypes
    size_t inputPlanesSize = 8 * 8 * 12; //MR ist das evlt nbNNInputValues? net->get_nb_input_values_total() siehe get_child_node_to_evaluate()
    // ckeck if the value of the current state is greater than any score of a fact that is active at the moment
    // this loop covers all facts on the board (first 12 planes of input representation)
    for (int i = 0; i < inputPlanesSize; i++)
    {
        if (!searchSettings->useFactPlanesOffset && inputPlanes[i] > 0) { //MR Ohne Offset -> eigenlich falsch, aber gute Ergebnisse!??
            chanel = i / 64;
            col = (i % 64) % 8;
            row = (i % 64) / 8;
            //info_string("//MR: ohne Offset!! i = " + to_string(i) + " | chanel = " + to_string(chanel) + " | row = " + to_string(row) + " | col = " + to_string(col)
              //  + " und valueOutputs[batchIdx] = " + to_string(valueOutputs[batchIdx]) + " und factPlanes[i] = " + to_string(factPlanes[i]));
            if (valueOutputs[batchIdx] > factPlanes[i]) {
                factPlanes[i] = valueOutputs[batchIdx];
                isNovel = true;
                numberOfNovelFacts++; //MR raus nach debug!
            }
        }     

        if (searchSettings->useFactPlanesOffset && inputPlanes[i + batchIdx * numberInputTotal] > 0) { //MR HIER MUSS DER SHIFT HIN MIT bathIdx!!!!!!
            index = i + batchIdx * numberInputTotal;
            chanel = i / 64;
            col = (i % 64) % 8;
            row = (i % 64) / 8;
            //info_string("//MR: chess mit Offset! idx=" + to_string(index) + "i=" + to_string(i) + " | chanel=" + to_string(chanel) + " | row=" + to_string(row) + " | col=" + to_string(col)
              //  + " | batchIdx= " + to_string(batchIdx) + " | numbInpTotal=" + to_string(numberInputTotal) + " | valueOutputs[batchIdx]=" + to_string(valueOutputs[batchIdx]) + " | factPlanes[i]=" + to_string(factPlanes[i]));
            if (valueOutputs[batchIdx] > factPlanes[i]) {
                factPlanes[i] = valueOutputs[batchIdx];
                isNovel = true;
                numberOfNovelFacts++; //MR raus nach debug!
            }
        }
    }
    info_string("//MR: MODE_CHESS and MODE_CRAZYHOUSE!");
#ifdef MODE_CRAZYHOUSE
    info_string("//MR: mode crazyhouse");
    if (searchSettings->usePocketForNovelty) {
        // this loop covers the facts for the pocket pieces (planes with index 14 to 23)
        inputPlanesSize = 8 * 8 * 10;
        int offsetForPocketPieces = 8 * 8 * 14; // first 14 planes -> see ppt pdf
        for (int i = offsetForPocketPieces; i < inputPlanesSize + offsetForPocketPieces; i++)
        {
            if (!searchSettings->useFactPlanesOffset && inputPlanes[i] > 0) {
                index = i + batchIdx * numberInputTotal;
                chanel = i / 64;
                //info_string("//MR: pocket kein Offset! idx=" + to_string(index) + "i=" + to_string(i) + " | chanel=" + to_string(chanel)
                  //  + " | batchIdx= " + to_string(batchIdx) + " | numbInp=" + to_string(numberInputTotal) + " | value[bIdx]=" + to_string(valueOutputs[batchIdx]) + " | factPlanes[i]=" + to_string(factPlanes[i]));
                if (valueOutputs[batchIdx] > factPlanes[i]) {
                    factPlanes[i] = valueOutputs[batchIdx];
                    isNovel = true;
                    numberOfNovelFacts++; //MR raus nach debug!
                    numberOfNovelPocketPieces++; //MR raus nach debug!
                }
            }

            if (searchSettings->useFactPlanesOffset && inputPlanes[i + batchIdx * numberInputTotal] > 0) {
                index = i + batchIdx * numberInputTotal;
                chanel = i / 64;
                //info_string("//MR: pocket Offset! idx=" + to_string(index) + "i=" + to_string(i) + " | chanel=" + to_string(chanel)
                  //  + " | batchIdx= " + to_string(batchIdx) + " | numbInp=" + to_string(numberInputTotal) + " | value[bIdx]=" + to_string(valueOutputs[batchIdx]) + " | factPlanes[i]=" + to_string(factPlanes[i]));
                if (valueOutputs[batchIdx] > factPlanes[i]) {
                    factPlanes[i] = valueOutputs[batchIdx];
                    isNovel = true;
                    numberOfNovelFacts++; //MR raus nach debug!
                    numberOfNovelPocketPieces++; //MR raus nach debug!
                }
            }
        }
    }
#endif // MODE_CRAZYHOUSE

    if (isNovel) {
        //info_string("//MR: float searchSettings->noveltyValue = " + to_string(searchSettings->noveltyValue) + " und in double ist es: " + to_string((double) searchSettings->noveltyValue));
        node->set_novelty_score((double) searchSettings->noveltyValue);
    }
    //info_string("//MR: newNode node_assign_novelty_score(): isNovel = " + to_string(isNovel) + " , noveltyScore = " + to_string(node->get_novelty_score()) + " and number of novel facts = " + to_string(numberOfNovelFacts));
}

void node_post_process_policy(Node *node, float temperature, const SearchSettings* searchSettings)
{
    node->enhance_moves(searchSettings);
    node->apply_temperature_to_prior_policy(temperature);
}

size_t get_random_depth()
{
    const int randInt = rand() % 100 + 1;
    return std::ceil(-std::log2(1 - randInt / 100.0) - 1);
}
