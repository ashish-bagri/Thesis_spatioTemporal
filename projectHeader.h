#ifndef _PROJ_HEAD
#define _PROJ_HEAD
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include "cv.h"
#include "highgui.h"
#include "opencv2/legacy/legacy.hpp"
#include "opencv2/ml/ml.hpp"

using namespace std;

const int start_scale_num = 2; //including
const int end_scale_num = 6; // excluding

const int featureDimension_hog               = 32;
const int featureDimension_hof               = 36;
const int featureDimension_mbhXY             = 64;
const int featureDimension_combined          = 132;
const int featureDimension_baseline          = 128;
//int featureDimension_mbhX = 32;
//int featureDimension_mbhY = 32;
const int MAX_FEATURES           = 5000;
const int dictionarySize                 = 400;
//const int CombinedDictionarySize         = 1200;

const int FRAME_OFFSET_TIGER              = 25;
const int NUM_POS_TRAIN_TIGER          = 223;
const int NUM_NEG_TRAIN_TIGER          = 265;
const int NUM_POS_TEST_TIGER           = 226;
const int NUM_NEG_TEST_TIGER           = 220;
const string tiger_train_pos_shotfilename = "tiger-train-30-pos-refined-shots.txt";
const string tiger_train_neg_shotfilename = "tiger-train-30-neg-refined-shots.txt";
const string tiger_test_pos_shotfilename = "tiger-test-30-pos-refined-shots.txt";
const string tiger_test_neg_shotfilename = "tiger-test-30-neg-refined-shots.txt";



const int FRAME_OFFSET_LEOPARD           = 1;
const int NUM_POS_TRAIN_LEOPARD            = 169; //350;
const int NUM_NEG_TRAIN_LEOPARD           = 169; //296;
const int NUM_POS_TEST_LEOPARD            = 242; //358;
const int NUM_NEG_TEST_LEOPARD            = 242; //301;
const string leopard_train_pos_shotfilename = "leopard-train-30-pos";
const string leopard_train_neg_shotfilename = "leopard-train-30-neg";
const string leopard_test_pos_shotfilename = "leopard-test-30-pos";
const string leopard_test_neg_shotfilename = "leopard-test-30-neg";


/*
const int FRAME_OFFSET = FRAME_OFFSET_TIGER;
const int NUM_POS_TRAIN = NUM_POS_TRAIN_TIGER;
const int NUM_NEG_TRAIN = NUM_NEG_TRAIN_TIGER;
const int NUM_POS_TEST = NUM_POS_TEST_TIGER;
const int NUM_NEG_TEST = NUM_NEG_TEST_TIGER;
const string train_pos_shotfilename = tiger_train_pos_shotfilename;
const string train_neg_shotfilename = tiger_train_neg_shotfilename;
const string test_pos_shotfilename = tiger_test_pos_shotfilename;
const string test_neg_shotfilename = tiger_test_neg_shotfilename;
*/

const int FRAME_OFFSET = FRAME_OFFSET_LEOPARD;
const int NUM_POS_TRAIN = NUM_POS_TRAIN_LEOPARD;
const int NUM_NEG_TRAIN = NUM_NEG_TRAIN_LEOPARD;
const int NUM_POS_TEST = NUM_POS_TEST_LEOPARD;
const int NUM_NEG_TEST = NUM_NEG_TEST_LEOPARD;
const string train_pos_shotfilename = leopard_train_pos_shotfilename;
const string train_neg_shotfilename = leopard_train_neg_shotfilename;
const string test_pos_shotfilename = leopard_test_pos_shotfilename;
const string test_neg_shotfilename = leopard_test_neg_shotfilename;
#endif
