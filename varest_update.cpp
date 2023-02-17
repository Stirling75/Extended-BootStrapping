#include "varEBS.cpp"
#include "tlwekeyswitch.h"
#include "tlwekeyswitch.cpp"
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <fstream>
#include <random>
#include <tfhe/lwe-functions.h>
#include <tfhe/lwesamples.h>
#include <tfhe/polynomials.h>
#include <tfhe/tfhe_core.h>
#include <tfhe/tfhe_gate_bootstrapping_functions.h>
#include <tfhe/tlwe.h>
#include <tfhe/tfhe_garbage_collector.h>
#include <tfhe/tlwe_functions.h>
#include <chrono>
#include <unistd.h>
#include <sstream>

using namespace std;

TFheGateBootstrappingParameterSet * bit80_param1() {
    static const int n = 750;
    static const int N = 1024;
    static const int k = 1;
    static const double max_stdev = pow(2., -11);

    static const int bk_Bgbit    = 4;  
    static const int bk_l        = 7;
    static const double bk_stdev = pow(2., -29.3); 
    // static const double bk_stdev = 0; 

    static const int ks_basebit  = 8; 
    static const int ks_length   = 3;
    static const double ks_stdev = pow(2., -21.2);
    // static const double ks_stdev = 0;

    LweParams  *params_in    = new_LweParams (n, ks_stdev, max_stdev);
    TLweParams *params_accum = new_TLweParams(N, k, bk_stdev, max_stdev);
    TGswParams *params_bk    = new_TGswParams(bk_l, bk_Bgbit, params_accum);

    TfheGarbageCollector::register_param(params_in);
    TfheGarbageCollector::register_param(params_accum);
    TfheGarbageCollector::register_param(params_bk);

    return new TFheGateBootstrappingParameterSet(ks_length, ks_basebit, params_in, params_bk);
}

TFheGateBootstrappingParameterSet * bit80_param1_2048() {
    static const int n = 750;
    static const int N = 2048;
    static const int k = 1;
    static const double max_stdev = pow(2., -11);

    static const int bk_Bgbit    = 4;  
    static const int bk_l        = 7;
    static const double bk_stdev = pow(2., -32); 
    // static const double bk_stdev = 0; 

    static const int ks_basebit  = 8; 
    static const int ks_length   = 3;
    static const double ks_stdev = pow(2., -21.2);
    // static const double ks_stdev = 0;

    LweParams  *params_in    = new_LweParams (n, ks_stdev, max_stdev);
    TLweParams *params_accum = new_TLweParams(N, k, bk_stdev, max_stdev);
    TGswParams *params_bk    = new_TGswParams(bk_l, bk_Bgbit, params_accum);

    TfheGarbageCollector::register_param(params_in);
    TfheGarbageCollector::register_param(params_accum);
    TfheGarbageCollector::register_param(params_bk);

    return new TFheGateBootstrappingParameterSet(ks_length, ks_basebit, params_in, params_bk);
}

TFheGateBootstrappingParameterSet * bit80_param1_4096() {
    static const int n = 750;
    static const int N = 4096;
    static const int k = 1;
    static const double max_stdev = pow(2., -11);

    static const int bk_Bgbit    = 4;  
    static const int bk_l        = 7;
    static const double bk_stdev = pow(2., -32); 
    // static const double bk_stdev = 0; 

    static const int ks_basebit  = 8; 
    static const int ks_length   = 3;
    static const double ks_stdev = pow(2., -21.2);
    // static const double ks_stdev = 0;

    LweParams  *params_in    = new_LweParams (n, ks_stdev, max_stdev);
    TLweParams *params_accum = new_TLweParams(N, k, bk_stdev, max_stdev);
    TGswParams *params_bk    = new_TGswParams(bk_l, bk_Bgbit, params_accum);

    TfheGarbageCollector::register_param(params_in);
    TfheGarbageCollector::register_param(params_accum);
    TfheGarbageCollector::register_param(params_bk);

    return new TFheGateBootstrappingParameterSet(ks_length, ks_basebit, params_in, params_bk);
}

TFheGateBootstrappingParameterSet * bit80_param2() {
    static const int n = 900;
    static const int N = 2048;
    static const int k = 1;
    static const double max_stdev = pow(2., -11);

    static const int bk_Bgbit    = 4;  
    static const int bk_l        = 7;
    static const double bk_stdev = pow(2., -32); 
    // static const double bk_stdev = 0; 

    static const int ks_basebit  = 6; 
    static const int ks_length   = 5;
    static const double ks_stdev = pow(2., -25.7);
    // static const double ks_stdev = 0;

    LweParams  *params_in    = new_LweParams (n, ks_stdev, max_stdev);
    TLweParams *params_accum = new_TLweParams(N, k, bk_stdev, max_stdev);
    TGswParams *params_bk    = new_TGswParams(bk_l, bk_Bgbit, params_accum);

    TfheGarbageCollector::register_param(params_in);
    TfheGarbageCollector::register_param(params_accum);
    TfheGarbageCollector::register_param(params_bk);

    return new TFheGateBootstrappingParameterSet(ks_length, ks_basebit, params_in, params_bk);
}


TFheGateBootstrappingParameterSet * bit128_param3() {
    static const int n = 670;
    static const int N = 1024;
    static const int k = 1;
    static const double max_stdev = pow(2., -11);

    static const int bk_Bgbit    = 3;  
    static const int bk_l        = 8;
    static const double bk_stdev = pow(2., -20.1); 
    // static const double bk_stdev = 0; 

    static const int ks_basebit  = 5; 
    static const int ks_length   = 3;
    static const double ks_stdev = pow(2., -12.4);
    // static const double ks_stdev = 0;

    LweParams  *params_in    = new_LweParams (n, ks_stdev, max_stdev);
    TLweParams *params_accum = new_TLweParams(N, k, bk_stdev, max_stdev);
    TGswParams *params_bk    = new_TGswParams(bk_l, bk_Bgbit, params_accum);

    TfheGarbageCollector::register_param(params_in);
    TfheGarbageCollector::register_param(params_accum);
    TfheGarbageCollector::register_param(params_bk);

    return new TFheGateBootstrappingParameterSet(ks_length, ks_basebit, params_in, params_bk);
}

TFheGateBootstrappingParameterSet * bit128_param3_2048() {
    static const int n = 670;
    static const int N = 2048;
    static const int k = 1;
    static const double max_stdev = pow(2., -11);

    static const int bk_Bgbit    = 3;  
    static const int bk_l        = 8;
    static const double bk_stdev = pow(2., -32); 
    // static const double bk_stdev = 0; 

    static const int ks_basebit  = 5; 
    static const int ks_length   = 3;
    static const double ks_stdev = pow(2., -12.4);
    // static const double ks_stdev = 0;

    LweParams  *params_in    = new_LweParams (n, ks_stdev, max_stdev);
    TLweParams *params_accum = new_TLweParams(N, k, bk_stdev, max_stdev);
    TGswParams *params_bk    = new_TGswParams(bk_l, bk_Bgbit, params_accum);

    TfheGarbageCollector::register_param(params_in);
    TfheGarbageCollector::register_param(params_accum);
    TfheGarbageCollector::register_param(params_bk);

    return new TFheGateBootstrappingParameterSet(ks_length, ks_basebit, params_in, params_bk);
}

TFheGateBootstrappingParameterSet * bit128_param3_4096() {
    static const int n = 670;
    static const int N = 4096;
    static const int k = 1;
    static const double max_stdev = pow(2., -11);

    static const int bk_Bgbit    = 3;  
    static const int bk_l        = 8;
    static const double bk_stdev = pow(2., -32); 
    // static const double bk_stdev = 0; 

    static const int ks_basebit  = 5; 
    static const int ks_length   = 3;
    static const double ks_stdev = pow(2., -12.4);
    // static const double ks_stdev = 0;

    LweParams  *params_in    = new_LweParams (n, ks_stdev, max_stdev);
    TLweParams *params_accum = new_TLweParams(N, k, bk_stdev, max_stdev);
    TGswParams *params_bk    = new_TGswParams(bk_l, bk_Bgbit, params_accum);

    TfheGarbageCollector::register_param(params_in);
    TfheGarbageCollector::register_param(params_accum);
    TfheGarbageCollector::register_param(params_bk);

    return new TFheGateBootstrappingParameterSet(ks_length, ks_basebit, params_in, params_bk);
}

TFheGateBootstrappingParameterSet * bit128_param4() {
    static const int n = 1300;
    static const int N = 2048;
    static const int k = 1;
    static const double max_stdev = pow(2., -11);

    static const int bk_Bgbit    = 4;  
    static const int bk_l        = 7;
    static const double bk_stdev = pow(2., -32); 
    // static const double bk_stdev = 0; 

    static const int ks_basebit  = 6; 
    static const int ks_length   = 5;
    static const double ks_stdev = pow(2., -26.1);
    // static const double ks_stdev = 0;

    LweParams  *params_in    = new_LweParams (n, ks_stdev, max_stdev);
    TLweParams *params_accum = new_TLweParams(N, k, bk_stdev, max_stdev);
    TGswParams *params_bk    = new_TGswParams(bk_l, bk_Bgbit, params_accum);

    TfheGarbageCollector::register_param(params_in);
    TfheGarbageCollector::register_param(params_accum);
    TfheGarbageCollector::register_param(params_bk);

    return new TFheGateBootstrappingParameterSet(ks_length, ks_basebit, params_in, params_bk);
}


int main (int argc, char **argv){
    int32_t RS1[2] = {5, 6}; // Basebit_RS, t_RS
    int32_t RS2[2] = {5, 6}; // Basebit_RS, t_RS
    int32_t RS3[2] = {5, 6}; // Basebit_RS, t_RS
    int32_t RS4[2] = {6, 4}; // Basebit_RS, t_RS
    int32_t RS5[2] = {6, 4}; // Basebit_RS, t_RS
    int32_t RS6[2] = {6, 4}; // Basebit_RS, t_RS
    int32_t RS7[2] = {6, 4}; // Basebit_RS, t_RS
    int32_t RS8[2] = {5, 6}; // Basebit_RS, t_RS


    int32_t PM1[2] = {5, 5}; // Basebit_PM, l_PM
    int32_t PM2[2] = {5, 5}; // Basebit_PM, l_PM
    int32_t PM3[2] = {5, 5}; // Basebit_PM, l_PM
    int32_t PM4[2] = {5, 5}; // Basebit_PM, l_PM
    int32_t PM5[2] = {5, 5}; // Basebit_PM, l_PM
    int32_t PM6[2] = {5, 5}; // Basebit_PM, l_PM
    int32_t PM7[2] = {5, 5}; // Basebit_PM, l_PM
    int32_t PM8[2] = {5, 5}; // Basebit_PM, l_PM
    TFheGateBootstrappingParameterSet* param1 = bit80_param1();
    TFheGateBootstrappingParameterSet* param2 = bit80_param1_2048();
    TFheGateBootstrappingParameterSet* param3 = bit80_param1_4096();
    TFheGateBootstrappingParameterSet* param4 = bit80_param2();
    TFheGateBootstrappingParameterSet* param5 = bit128_param3();
    TFheGateBootstrappingParameterSet* param6 = bit128_param3_2048();
    TFheGateBootstrappingParameterSet* param7 = bit128_param3_4096();
    TFheGateBootstrappingParameterSet* param8 = bit128_param4();

    vector<TFheGateBootstrappingParameterSet*> paramset = {param1, param2, param3, param4, param5, param6, param7, param8};
    vector<int*> RS = {RS1, RS2, RS3, RS4, RS5, RS6, RS7, RS8};
    vector<int*> PM = {PM1, PM2, PM3, PM4, PM5, PM6, PM7, PM8};
    int p = atoi(argv[1]);
    int q = p - 1;

    int32_t iter = 100;
    int32_t ext_factor = atoi(argv[2]);

    std::cout << BOLDWHITE "======================== Param " << p << " =========================== \n" RESET;
    param_quality_woMspace(paramset[q], RS[q], PM[q]);
    TFheGateBootstrappingSecretKeySet* key = new_random_gate_bootstrapping_secret_keyset(paramset[q]);
    const LweKey* lwe_key = key->lwe_key;
    const TLweKey* tlwe_key = &key->tgsw_key->tlwe_key;
    LweKey* ext_lwekey = new_LweKey(&paramset[q]->tgsw_params->tlwe_params->extracted_lweparams);
    tLweExtractKey(ext_lwekey, tlwe_key);
    int32_t N = tlwe_key->params->N;
    int32_t PM_temp[2] = {PM[q][0], PM[q][1]};
    

    TLweKeySwitchKey* lwe_to_tlwekey = new_TLweKeySwitchKey(paramset[q]->in_out_params->n,
        RS[q][1],
        RS[q][0],
        paramset[q]->tgsw_params->tlwe_params);

    TLweCreateKeySwitchKey(lwe_to_tlwekey, lwe_key, tlwe_key);

    prec_est(key);
    std_est(key, lwe_to_tlwekey, RS[q], 0);
    fd_est(key, lwe_to_tlwekey, PM[q], 0);
    fd_est_real(key, lwe_to_tlwekey, PM[q], 0, 64);
}