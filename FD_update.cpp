#include "EBS.h"
#include "EBS.cpp"
#include "tlwekeyswitch.h"
#include "tlwekeyswitch.cpp"
#include <fstream>
#include <random>
#include <tfhe/lwe-functions.h>
#include <tfhe/lwesamples.h>
#include <tfhe/polynomials.h>
#include <tfhe/tlwe.h>
#include <tfhe/tlwe_functions.h>
#include "ProgressBar.h"
#include "ProgressBar.cpp"
#include <boost/math/constants/constants.hpp>
#include <boost/multiprecision/float128.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics.hpp>
#include <boost/accumulators/numeric/functional.hpp>
#include <new>
#include <boost/utility.hpp>

using namespace std;
using namespace boost::multiprecision;
using boost::multiprecision::cpp_dec_float_100;
using namespace boost::accumulators;


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


template< typename DEFAULT_INITIALIZABLE >
inline void clear( DEFAULT_INITIALIZABLE& object )
{
    object.DEFAULT_INITIALIZABLE::~DEFAULT_INITIALIZABLE() ;
    ::new ( boost::addressof(object) ) DEFAULT_INITIALIZABLE() ;
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
    int32_t param_num = q;

    
    int32_t max_ext = atoi(argv[2]) + 1;
    int32_t iter = atoi(argv[3]);

    std::cout << BOLDWHITE "======================== Param " << p << " =========================== \n" RESET;
    // public_keysize(paramset[q], RS[q], PM[q]);
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
    // cout << BOLDBLUE << "TLWE-to-TRLWE Keyswitch KeyGen finished!" <<"\n" RESET;
    // std::cout << BOLDWHITE "========================================================== \n" RESET;
    // std::cout << BOLDWHITE "========================================================== \n" RESET;

    std::random_device rd; // obtain a random number from hardware
    std::mt19937 gen(rd()); // seed the generator

    // ofstream myfile ("./result/HDFBS_identity4.txt");
    // ofstream fdfb ("./result/fdfb.txt");
    // ofstream tota ("./result/tota.txt");
    // ofstream comp ("./result/comp.txt");
    
    int32_t exp_param[2] = {max_ext, iter}; //  0 <= ext_factor < arg[1]  , iteration = arg[2]
    LweSample* c1 = new_LweSample(paramset[param_num]->in_out_params);
    LweSample* c2 = new_LweSample(paramset[param_num]->in_out_params);
    LweSample* res1 = new_LweSample(paramset[param_num]->in_out_params);
    LweSample* res2 = new_LweSample(paramset[param_num]->in_out_params);
    LweSample* res3 = new_LweSample(paramset[param_num]->in_out_params);
    LweSample* res4 = new_LweSample(paramset[param_num]->in_out_params);

    int32_t plain_interval = 64;
    double delta = pow(2., 32)/(plain_interval*2);
    
    auto distr = std::bind(std::uniform_int_distribution<int32_t>{std::numeric_limits<int32_t>::min(),std::numeric_limits<int32_t>::max()}, std::mt19937(std::random_device{}()));
    // auto distr = std::bind(std::uniform_int_distribution<int32_t>{-1610612736, 1610612736}, std::mt19937(std::random_device{}()));
    accumulator_set<double, stats<tag::max, tag::min, tag::mean, tag::variance> > acc1, acc2, acc3, acc4;
    for (int j = 8; j < exp_param[0]; j++) {  // ad_bit
        std::cout << BOLDWHITE "================ Param " << p << ", Ext factor " << j <<" ================== \n" RESET;
        ofstream myfile ("./result/Param_Num_"+to_string(p)+"_Extfact_"+to_string(j)+".txt");
        // ProgressBar progress(exp_param[1], "Bootstrapping...");
        for (int k = 0; k < exp_param[1]; k++) { // num_of_runs
            Torus32 mu = distr();
            lweSymEncrypt(c1, mu, paramset[param_num]->in_out_params->alpha_min, key->lwe_key);
            // lweSymEncrypt(c2, abs(mu), paramset[param_num]->in_out_params->alpha_min, key->lwe_key);
            // HDEBS(res1, key->cloud.bkFFT, j, c2, N*(1<<j));
            FDFB_EBS_Modular(res2, key->cloud.bkFFT, lwe_to_tlwekey, j, c1, PM[param_num], 2*N*(1<<j));
            TOTA_EBS_Modular(res3, key->cloud.bkFFT, j, c1, N*(1<<j));
            Comp_EBS_Modular(res4, key->cloud.bkFFT, j, c1, N*(1<<j));

            // acc1((abs(mu) - lwePhase(res1, lwe_key))/pow(2, 32));
            acc2((mu - lwePhase(res2, lwe_key))/pow(2., 32));
            acc3((mu - lwePhase(res3, lwe_key))/pow(2., 32));
            acc4((mu - lwePhase(res4, lwe_key))/pow(2., 32));

            // Torus32 mu2 = distr();
            // // Torus32 mu1 = mu2/2 + modSwitchToTorus32(1, 4);
            // double rx = ((double) mu2/delta);
            // // double ry = (double) 43*sin(rx*boost::math::double_constants::pi / 4); // exp1
            // // double ry = (double)  (40*(exp(rx/24) - exp(-rx/24))/(exp(rx/24)+exp(-rx/24)));
            // double ry = (double) 43*sin(rx*boost::math::double_constants::pi / 32); // exp1
            // // lweSymEncrypt(c1, mu1, paramset[param_num]->in_out_params->alpha_min, key->lwe_key);
            // lweSymEncrypt(c2, mu2, paramset[param_num]->in_out_params->alpha_min, key->lwe_key);
            // // HDEBS_R(res1, key->cloud.bkFFT, j, c1, plain_interval);
            // FDFB_EBS_Real(res2, key->cloud.bkFFT, lwe_to_tlwekey, j, c2, PM[param_num], plain_interval);
            // TOTA_EBS_Real(res3, key->cloud.bkFFT, j, c2, plain_interval);
            // Comp_EBS_Real(res4, key->cloud.bkFFT, j, c2, plain_interval);
            // acc2((ry - lwePhase(res2, lwe_key)/delta)/(plain_interval*2));
            // acc3((ry - lwePhase(res3, lwe_key)/delta)/(plain_interval*2));
            // acc4((ry - lwePhase(res4, lwe_key)/delta)/(plain_interval*2));
            // // hdebs += (31. - log2(abs(mu - lwePhase(res1, lwe_key))))/(exp_param[1]);
            // // fdfb += (31. - log2(abs(mu - lwePhase(res2, lwe_key))))/(exp_param[1]);
            // // tota += (31. - log2(abs(mu - lwePhase(res3, lwe_key))))/(exp_param[1]);
            // // comp += (31. - log2(abs(mu - lwePhase(res4, lwe_key))))/(exp_param[1]);
            // // ++progress;
            // myfile << rx << " " << ry << " " << (ry - lwePhase(res2, lwe_key)/delta) << " " << (ry - lwePhase(res3, lwe_key)/delta) << " " << (ry - lwePhase(res4, lwe_key)/delta) <<  "\n";
        
        
        }

        // progress.endProgressBar();
        // std::cout << "HDEBS Max :   " << boost::accumulators::max(acc1) << std::endl;
        // std::cout << "HDEBS min :   " << boost::accumulators::min(acc1) << std::endl;
        // std::cout << "HDEBS Mean :   " << boost::accumulators::mean(acc1) << std::endl;
        // std::cout << "HDEBS Variance (log2) :   " << log2(boost::accumulators::variance(acc1)) << std::endl;
        // std::cout << "HDEBS Stdev (log2) :   " << log2(boost::accumulators::variance(acc1))/2 << std::endl;
        // std::cout << "============================================================================== \n";
        std::cout << "FDFB Max :   " << boost::accumulators::max(acc2) << std::endl;
        std::cout << "FDFB min :   " << boost::accumulators::min(acc2) << std::endl;
        std::cout << "FDFB Max (log2):   " << log2(boost::accumulators::max(acc2)) << std::endl;
        std::cout << "FDFB min (abs log2):   " << log2(abs(boost::accumulators::min(acc2))) << std::endl;        
        std::cout << "FDFB Mean :   " << boost::accumulators::mean(acc2) << std::endl;
        std::cout << "FDFB Variance (log2) :   " << log2(boost::accumulators::variance(acc2)) << std::endl;
        std::cout << "FDFB Stdev (log2) :   " << log2(boost::accumulators::variance(acc2))/2 << std::endl;
        std::cout << "============================================================================== \n";
        std::cout << "TOTA Max :   " << boost::accumulators::max(acc3) << std::endl;
        std::cout << "TOTA min :   " << boost::accumulators::min(acc3) << std::endl;
        std::cout << "TOTA Max (log2):   " << log2(boost::accumulators::max(acc3)) << std::endl;
        std::cout << "TOTA min (abs log2):   " << log2(abs(boost::accumulators::min(acc3))) << std::endl;   
        std::cout << "TOTA Mean :   " << boost::accumulators::mean(acc3) << std::endl;
        std::cout << "TOTA Variance (log2) :   " << log2(boost::accumulators::variance(acc3)) << std::endl;
        std::cout << "TOTA Stdev (log2) :   " << log2(boost::accumulators::variance(acc3))/2 << std::endl;
        std::cout << "============================================================================== \n";
        std::cout << "Comp Max :   " << boost::accumulators::max(acc4) << std::endl;
        std::cout << "Comp min :   " << boost::accumulators::min(acc4) << std::endl;
        std::cout << "Comp Max (log2):   " << log2(boost::accumulators::max(acc4)) << std::endl;
        std::cout << "Comp min (abs log2):   " << log2(abs(boost::accumulators::min(acc4))) << std::endl;   
        std::cout << "Comp Mean :   " << boost::accumulators::mean(acc4) << std::endl;
        std::cout << "Comp Variance (log2) :   " << log2(boost::accumulators::variance(acc4)) << std::endl;
        std::cout << "Comp Stdev (log2) :   " << log2(boost::accumulators::variance(acc4))/2 << std::endl;
        std::cout << "============================================================================== \n";
        clear(acc1);
        clear(acc2);
        clear(acc3);
        clear(acc4);

        // LweSample* c = new_LweSample(paramset[q]->in_out_params);
        // LweSample* res = new_LweSample(paramset[q]->in_out_params);
        // lweSymEncrypt(c, 0, paramset[q]->in_out_params->alpha_min, lwe_key);
        // double duration1, duration2, duration3, duration4;
        // int32_t iter2 = 100;
        
        // // ProgressBar progress1(iter, "HDEBS Running...");
        // auto start =  chrono::steady_clock::now();
        // for (int i = 0; i < iter2; i++){
        //     HDEBS(res, key->cloud.bkFFT, j, c, N*(1<<j));
        //     // ++progress1;
        // }
        // auto finish =  chrono::steady_clock::now();
        // // progress1.endProgressBar();
        // duration1 = chrono::duration_cast<chrono::milliseconds>(finish - start).count()/((float)iter2);
        // cout << BOLDRED << "HDEBS : " << duration1 << " ms \n" RESET;

        // // // ProgressBar progress2(iter, "FDFB Running...");
        // start =  chrono::steady_clock::now();
        // for (int i = 0; i < iter2; i++){
        //     FDFB_EBS_Modular(res, key->cloud.bkFFT, lwe_to_tlwekey, j, c, PM_temp, 2*N*(1<<j));
        //     // ++progress2;
        // }
        // finish =  chrono::steady_clock::now();
        // // progress2.endProgressBar();
        // duration2 = chrono::duration_cast<chrono::milliseconds>(finish - start).count()/((float)iter2);
        // cout << BOLDGREEN << "FDFB : " << duration2 << " ms \n" RESET;

        // // ProgressBar progress3(iter, "TOTA Running...");
        // start =  chrono::steady_clock::now();
        // for (int i = 0; i < iter2; i++){
        //     TOTA_EBS_Modular(res, key->cloud.bkFFT, j, c, N*(1<<j));
        //     // ++progress3;
        // }
        // finish = chrono::steady_clock::now();
        // // progress3.endProgressBar();
        // duration3 = chrono::duration_cast<chrono::milliseconds>(finish - start).count()/((float)iter2);
        // cout << BOLDYELLOW << "TOTA : " << duration3 << " ms \n" RESET;
        
        // // ProgressBar progress4(iter, "Comp Running...");
        // start =  chrono::steady_clock::now();
        // for (int i = 0; i < iter2; i++){
        //     Comp_EBS_Modular(res, key->cloud.bkFFT, j, c, 2*N*(1<<j));
        //     // ++progress4;
        // }
        // finish =  chrono::steady_clock::now();
        // // progress4.endProgressBar();
        // duration4 = chrono::duration_cast<chrono::milliseconds>(finish - start).count()/((float)iter2);
        // cout << BOLDBLUE << "Comp : " << duration4 << " ms \n" RESET;    
        // std::cout << BOLDWHITE "========================================================== \n" RESET;
    }
    
    std::cout << BOLDWHITE "========================================================== \n" RESET;
    std::cout << BOLDWHITE "========================================================== \n" RESET;

    // for (int i = 0; i < 100000; i++){
    //     LweSample* c = new_LweSample(param->in_out_params);
    //     LweSample* res1 = new_LweSample(param->in_out_params);
    //     LweSample* res2 = new_LweSample(param->in_out_params);
    //     LweSample* res3 = new_LweSample(param->in_out_params);
    //     // LweSample* res = new_LweSample(&param->tgsw_params->tlwe_params->extracted_lweparams);
    //     // TLweSample* ks = new_TLweSample(param->tgsw_params->tlwe_params);
    //     Torus32 mu = modSwitchToTorus32(distr(gen), M);
    //     cout << distr(gen) << "\n";
    //     // TorusPolynomial* mux = new_TorusPolynomial(N);
    //     lweSymEncrypt(c, mu, param->in_out_params->alpha_min, key->lwe_key);
        
    //     for (int j = 0; j < 1; j++){
    //         // FDHBS(res1, key1->cloud.bkFFT, j, c);
    //         // FDHBS_woKS(res, key1->cloud.bkFFT, j, res1);
    //         // TOTA_EBS_Modular(res1, key->cloud.bkFFT, j, c, M);
    //         // TLweKeySwitch(ks, lwe_to_tlwekey, c);
    //         // FDFB_EBS_Modular(res2, key->cloud.bkFFT, lwe_to_tlwekey, j, c, PM_param, M);
    //         // Comp_EBS_Modular(res3, key->cloud.bkFFT, j, c, M);

    //         int32_t MU1 = lwePhase(res1, lwe_key);
    //         int32_t MU2 = lwePhase(res2, lwe_key);
    //         int32_t MU3 = lwePhase(res3, lwe_key);
    //         // int32_t Mu = lwePhase(res, ext_lwekey);
    //         // int32_t MU = lweSymDecrypt(res, ext_lwekey, M);
    //         int32_t Mu1 = MU1 - mu;
    //         int32_t Mu2 = MU2 - mu;
    //         int32_t Mu3 = MU3 - mu;
            
    //         // tLweSymDecrypt(mux, ks, tlwe_key, M);
    //         // int32_t Mu = modSwitchFromTorus32(mux->coefsT[52], M);
            
    //         // cout << BOLDMAGENTA << Mu << " " RESET;
    //         // cout << i << "\n";
    //         // tota << Mu1 << "\n";
    //         // fdfb << Mu2 << "\n";
    //         // comp << Mu3 << "\n";
    //         // myfile << Mu << " ";
    //     }
    //     cout << " \n";
    //     // myfile << "\n";
    // }

    // for (int i = 0; i < M; i++){
    //     LweSample* c = new_LweSample(param->in_out_params);
    //     LweSample* res1 = new_LweSample(param->in_out_params);
    //     LweSample* res2 = new_LweSample(param->in_out_params);
    //     LweSample* res3 = new_LweSample(param->in_out_params);
    //     // LweSample* res = new_LweSample(&param->tgsw_params->tlwe_params->extracted_lweparams);
    //     // TLweSample* ks = new_TLweSample(param->tgsw_params->tlwe_params);
    //     Torus32 mu = modSwitchToTorus32(i+M/2, M);
    //     TorusPolynomial* mux = new_TorusPolynomial(N);
    //     lweSymEncrypt(c, mu, param->in_out_params->alpha_min, key->lwe_key);
        
    //     for (int j = 0; j < 1; j++){
    //         // FDHBS(res1, key1->cloud.bkFFT, j, c);
    //         // FDHBS_woKS(res, key1->cloud.bkFFT, j, res1);
    //         // TOTA_EBS_Modular(res, key->cloud.bkFFT, j, c, M);
    //         // TLweKeySwitch(ks, lwe_to_tlwekey, c);
    //         // FDFB_EBS_Modular(res, key->cloud.bkFFT, lwe_to_tlwekey, j, c, PM_param, M);
    //         Comp_EBS_Modular(res1, key->cloud.bkFFT, j, c, M);

    //         int32_t MU = lweSymDecrypt(res1, lwe_key, M);
    //         // int32_t Mu = lwePhase(res, ext_lwekey);
    //         // int32_t MU = lweSymDecrypt(res, ext_lwekey, M);
    //         int32_t Mu = modSwitchFromTorus32(MU, M);
            
    //         // tLweSymDecrypt(mux, ks, tlwe_key, M);
    //         // int32_t Mu = modSwitchFromTorus32(mux->coefsT[52], M);
            
    //         cout << BOLDMAGENTA << Mu << " " RESET;
    //         // myfile << Mu << " ";
    //     }
    //     cout << " \n";
    //     // myfile << "\n";
    // }

    // myfile.close();
    // fdfb.close();
    // tota.close();
    // comp.close();

}