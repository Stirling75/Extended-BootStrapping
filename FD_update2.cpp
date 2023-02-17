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

// void param_quality_file(TFheGateBootstrappingParameterSet* param, int32_t RS_param[2], int32_t PM_param[2], int32_t msize, std::ofstream& file) {
//     cpp_dec_float_100 n = param->in_out_params->n;
//     cpp_dec_float_100 N = param->tgsw_params->tlwe_params->N;
//     cpp_dec_float_100 k = param->tgsw_params->tlwe_params->k;
//     cpp_dec_float_100 p = msize;

//     cpp_dec_float_100 bk_Bg = param->tgsw_params->Bg;
//     cpp_dec_float_100 bk_halfBg = param->tgsw_params->halfBg;  
//     cpp_dec_float_100 bk_l = param->tgsw_params->l;
//     cpp_dec_float_100 bk_stdev = param->tgsw_params->tlwe_params->alpha_min; 
//     // static const double bk_stdev = 0; 

//     cpp_dec_float_100 ks_Bg = 1<< param->ks_basebit; 
//     cpp_dec_float_100 ks_l  = param->ks_t;
//     cpp_dec_float_100 ks_stdev = param->in_out_params->alpha_min;

//     cpp_dec_float_100 rs_Bg = 1<<RS_param[0];
//     cpp_dec_float_100 rs_l = RS_param[1];

//     cpp_dec_float_100 pm_Bg = 1<<PM_param[0];
//     cpp_dec_float_100 pm_l = PM_param[1];

//     cpp_dec_float_100 log2_VMS = log2(n+1) - 4 * log2(3) - 2 * log2(N);
//     cpp_dec_float_100 V_MS = pow(2., log2_VMS);
//     cpp_dec_float_100 log2_stdMS = log2_VMS/2; 

//     cpp_dec_float_100 BR_1 = log2(n) + log2(N) + log2(k+1) + log2(bk_l) + 2*log2(bk_halfBg) + 2*log2(bk_stdev);
//     cpp_dec_float_100 BR_2 = log2(n) + log2(1 + k*N) - 2 * bk_l * log2(bk_Bg) - 2;
//     cpp_dec_float_100 V_BR = pow(2., BR_1) + pow(2., BR_2);
//     cpp_dec_float_100 log2_VBR = log2(V_BR); 
//     cpp_dec_float_100 KS_1 = log2(k) + log2(N) + log2(ks_l) + 2 * log2(ks_stdev);
//     cpp_dec_float_100 KS_2 = log2(k) -2 * ks_l * log2(ks_Bg) + log2(N) - 2 - log2(3);
//     cpp_dec_float_100 V_KS = pow(2., KS_1) + pow(2., KS_2);
//     cpp_dec_float_100 log2_VKS = log2(V_KS);
//     cpp_dec_float_100 V_BS = V_BR + V_KS;
//     cpp_dec_float_100 log2_VBS = log2(V_BS);
//     cpp_dec_float_100 RS_1 = log2(n) + log2(rs_l) + 2 * log2(bk_stdev);
//     cpp_dec_float_100 RS_2 = -2 * rs_l * log2(rs_Bg) + log2(n) - 2 - log2(3);
//     cpp_dec_float_100 V_RS = pow(2., RS_1) + pow(2., RS_2);
//     cpp_dec_float_100 log2_VRS = log2(V_RS);
//     cpp_dec_float_100 FDFBACC_1 = log2(N) + log2(pm_l) + 2*log2(pm_Bg) + log2(V_RS + V_BS) - 2;
//     cpp_dec_float_100 FDFBACC_2 = log2(1 + k*N) - 2*pm_l*log2(pm_Bg)- 2;
//     cpp_dec_float_100 V_FDFBACC = pow(2., FDFBACC_1) + pow(2., FDFBACC_2);
//     cpp_dec_float_100 log2_VFDFBACC = log2(V_FDFBACC);

//     cpp_dec_float_100 V_ct = pow(ks_stdev, 2);
//     // cpp_dec_float_100 V_ct = 2*V_BS;
    
//     // We assume V_ct = V_TLWE
//     cpp_dec_float_100 log2_HDFB = log2(erf(1/(4*p*sqrt(2)*sqrt(V_MS+V_ct)))) + log2(erf(1/(4*p*sqrt(2)*sqrt(V_BS))));
//     cpp_dec_float_100 log2err_HDFB = log2(1 - pow(2., log2_HDFB));

//     cpp_dec_float_100 log2_FDFB = log2(erf(1/(2*p*sqrt(2)*sqrt(V_ct+V_MS)))) + log2(erf(1/(2*p*sqrt(2)*sqrt(V_FDFBACC + V_BS))));
//     cpp_dec_float_100 log2err_FDFB = log2(1 - pow(2., log2_FDFB));

//     cpp_dec_float_100 log2_TOTA = log2(erf(1/(2*p*sqrt(2)*sqrt(V_ct + 4*V_MS)))) + log2(erf(1/(2*p*sqrt(2)*sqrt(V_ct + V_BS + 5*V_MS)))) + log2(erf(1/(2*p*sqrt(2)*sqrt(V_BS))));
//     cpp_dec_float_100 log2err_TOTA = log2(1 - pow(2., log2_TOTA));

//     cpp_dec_float_100 log2_Comp = log2(erf(1/(2*p*sqrt(2)*sqrt(V_ct + V_MS)))) + 2*log2(erf(1/(2*p*sqrt(2)*sqrt(V_BS + V_MS)))) + log2(erf(1/(2*p*sqrt(2)*sqrt(2*V_BS))));
//     cpp_dec_float_100 log2err_Comp = log2(1 - pow(2., log2_Comp));

//     // std::cout << BOLDWHITE "========================================================== \n" RESET;
//     // std::cout << "    Variance of BlindRotate : 2^{" << setprecision(numeric_limits<float>::digits10) << log2_VBR <<"} \n";
//     // std::cout << "      Variance of KeySwitch : 2^{" << setprecision(numeric_limits<float>::digits10) << log2_VKS <<"} \n";
//     // std::cout << "  Variance of Bootstrapping : 2^{" << setprecision(numeric_limits<float>::digits10) << log2_VBS << "} \n";
//     // std::cout << " Variance of LWE-to-TLWE KS : 2^{" << setprecision(numeric_limits<float>::digits10) << log2_VRS << "} \n";
//     // std::cout << "        Variance of FDFBACC : 2^{" << setprecision(numeric_limits<float>::digits10) << log2_VFDFBACC << "} \n";
//     std::cout << BOLDWHITE "========================================================== \n" RESET;
//     std::cout << BOLDYELLOW "    Stdev of Discretization : 2^{" << setprecision(numeric_limits<float>::digits10) << log2_stdMS <<"} \n" RESET;
//     std::cout << "       Stdev of BlindRotate : 2^{" << setprecision(numeric_limits<float>::digits10) << log2_VBR/2 <<"} \n";
//     std::cout << "         Stdev of KeySwitch : 2^{" << setprecision(numeric_limits<float>::digits10) << log2_VKS/2 <<"} \n";
//     std::cout << "     Stdev of Bootstrapping : 2^{" << setprecision(numeric_limits<float>::digits10) << log2_VBS/2 << "} \n";
//     std::cout << "   Stdev of TRLWE Keyswitch : 2^{" << setprecision(numeric_limits<float>::digits10) << log2_VRS/2 << "} \n";
//     std::cout << "          Stdev of FDFB-ACC : 2^{" << setprecision(numeric_limits<float>::digits10) << log2_VFDFBACC/2 << "} \n";
//     std::cout << BOLDWHITE "========================================================== \n" RESET;
//     std::cout << "       Precision estimated with 4-sigma (99.9936%) \n";
//     std::cout << "    Precision of output after BlindRotate : " << BOLDGREEN << setprecision(numeric_limits<float>::digits10) << (int) floor(32 - (32 + log2_VBR/2 + 3)) << " Bits \n" RESET;
//     std::cout << "      Precision of output after Bootstrap : " << BOLDRED << setprecision(numeric_limits<float>::digits10) << (int) floor(32 - (32 + log2_VBS/2 + 3)) << " Bits \n" RESET; 
//     std::cout << "                      Desired value for k : " << BOLDCYAN << setprecision(numeric_limits<float>::digits10) << max(0, (int) ceil(log2_stdMS - log2_VBS/2)) << " ~ " << max(0, (int) ceil(log2_stdMS - log2_VBR/2)) << "\n" RESET;
//     std::cout << BOLDWHITE "========================================================== \n" RESET;
//     std::cout << BOLDWHITE "With Message space p = " << msize << " and extension factor nu = 0, \n" RESET;
//     std::cout << BOLDRED "Log2 error rate for HDFB <= " << setprecision(numeric_limits<float>::digits10) << log2err_HDFB << "\n" RESET;
//     std::cout << BOLDGREEN "Log2 error rate for FDFB <= " << setprecision(numeric_limits<float>::digits10) << log2err_FDFB << "\n" RESET;
//     std::cout << BOLDYELLOW "Log2 error rate for TOTA <= " << setprecision(numeric_limits<float>::digits10) << log2err_TOTA << "\n" RESET;
//     std::cout << BOLDBLUE "Log2 error rate for Comp <= " << setprecision(numeric_limits<float>::digits10) << log2err_Comp << "\n" RESET; 
//     std::cout << BOLDWHITE "========================================================== \n" RESET;
//     file << setprecision(numeric_limits<float>::digits10) << log2_stdMS << " & " << log2_VBR/2 << " & " << log2_VKS/2 << " & " << log2_VBS/2 << " & " << log2_VRS/2 << " & " << log2_VFDFBACC/2 << " \\\\ \\hline \n";
//     // FDB_quality_f100(V_ct, V_MS, V_BS, V_FDFBACC, p, 1);
//     // FDB_quality_f100(V_ct, V_MS, V_BS, V_FDFBACC, p, 2);
//     // FDB_quality_f100(V_ct, V_MS, V_BS, V_FDFBACC, p, 3);
//     // FDB_quality_f100(V_ct, V_MS, V_BS, V_FDFBACC, p, 4);
//     // FDB_quality_f100(V_ct, V_MS, V_BS, V_FDFBACC, p, 5);
//     // FDB_quality_f100(V_ct, V_MS, V_BS, V_FDFBACC, p, 6);
//     // FDB_quality_f100(V_ct, V_MS, V_BS, V_FDFBACC, p, 7);
// }

// void public_keysize(TFheGateBootstrappingParameterSet* param, int32_t RS_param[2], int32_t PM_param[2]) {
//     cpp_dec_float_100 n = param->in_out_params->n;
//     cpp_dec_float_100 N = param->tgsw_params->tlwe_params->N;
//     cpp_dec_float_100 k = param->tgsw_params->tlwe_params->k;

//     cpp_dec_float_100 bk_Bg = param->tgsw_params->Bg;
//     cpp_dec_float_100 bk_l = param->tgsw_params->l;

//     cpp_dec_float_100 ks_Bg = 1<< param->ks_basebit; 
//     cpp_dec_float_100 ks_l  = param->ks_t;

//     cpp_dec_float_100 rs_Bg = 1<<RS_param[0];
//     cpp_dec_float_100 rs_l = RS_param[1];

//     cpp_dec_float_100 pm_Bg = 1<<PM_param[0];
//     cpp_dec_float_100 pm_l = PM_param[1];

//     cpp_dec_float_100 TLWE = sizeof(int32_t) * (n + 1); // byte
//     cpp_dec_float_100 TRLWE = sizeof(int32_t) * (k + 1) * N; // byte
//     cpp_dec_float_100 TRGSW = TRLWE * bk_l * (k + 1); // byte

//     cout << "   TLWE : " << TLWE / 1000 << " kB \n";
//     cout << "  TRLWE : " << TRLWE / 1000 << " kB \n";
//     cout << "  TRGSW : " << TRGSW / 1000 << " kB \n";


//     cpp_dec_float_100 BK = TRGSW * n;
//     cpp_dec_float_100 KS = TLWE * N * ks_l * ks_Bg * k;
//     cpp_dec_float_100 RS = TRLWE * n * rs_l * rs_Bg;
//     // cout << "===================================  \n";
//     // cout << "    BSK : " << BK / 1000000 << " MB \n";
//     // cout << "    KSK : " << KS / 1000000 << " MB \n";
//     // cout << "    RSK : " << RS / 1000000 << " MB \n";
//     // cout << BOLDWHITE "Public Key in total : " << (BK+KS+RS)/1000000 << " MB \n" RESET;

//     cout << "    BSK : " << BK / 1000000000 << " GB \n";
//     cout << "    KSK : " << KS / 1000000000 << " GB \n";
//     cout << "    RSK : " << RS / 1000000000 << " GB \n";
//     cout << BOLDWHITE "Public Key in total : " << (BK+KS+RS)/1000000000 << " GB \n" RESET;
// }

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
    // LweSample* c1 = new_LweSample(paramset[param_num]->in_out_params);
    // LweSample* c2 = new_LweSample(paramset[param_num]->in_out_params);
    // LweSample* res1 = new_LweSample(paramset[param_num]->in_out_params);
    // LweSample* res2 = new_LweSample(paramset[param_num]->in_out_params);
    // LweSample* res3 = new_LweSample(paramset[param_num]->in_out_params);
    // LweSample* res4 = new_LweSample(paramset[param_num]->in_out_params);
    
    auto distr = std::bind(std::uniform_int_distribution<int32_t>{-1610612736, 1610612736}, std::mt19937(std::random_device{}()));
    accumulator_set<double, stats<tag::max, tag::min, tag::mean, tag::variance> > acc1, acc2, acc3, acc4;
    for (int j = 0; j < exp_param[0]; j++) {  // ad_bit
        std::cout << BOLDWHITE "================ Param " << p << ", Ext factor " << j <<" ================== \n" RESET;

        LweSample* c = new_LweSample(paramset[q]->in_out_params);
        LweSample* res = new_LweSample(paramset[q]->in_out_params);
        lweSymEncrypt(c, 0, paramset[q]->in_out_params->alpha_min, lwe_key);
        double duration1, duration2, duration3, duration4;
        
        auto start =  chrono::steady_clock::now();
        for (int i = 0; i < iter; i++){
            HDEBS(res, key->cloud.bkFFT, j, c, N*(1<<j));
        }
        auto finish =  chrono::steady_clock::now();
        duration1 = chrono::duration_cast<chrono::milliseconds>(finish - start).count()/((float)iter);
        cout << BOLDRED << "HDEBS : " << duration1 << " ms \n" RESET;

        start =  chrono::steady_clock::now();
        for (int i = 0; i < iter; i++){
            FDFB_EBS_Modular(res, key->cloud.bkFFT, lwe_to_tlwekey, j, c, PM_temp, 2*N*(1<<j));
        }
        finish =  chrono::steady_clock::now();
        duration2 = chrono::duration_cast<chrono::milliseconds>(finish - start).count()/((float)iter);
        cout << BOLDGREEN << "FDFB : " << duration2 << " ms \n" RESET;

        start =  chrono::steady_clock::now();
        for (int i = 0; i < iter; i++){
            TOTA_EBS_Modular(res, key->cloud.bkFFT, j, c, N*(1<<j));
        }
        finish = chrono::steady_clock::now();
        duration3 = chrono::duration_cast<chrono::milliseconds>(finish - start).count()/((float)iter);
        cout << BOLDYELLOW << "TOTA : " << duration3 << " ms \n" RESET;
        
        start =  chrono::steady_clock::now();
        for (int i = 0; i < iter; i++){
            Comp_EBS_Modular(res, key->cloud.bkFFT, j, c, N*(1<<j));
        }
        finish =  chrono::steady_clock::now();
        duration4 = chrono::duration_cast<chrono::milliseconds>(finish - start).count()/((float)iter);
        cout << BOLDBLUE << "Comp : " << duration4 << " ms \n" RESET;    
        std::cout << BOLDWHITE "========================================================== \n" RESET;
    }
    
    std::cout << BOLDWHITE "========================================================== \n" RESET;
    std::cout << BOLDWHITE "========================================================== \n" RESET;
}