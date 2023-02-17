#ifndef EBS_H
#include <tfhe/lagrangehalfc_arithmetic.h>
#include <tfhe/lwe-functions.h>
#include <tfhe/lwekey.h>
#include <tfhe/lwesamples.h>
#include <tfhe/numeric_functions.h>
#include <tfhe/polynomials.h>
#include <tfhe/polynomials_arithmetic.h>
#include <tfhe/tfhe.h>
#include <tfhe/tfhe_core.h>
#include <tfhe/tfhe_io.h>
#include <math.h>
#include <random>
#include <cstdlib>
#include <iostream>
#include <cassert>
#include <limits.h>
#include <tfhe/tgsw_functions.h>
#include <tfhe/tgsw.h>
#include <tfhe/tlwe.h>
#include <tfhe/tlwe_functions.h>
#include <tfhe/tlwe.h>
#include <tfhe/tfhe_garbage_collector.h>
#include "tlwekeyswitch.h"
#include <chrono>
#include <boost/multiprecision/float128.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>
#define RESET   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */
#define BOLDBLACK   "\033[1m\033[30m"      /* Bold Black */
#define BOLDRED     "\033[1m\033[31m"      /* Bold Red */
#define BOLDGREEN   "\033[1m\033[32m"      /* Bold Green */
#define BOLDYELLOW  "\033[1m\033[33m"      /* Bold Yellow */
#define BOLDBLUE    "\033[1m\033[34m"      /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m"      /* Bold Magenta */
#define BOLDCYAN    "\033[1m\033[36m"      /* Bold Cyan */
#define BOLDWHITE   "\033[1m\033[37m"      /* Bold White */
#define EBS_H


using namespace std;
using boost::multiprecision::cpp_dec_float_100;


void tLweArrayMulByXaiMinusOne(TLweSample *result, int32_t ai, const TLweSample *bk, int32_t k2, const TLweParams *params);
void ext_MuxRotate_FFT(TLweSample *result, const TLweSample *accum, const TGswSampleFFT *bki, const int32_t barai, int32_t k2, const TGswParams *bk_params);

void ext_blindRotate_FFT(TLweSample *accum, const TGswSampleFFT *bkFFT, const int32_t *bara, const int32_t n, const int32_t k2, const TGswParams *bk_params);
void torusPolynomialArrayMulByXai(TorusPolynomial *result, int32_t k2, int32_t a, const TorusPolynomial *source);
void ext_blindRotateAndExtract_FFT(LweSample *result, const TorusPolynomial *v, const TGswSampleFFT *bk, const int32_t barb,const int32_t *bara, const int32_t n, const int32_t ad_bit, const TGswParams *bk_params);

// HDEBS
void HDEBS_Mfunceval_woKS_FFT(LweSample *result, const LweBootstrappingKeyFFT *bk, int32_t ad_bit, const LweSample *x, const int32_t msize);
void HDEBS(LweSample *result, const LweBootstrappingKeyFFT *bk, int32_t ad_bit, const LweSample *x, const int32_t msize);
void HDEBS_Rfunceval_woKS_FFT(LweSample *result, const LweBootstrappingKeyFFT *bk, int32_t ad_bit, const LweSample *x, int32_t interval);
void HDEBS_R(LweSample *result, const LweBootstrappingKeyFFT *bk, int32_t ad_bit, const LweSample *x, const int32_t msize);

// FDFB
void FDFB_signeval_FFT(TLweSample *result, const LweBootstrappingKeyFFT *bk, TLweKeySwitchKey* rks, int32_t ad_bit, const LweSample *x, int32_t PM_param[2]);
void FDFB_PubMux(TLweSample* ACC, TLweSample* TGSWp, const LweBootstrappingKeyFFT *bk, int32_t ad_bit, int32_t PM_param[2], int32_t msize, int32_t barb);
void FDFB_PubMux_Real(TLweSample* ACC, TLweSample* TGSWp, const LweBootstrappingKeyFFT *bk, int32_t ad_bit, int32_t PM_param[2], int32_t interval, int32_t barb);
void FDFB_BRnSE(LweSample* result, TLweSample* ACC, const LweBootstrappingKeyFFT *bk, int32_t ad_bit, const LweSample *x);
void FDFB_EBS_Modular(LweSample *result, const LweBootstrappingKeyFFT *bk, TLweKeySwitchKey* rks,  int32_t ad_bit, const LweSample *x, int32_t PM_param[2], const int32_t msize);
void FDFB_EBS_Real(LweSample *result, const LweBootstrappingKeyFFT *bk, TLweKeySwitchKey* rks,  int32_t ad_bit, const LweSample *x, int32_t PM_param[2], int32_t interval);

// TOTA
void TOTA_signeval_woKS_FFT(LweSample *result, const LweBootstrappingKeyFFT *bk, int32_t ad_bit, const LweSample *x);
void TOTA_Rfunceval_woKS_FFT(LweSample *result, const LweBootstrappingKeyFFT *bk, int32_t ad_bit, const LweSample *x, const LweSample *signx, int32_t interval);
void TOTA_Mfunceval_woKS_FFT(LweSample *result, const LweBootstrappingKeyFFT *bk, int32_t ad_bit, const LweSample *x, const LweSample *signx, const int32_t msize);
void TOTA_woKS_Real(LweSample *result, const LweBootstrappingKeyFFT *bk, int32_t ad_bit, const LweSample *x, int32_t interval );
void TOTA_EBS_Real(LweSample *result, const LweBootstrappingKeyFFT *bk, int32_t ad_bit, const LweSample *x, int32_t interval);
void TOTA_woKS_Modular(LweSample *result, const LweBootstrappingKeyFFT *bk, int32_t ad_bit, const LweSample *x, const int32_t msize);
void TOTA_EBS_Modular(LweSample *result, const LweBootstrappingKeyFFT *bk, int32_t ad_bit, const LweSample *x, const int32_t msize);

// Comp
void Comp_odd_h1(LweSample *result, const LweBootstrappingKeyFFT *bk, int32_t ad_bit, const LweSample *x, const int32_t msize);
void Comp_even_h2(LweSample *result, const LweBootstrappingKeyFFT *bk, int32_t ad_bit, const LweSample *x, const int32_t msize);
void Comp_funceval(LweSample *result, const LweBootstrappingKeyFFT *bk, TorusPolynomial* testvect, int32_t ad_bit, const LweSample *x, const int32_t msize);
void Comp_EBS_Modular(LweSample *result, const LweBootstrappingKeyFFT *bk, int32_t ad_bit, const LweSample *x, const int32_t msize);
void Comp_EBS_Modular_parallel(LweSample *result, const LweBootstrappingKeyFFT *bk, int32_t ad_bit, const LweSample *x, const int32_t msize); // experimental
void Comp_odd_Rh1(LweSample *result, const LweBootstrappingKeyFFT *bk, int32_t ad_bit, const LweSample *x, const int32_t interval);
void Comp_even_Rh2(LweSample *result, const LweBootstrappingKeyFFT *bk, int32_t ad_bit, const LweSample *x, const int32_t interval);
void Comp_Rfunceval(LweSample *result, const LweBootstrappingKeyFFT *bk, TorusPolynomial* testvect, int32_t ad_bit, const LweSample *x, const int32_t interval);
void Comp_EBS_Real(LweSample *result, const LweBootstrappingKeyFFT *bk, int32_t ad_bit, const LweSample *x, const int32_t interval);

//Param Quality
void param_quality(TFheGateBootstrappingParameterSet* param);
void FDB_quality(boost::multiprecision::float128 V_ct, boost::multiprecision::float128 V_MS, boost::multiprecision::float128 V_BS, boost::multiprecision::float128 V_FDFBACC, boost::multiprecision::float128 p, int32_t ad_bit);
void param_quality_f100(TFheGateBootstrappingParameterSet* param, int32_t RS_param[2], int32_t PM_param[2], int32_t msize);
void FDB_quality_f100(cpp_dec_float_100 V_ct, cpp_dec_float_100 V_MS, cpp_dec_float_100 V_BS, cpp_dec_float_100 V_FDFBACC, cpp_dec_float_100 p, int32_t ad_bit);
void param_quality_FDB(TFheGateBootstrappingParameterSet* param, int32_t RS_param[2], int32_t PM_param[2], int32_t msize, int32_t ad_bit);
void print_result(int32_t msize, int32_t ad_bit, int32_t run, int32_t hdebs, int32_t fdfb, int32_t tota, int32_t comp);

#endif