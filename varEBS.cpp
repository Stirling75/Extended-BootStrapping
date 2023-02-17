#include <bits/c++config.h>
#include "tlwekeyswitch.h"
#include <boost/math/constants/constants.hpp>
#include <math.h>
#include <tfhe/lwe-functions.h>
#include <tfhe/lwesamples.h>
#include <tfhe/numeric_functions.h>
#include <tfhe/polynomials.h>
#include <tfhe/polynomials_arithmetic.h>
#include <tfhe/tfhe_core.h>
#include <tfhe/tlwe.h>
#include <tfhe/tlwe_functions.h>
#include <boost/multiprecision/float128.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics.hpp>
#include <boost/accumulators/numeric/functional.hpp>

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

using namespace std;
using namespace boost::multiprecision;
using boost::multiprecision::cpp_dec_float_100;
using namespace boost::accumulators;


void tLweArrayMulByXaiMinusOne(TLweSample *result, int32_t ai, const TLweSample *bk, int32_t k2, const TLweParams *params) {
    const int32_t k = params->k;
    const int32_t qi = ai / k2;
    const int32_t ri = ai % k2;
    const int32_t N = params->N;
    if (ri != 0) {
        for (int32_t i = 0; i < ri; i++){
            int32_t h = k2 - ri + i;
            for (int32_t j = 0; j <= k; j++) {
                torusPolynomialMulByXai(&(result + i)->a[j], (qi + 1) % (2 * N), &(bk + h)->a[j]);
                torusPolynomialSubTo(&(result + i)->a[j], &(bk + i)->a[j]);
            }
        }
        for (int32_t i = ri; i < k2; i++){
            int32_t h = i - ri;
            for (int32_t j = 0; j <= k; j++) {
                torusPolynomialMulByXai(&(result + i)->a[j], qi, &(bk + h)->a[j]);
                torusPolynomialSubTo(&(result + i)->a[j], &(bk + i)->a[j]);
            }
        }
    } else {
        for (int32_t i = 0; i < k2; i++){
            for (int32_t j = 0; j <= k; j++) {
                torusPolynomialMulByXaiMinusOne(&(result + i)->a[j], qi, &(bk + i)->a[j]);
            }
        }
    }

}

void ext_MuxRotate_FFT(TLweSample *result, const TLweSample *accum, const TGswSampleFFT *bki, const int32_t barai, int32_t k2,
                        const TGswParams *bk_params) {
    // ACC = BKi*[(X^barai-1)*ACC]+ACC
    // temp = (X^barai-1)*ACC
    tLweArrayMulByXaiMinusOne(result, barai, accum, k2, bk_params->tlwe_params);
    // temp *= BKi
    // #pragma omp parallel num_threads(min(k2, 64))
    #pragma omp parallel num_threads(std::min(k2, 32))
    // #pragma omp parallel num_threads(min(k2/2, 32))
    #pragma omp for
    for (int i = 0; i < k2; i++){
        tGswFFTExternMulToTLwe(result + i, bki, bk_params);
        tLweAddTo(result + i, accum + i, bk_params->tlwe_params);
    }
    // ACC += temp
    
}

void ext_blindRotate_FFT(TLweSample *accum,
                                 const TGswSampleFFT *bkFFT,
                                 const int32_t *bara,
                                 const int32_t n,
                                 const int32_t k2,
                                 const TGswParams *bk_params) {

    //TGswSampleFFT* temp = new_TGswSampleFFT(bk_params);
    TLweSample *temp = new_TLweSample_array(k2, bk_params->tlwe_params);
    TLweSample *temp2 = temp;
    TLweSample *temp3 = accum;
    const int32_t N = bk_params->tlwe_params->N;

    for (int32_t i = 0; i < n; i++) {
        const int32_t barai = bara[i];
        if (barai == 0) continue; //indeed, this is an easy case!
        ext_MuxRotate_FFT(temp2, temp3, bkFFT + i, barai, k2, bk_params);
        swap(temp2, temp3);
    }
    if (temp3 != accum) {
        for (int32_t i = 0; i < k2; i++){
            tLweCopy(accum+i, temp3+i, bk_params->tlwe_params);
        }
    }

    delete_TLweSample_array(k2, temp);
    //delete_TGswSampleFFT(temp);
}

void torusPolynomialArrayMulByXai(TorusPolynomial *result, int32_t k2, int32_t a, const TorusPolynomial *source){
    const int32_t N = source->N;
    int32_t q = a / k2;
    int32_t r = a % k2;
    if (r != 0){
        for (int32_t i = 0; i < k2-r; i++){
            int32_t j = r+i;
            torusPolynomialMulByXai(result+j, q, source+i);
        }
        for (int32_t i = k2-r; i < k2; i++){
            int32_t j = i-k2+r;
            torusPolynomialMulByXai(result+j, (q + 1) % (2 * N), source+i);
        }
    } else {
        for (int32_t i = 0; i < k2; i++){
            torusPolynomialMulByXai(result+i, q, source+i);
        }
    }
}

void ext_blindRotateAndExtract_FFT(LweSample *result,
                                           const TorusPolynomial *v,
                                           const TGswSampleFFT *bk,
                                           const int32_t barb,
                                           const int32_t *bara,
                                           const int32_t n,
                                           const int32_t ad_bit,
                                           const TGswParams *bk_params) {

    const TLweParams *accum_params = bk_params->tlwe_params;
    const LweParams *extract_params = &accum_params->extracted_lweparams;
    const int32_t N = accum_params->N;
    const int32_t k2 = pow(2, ad_bit);
    const int32_t Nk2 = 2*N*k2;

    // Test polynomial 
    TorusPolynomial *testvectbis = new_TorusPolynomial_array(k2, N);
    // Accumulator
    TLweSample *acc = new_TLweSample_array(k2, accum_params);

    // testvector = X^{2N-barb}*v
    if (barb != 0) torusPolynomialArrayMulByXai(testvectbis, k2, Nk2 - barb, v);
    else for (int32_t i = 0; i < k2; i++) torusPolynomialCopy(testvectbis+i, v+i);
    
    for (int32_t i = 0; i < k2; i++) tLweNoiselessTrivial(acc+i, testvectbis+i, accum_params);
    // Blind rotation
    ext_blindRotate_FFT(acc, bk, bara, n, k2, bk_params);
    // Extraction
    tLweExtractLweSample(result, acc+0, extract_params, accum_params);

    delete_TLweSample_array(k2, acc);
    delete_TorusPolynomial_array(k2, testvectbis);
}


void ext_blindRotate_woExtract_FFT(TLweSample *result,
                                           const TorusPolynomial *v,
                                           const TGswSampleFFT *bk,
                                           const int32_t barb,
                                           const int32_t *bara,
                                           const int32_t n,
                                           const int32_t ad_bit,
                                           const TGswParams *bk_params) {

    const TLweParams *accum_params = bk_params->tlwe_params;
    const LweParams *extract_params = &accum_params->extracted_lweparams;
    const int32_t N = accum_params->N;
    const int32_t k2 = pow(2, ad_bit);
    const int32_t Nk2 = 2*N*k2;

    // Test polynomial 
    TorusPolynomial *testvectbis = new_TorusPolynomial_array(k2, N);
    // Accumulator
    TLweSample *acc = new_TLweSample_array(k2, accum_params);

    // testvector = X^{2N-barb}*v
    if (barb != 0) torusPolynomialArrayMulByXai(testvectbis, k2, Nk2 - barb, v);
    else for (int32_t i = 0; i < k2; i++) torusPolynomialCopy(testvectbis+i, v+i);
    
    for (int32_t i = 0; i < k2; i++) tLweNoiselessTrivial(acc+i, testvectbis+i, accum_params);
    // Blind rotation
    ext_blindRotate_FFT(acc, bk, bara, n, k2, bk_params);
    for (int i = 0; i < k2; i++){
        tLweCopy(result + i, acc + i, accum_params);
    }

    delete_TLweSample_array(k2, acc);
    delete_TorusPolynomial_array(k2, testvectbis);
}


void param_quality_woMspace(TFheGateBootstrappingParameterSet* param, int32_t RS_param[2], int32_t PM_param[2]) {
    float128 n = param->in_out_params->n;
    float128 N = param->tgsw_params->tlwe_params->N;
    float128 k = param->tgsw_params->tlwe_params->k;

    float128 bk_Bg = param->tgsw_params->Bg;
    float128 bk_halfBg = param->tgsw_params->halfBg;  
    float128 bk_l = param->tgsw_params->l;
    float128 bk_stdev = param->tgsw_params->tlwe_params->alpha_min; 
    // static const double bk_stdev = 0; 

    float128 ks_Bg = 1<< param->ks_basebit; 
    float128 ks_l  = param->ks_t;
    float128 ks_stdev = param->in_out_params->alpha_min;
    float128 V_ct = pow(ks_stdev, 2);

    float128 rs_Bg = 1<<RS_param[0];
    float128 rs_l = RS_param[1];

    float128 pm_Bg = 1<<PM_param[0];
    float128 pm_l = PM_param[1];

    float128 log2_VMS = log2(n+1) - 4 - log2(3) - 2 * log2(N);
    float128 V_MS = pow(2., log2_VMS);
    float128 log2_stdMS = log2_VMS/2; 

    float128 BR_1 = log2(n) + log2(N) + log2(k+1) + log2(bk_l) + 2*log2(bk_halfBg) + 2*log2(bk_stdev);
    float128 BR_2 = log2(n) + log2(1 + k*N) - 2 * bk_l * log2(bk_Bg) - 2;
    float128 V_BR = pow(2., BR_1) + pow(2., BR_2);
    float128 log2_VBR = log2(V_BR); 
    float128 KS_1 = log2(k) + log2(N) + log2(ks_l) + 2 * log2(ks_stdev);
    float128 KS_2 = log2(k) -2 * ks_l * log2(ks_Bg) + log2(N) - 2 - log2(3);
    float128 V_KS = pow(2., KS_1) + pow(2., KS_2);
    float128 log2_VKS = log2(V_KS);
    float128 V_BS = V_BR + V_KS;
    float128 log2_VBS = log2(V_BS);
    float128 RS_1 = log2(n) + log2(rs_l) + 2 * log2(bk_stdev);
    float128 RS_2 = -2 * rs_l * log2(rs_Bg) + log2(n) - 2 - log2(3);
    float128 V_RS = pow(2., RS_1) + pow(2., RS_2);
    float128 log2_VRS = log2(V_RS);
    float128 FDFBACC_1 = log2(N) + log2(pm_l) + 2*log2(pm_Bg) + log2(V_RS + V_BS) - 2;
    float128 FDFBACC_2 = log2(1 + k*N) - 2*pm_l*log2(pm_Bg)- 2;
    float128 V_FDFBACC = pow(2., FDFBACC_1) + pow(2., FDFBACC_2);
    float128 log2_VFDFBACC = log2(V_FDFBACC);
    
    // We assume V_ct = V_TLWE
    float128 V_HDFB = V_BS;
    float128 log2_HDFB = log2(V_HDFB);

    float128 V_FDFB = V_BS + V_FDFBACC;
    float128 log2_FDFB = log2(V_FDFB);

    float128 V_TOTA = V_BS;
    float128 log2_TOTA = log2(V_TOTA);

    float128 V_Comp = 2*V_BS;
    float128 log2_Comp = log2(V_Comp);

    std::cout << BOLDWHITE "========================================================== \n" RESET;
    std::cout << "    Variance of BlindRotate : 2^{" << log2_VBR <<"} \n";
    std::cout << "      Variance of KeySwitch : 2^{" << log2_VKS <<"} \n";
    std::cout << "  Variance of Bootstrapping : 2^{" << log2_VBS << "} \n";
    std::cout << " Variance of LWE-to-TLWE KS : 2^{" << log2_VRS << "} \n";
    std::cout << "        Variance of FDFBACC : 2^{" << log2_VFDFBACC << "} \n";
    std::cout << BOLDWHITE "========================================================== \n" RESET;
    std::cout << BOLDYELLOW "    Stdev of Discretization : 2^{" << log2_stdMS <<"} \n" RESET;
    std::cout << "       Stdev of BlindRotate : 2^{" << log2_VBR/2 <<"} \n";
    std::cout << "         Stdev of KeySwitch : 2^{" << log2_VKS/2 <<"} \n";
    std::cout << "     Stdev of Bootstrapping : 2^{" << log2_VBS/2 << "} \n";
    std::cout << "    Stdev of LWE-to-TLWE KS : 2^{" << log2_VRS/2 << "} \n";
    std::cout << "           Stdev of FDFBACC : 2^{" << log2_VFDFBACC/2 << "} \n";
    std::cout << BOLDWHITE "========================================================== \n" RESET;
    std::cout << "       Precision estimated with 4-sigma (99.9936%) \n";
    std::cout << "    Precision of output after BlindRotate : " << BOLDGREEN <<(int) floor(32 - (32 + log2_VBR/2 + 3)) << " Bits \n" RESET;
    std::cout << "      Precision of output after Bootstrap : " << BOLDRED << (int) floor(32 - (32 + log2_VBS/2 + 3)) << " Bits \n" RESET; 
    std::cout << "                      Desired value for k : " << BOLDCYAN << std::max(0, (int) ceil(log2_stdMS - log2_VBS/2)) << " ~ " << std::max(0, (int) ceil(log2_stdMS - log2_VBR/2)) << "\n" RESET;
    std::cout << BOLDWHITE "========================================================== \n" RESET;
    std::cout << BOLDRED "Log2 error std for HDFB <= " << log2_HDFB/2 << "\n" RESET;
    std::cout << BOLDGREEN "Log2 error std for FDFB <= " << log2_FDFB/2 << "\n" RESET;
    std::cout << BOLDYELLOW "Log2 error std for TOTA <= " << log2_TOTA/2 << "\n" RESET;
    std::cout << BOLDBLUE "Log2 error std for Comp <= " << log2_Comp/2 << "\n" RESET; 
    std::cout << BOLDWHITE "========================================================== \n" RESET;
}


///////////////////////////////////////////////////////////////////////////

void HDEBS_Mfunceval_woKS_FFT(LweSample *result, const LweBootstrappingKeyFFT *bk, int32_t ad_bit, const LweSample *x, const int32_t msize) {

    const TGswParams *bk_params = bk->bk_params;
    const TLweParams *accum_params = bk->accum_params;
    const LweParams *in_params = bk->in_out_params;
    const int32_t N = accum_params->N;
    const int32_t n = in_params->n;
    const int32_t k2 = pow(2, ad_bit);
    const int32_t Nk2 = 2*N*k2;
    const int32_t Nk = N*k2;
    const int32_t halfNk = N*k2/2;
    const int32_t NkM = Nk / msize; 

    TorusPolynomial *testvect = new_TorusPolynomial_array(k2, N);
    int32_t *bara = new int32_t[N];


    int32_t barb = modSwitchFromTorus32(x->b, Nk2); 
    for (int i = 0; i < n; i++){
        bara[i] = modSwitchFromTorus32(x->a[i], Nk2);
    }

    // From 0 -> q-1 
    for (int32_t j = 0; j < N; j++) {
        for (int32_t i = 0; i < k2; i++) {
            int32_t m = (i + k2*j)/NkM;
            // int32_t m1 = m / sqrt(msize);
            // int32_t m2 = m - m1*sqrt(msize); 
            // (testvect+i)->coefsT[j] = modSwitchToTorus32((m1*m2)%((int32_t) M), M);
            // (testvect+i)->coefsT[j] = modSwitchToTorus32((m1*m2)/((int32_t) sqrt(msize)), msize);
            (testvect+i)->coefsT[j] = modSwitchToTorus32(m, msize*2);
        }
    }

    // Bootstrapping rotation and extraction
    ext_blindRotateAndExtract_FFT(result, testvect, bk->bkFFT, barb, bara, n, ad_bit, bk_params);


    delete[] bara;
    delete_TorusPolynomial_array(k2, testvect);
}

void HDEBS(LweSample *result, const LweBootstrappingKeyFFT *bk, int32_t ad_bit, const LweSample *x, const int32_t msize) {
    LweSample* c1 = new_LweSample(&bk->accum_params->extracted_lweparams);
    LweSample* ctemp = new_LweSample(bk->in_out_params);
    lweCopy(ctemp, x, bk->in_out_params);
    ctemp->b += modSwitchToTorus32(1, 4*msize);
    
    HDEBS_Mfunceval_woKS_FFT(c1, bk, ad_bit, ctemp, msize);
    lweKeySwitch(result, bk->ks, c1);

    delete_LweSample(c1);
    delete_LweSample(ctemp);
}

void HDEBS_Mfunceval_varest(TLweSample *result, TFheGateBootstrappingSecretKeySet* key, int32_t ad_bit, const LweSample *x, const int32_t msize) {
    const LweBootstrappingKeyFFT *bk = key->cloud.bkFFT;
    const TGswParams *bk_params = bk->bk_params;
    const TLweParams *accum_params = bk->accum_params;
    const LweParams *in_params = bk->in_out_params;
    const int32_t N = accum_params->N;
    const int32_t n = in_params->n;
    const int32_t k2 = pow(2, ad_bit);
    const int32_t Nk2 = 2*N*k2;
    const int32_t Nk = N*k2;
    const int32_t halfNk = N*k2/2;
    const int32_t NkM = Nk / msize; 
    
    TorusPolynomial *testvectmp = new_TorusPolynomial_array(k2, N);
    TorusPolynomial *testvect = new_TorusPolynomial_array(k2, N);
    int32_t *bara = new int32_t[N];


    int32_t barb = modSwitchFromTorus32(x->b, Nk2); 
    int32_t dphase = barb;
    for (int i = 0; i < n; i++){
        bara[i] = modSwitchFromTorus32(x->a[i], Nk2);
        dphase -= bara[i] * key->lwe_key->key[i];
    }
    dphase = ((dphase%Nk2) + Nk2)%Nk2;

    // From 0 -> q-1 
    auto distr = std::bind(std::uniform_int_distribution<int32_t>{0, 1610612736}, std::mt19937(std::random_device{}()));
    for (int32_t j = 0; j < N; j++) {
        for (int32_t i = 0; i < k2; i++) {
            int32_t m = (i + k2*j)/NkM;
            // (testvectmp+i)->coefsT[j] = modSwitchToTorus32(m, msize*2);
            (testvectmp+i)->coefsT[j] = distr(); //random testvect
        }
    }
    torusPolynomialArrayMulByXai(testvect, k2, dphase, testvectmp);


    // Bootstrapping rotation and extraction
    ext_blindRotate_woExtract_FFT(result, testvect, bk->bkFFT, barb, bara, n, ad_bit, bk_params);
    
    for (int i = 0; i < k2; i++){
        torusPolynomialSub((result+i)->b, (result+i)->b, testvectmp+i);
    }

    delete[] bara;
    delete_TorusPolynomial_array(k2, testvect);
    delete_TorusPolynomial_array(k2, testvectmp);
}

void HDEBS_varest(TLweSample *result, TFheGateBootstrappingSecretKeySet* key, int32_t ad_bit, const LweSample *x, const int32_t msize) {
    LweSample* ctemp = new_LweSample(key->lwe_key->params);
    lweCopy(ctemp, x, key->lwe_key->params);
    ctemp->b += modSwitchToTorus32(1, 4*msize);
    
    HDEBS_Mfunceval_varest(result, key, ad_bit, ctemp, msize);

    delete_LweSample(ctemp);
}

///////////////////////////////////////////////////////////////////////////
void TOTA_signeval_woKS_FFT(LweSample *result, const LweBootstrappingKeyFFT *bk, int32_t ad_bit, const LweSample *x) {

    const TGswParams *bk_params = bk->bk_params;
    const TLweParams *accum_params = bk->accum_params;
    const LweParams *in_params = bk->in_out_params;
    const int32_t N = accum_params->N;
    const int32_t n = in_params->n;
    const int32_t k2 = pow(2, ad_bit);
    const int32_t Nk2 = 2*N*k2;
    const int32_t Nk = N*k2;

    TorusPolynomial *testvect = new_TorusPolynomial_array(k2, N);
    int32_t *bara = new int32_t[N];


    int32_t barb = modSwitchFromTorus32(x->b, Nk);
    for (int i = 0; i < n; i++){
        bara[i] = modSwitchFromTorus32(x->a[i], Nk);
    }

    for (int32_t i = 0; i < k2; i++) {
        for (int32_t j = 0; j < N; j++) {
            (testvect+i)->coefsT[j] = modSwitchToTorus32(1, 4);
        }
    }

    // Bootstrapping rotation and extraction
    ext_blindRotateAndExtract_FFT(result, testvect, bk->bkFFT, barb, bara, n, ad_bit, bk_params);
    result->b -= modSwitchToTorus32(1, 4);


    delete[] bara;
    delete_TorusPolynomial_array(k2, testvect);
}

void TOTA_Mfunceval_woKS_FFT(TLweSample *result, TFheGateBootstrappingSecretKeySet* key, int32_t ad_bit, const LweSample *x, const LweSample *signx, const int32_t msize) {
    const LweBootstrappingKeyFFT* bk = key->cloud.bkFFT;
    const TGswParams *bk_params = bk->bk_params;
    const TLweParams *accum_params = bk->accum_params;
    const LweParams *in_params = bk->in_out_params;
    const int32_t N = accum_params->N;
    const int32_t n = in_params->n;
    const int32_t k2 = pow(2, ad_bit);
    const int32_t Nk2 = 2*N*k2;
    const int32_t Nk = N*k2;
    const int32_t halfNk = N*k2/2;
    const int32_t NkM = Nk / msize; 
    
    TorusPolynomial *testvectmp = new_TorusPolynomial_array(k2, N);
    TorusPolynomial *testvect = new_TorusPolynomial_array(k2, N);
    int32_t *bara = new int32_t[N];


    int32_t barb = (modSwitchFromTorus32(x->b, Nk) + modSwitchFromTorus32(signx->b, Nk2)) % Nk2; 
    int32_t dphase = barb;
    for (int i = 0; i < n; i++){
        bara[i] = (modSwitchFromTorus32(x->a[i], Nk) + modSwitchFromTorus32(signx->a[i], Nk2)) % Nk2;
        dphase -= bara[i] * key->lwe_key->key[i];
    }
    dphase = (dphase%Nk2 + Nk2)%Nk2;
    
    auto distr = std::bind(std::uniform_int_distribution<int32_t>{std::numeric_limits<int32_t>::min(),std::numeric_limits<int32_t>::max()}, std::mt19937(std::random_device{}()));
    // From 0 -> q-1 
    for (int32_t j = 0; j < N; j++) {
        for (int32_t i = 0; i < k2; i++) {
            int32_t m = (i + k2*j)/NkM;
            (testvectmp+i)->coefsT[j] = modSwitchToTorus32(m, msize);
            // (testvectmp+i)->coefsT[j] = distr();
        }
    }
    torusPolynomialArrayMulByXai(testvect, k2, dphase, testvectmp);


    // Bootstrapping rotation and extraction
    ext_blindRotate_woExtract_FFT(result, testvect, bk->bkFFT, barb, bara, n, ad_bit, bk_params);
    for (int i = 0; i < k2; i++){
        torusPolynomialSub((result+i)->b, (result+i)->b, testvectmp+i);
    }

    delete[] bara;
    delete_TorusPolynomial_array(k2, testvect);
    delete_TorusPolynomial_array(k2, testvectmp);
}

void TOTA_Rfunceval_woKS_FFT(TLweSample *result, TFheGateBootstrappingSecretKeySet* key, int32_t ad_bit, const LweSample *x, const LweSample *signx, int32_t interval) {
    const LweBootstrappingKeyFFT* bk = key->cloud.bkFFT;
    const TGswParams *bk_params = bk->bk_params;
    const TLweParams *accum_params = bk->accum_params;
    const LweParams *in_params = bk->in_out_params;
    const int32_t N = accum_params->N;
    const int32_t n = in_params->n;
    const int32_t k2 = pow(2, ad_bit);
    const int32_t Nk2 = 2*N*k2;
    const int32_t Nk = N*k2;
    const int32_t halfNk = N*k2/2;
    const int32_t int_length = interval * 2;
    double delta = pow(2., 32)/(interval*2);
    
    TorusPolynomial *testvectmp = new_TorusPolynomial_array(k2, N);
    TorusPolynomial *testvect = new_TorusPolynomial_array(k2, N);
    int32_t *bara = new int32_t[N];


    int32_t barb = (modSwitchFromTorus32(x->b, Nk) + modSwitchFromTorus32(signx->b, Nk2)) % Nk2; 
    int32_t dphase = barb;
    for (int i = 0; i < n; i++){
        bara[i] = (modSwitchFromTorus32(x->a[i], Nk) + modSwitchFromTorus32(signx->a[i], Nk2)) % Nk2;
        dphase -= bara[i] * key->lwe_key->key[i];
    }
    dphase = (dphase%Nk2 + Nk2)%Nk2;

    // From -1/2 -> 0
    for (int32_t j = 0; j < N/2; j++) {
        for (int32_t i = 0; i < k2; i++) {
            double m = int_length*(j*k2 + i - halfNk)/(double)Nk;
            // (testvect+i)->coefsT[j] = (Torus32) ((double) delta * (m+50)*(m+7)*(m-50)/(2000)); 
            // (testvect+i)->coefsT[j] = (Torus32) ((double) delta * (43*sin(m*boost::math::double_constants::pi / 4))); 
            // (testvect+i)->coefsT[j] = (Torus32) ((double) delta * (40*(exp(m/24) - exp(-m/24))/(exp(m/24)+exp(-m/24))));
            (testvectmp+i)->coefsT[j] = (Torus32) ((double) delta * (43*sin(m*boost::math::double_constants::pi / 32))); 
        }
    }

    // 0 -> 1/2
    for (int32_t j = N/2; j < N; j++) {
        for (int32_t i = 0; i < k2; i++) {
            double m = int_length*(j*k2 + i - halfNk)/(double)Nk;
            // (testvect+i)->coefsT[j] = (Torus32) ((double) delta * ((m+50)*(m+7)*(m-50)/(2000))); 
            // (testvect+i)->coefsT[j] = (Torus32) ((double) delta * (43*sin(m*boost::math::double_constants::pi / 4))); 
            // (testvect+i)->coefsT[j] = (Torus32) ((double) delta * (40*(exp(m/24) - exp(-m/24))/(exp(m/24)+exp(-m/24))));
            (testvectmp+i)->coefsT[j] = (Torus32) ((double) delta * (43*sin(m*boost::math::double_constants::pi / 32))); 
        }
    }
    torusPolynomialArrayMulByXai(testvect, k2, dphase, testvectmp);

    // Bootstrapping rotation and extraction
    ext_blindRotate_woExtract_FFT(result, testvect, bk->bkFFT, barb, bara, n, ad_bit, bk_params);
    for (int i = 0; i < k2; i++){
        torusPolynomialSub((result+i)->b, (result+i)->b, testvectmp+i);
    }


    delete[] bara;
    delete_TorusPolynomial_array(k2, testvect);
    delete_TorusPolynomial_array(k2, testvectmp);
}

void TOTA_EBS_Modular(TLweSample *result, TFheGateBootstrappingSecretKeySet* key, int32_t ad_bit, const LweSample *x, const int32_t msize) {
    const LweBootstrappingKeyFFT* bk = key->cloud.bkFFT;
    LweSample* c1 = new_LweSample(&bk->accum_params->extracted_lweparams);
    LweSample* c1ks = new_LweSample(bk->in_out_params);
    LweSample* ctemp = new_LweSample(bk->in_out_params);
    lweCopy(ctemp, x, bk->in_out_params);
    ctemp->b += modSwitchToTorus32(1, 2*msize);

    TOTA_signeval_woKS_FFT(c1, bk, ad_bit, ctemp); // c1 = TLWE_{K}(1/2 * sign(?*2^k N))
    lweKeySwitch(c1ks, bk->ks, c1); 

    TOTA_Mfunceval_woKS_FFT(result, key, ad_bit, ctemp, c1ks, msize);
    // lweKeySwitch(result, bk->ks, c1);

    delete_LweSample(c1);
    delete_LweSample(c1ks);
    delete_LweSample(ctemp);
}

void TOTA_EBS_Real(TLweSample *result, TFheGateBootstrappingSecretKeySet* key, int32_t ad_bit, const LweSample *x, int32_t interval) {
    const LweBootstrappingKeyFFT* bk = key->cloud.bkFFT;
    LweSample* c1 = new_LweSample(&bk->accum_params->extracted_lweparams);
    LweSample* c1ks = new_LweSample(bk->in_out_params);
    LweSample* ctemp = new_LweSample(bk->in_out_params);
    lweCopy(ctemp, x, bk->in_out_params);
    ctemp->b += modSwitchToTorus32(1, 2);

    TOTA_signeval_woKS_FFT(c1, bk, ad_bit, ctemp); // c1 = TLWE_{K}(1/2 * sign(?*2^k N))
    lweKeySwitch(c1ks, bk->ks, c1); 

    TOTA_Rfunceval_woKS_FFT(result, key, ad_bit, ctemp, c1ks, interval);
    // lweKeySwitch(result, bk->ks, c1);

    delete_LweSample(c1);
    delete_LweSample(c1ks);
    delete_LweSample(ctemp);
}
///////////////////////////////////////////////////////////////////////////

void FDFB_signeval_FFT(TLweSample *result, const LweBootstrappingKeyFFT *bk, TLweKeySwitchKey* rks, int32_t ad_bit, const LweSample *x, int32_t PM_param[2]) {
    const TGswParams *bk_params = bk->bk_params;
    const TLweParams *accum_params = bk->accum_params;
    const LweParams *in_params = bk->in_out_params;
    const int32_t N = accum_params->N;
    const int32_t n = in_params->n;
    const int32_t k2 = pow(2, ad_bit);
    const int32_t Nk2 = 2*N*k2;
    const int32_t Nk = N*k2;
    const int32_t base = 1<<PM_param[0];

    
    LweSample* smallLWE = new_LweSample_array(PM_param[1], in_params);
    LweSample* LargeLWE = new_LweSample_array(PM_param[1], bk->extract_params);
    int32_t *bara = new int32_t[N];



    int32_t barb = modSwitchFromTorus32(x->b, Nk2);
    for (int i = 0; i < n; i++){
        bara[i] = modSwitchFromTorus32(x->a[i], Nk2);
    }

    // #pragma omp parallel num_threads(PM_param[1])
    // #pragma omp for
    for (int32_t l = 0; l < PM_param[1]; l++) {
        TorusPolynomial *testvect = new_TorusPolynomial_array(k2, N);
        for (int32_t i = 0; i < k2; i++) {
            for (int32_t j = 0; j < N; j++) {
                (testvect+i)->coefsT[j] = 1<<(31-(l+1)*PM_param[0]);
            }
        }

        // Bootstrapping rotation and extraction
        ext_blindRotateAndExtract_FFT(LargeLWE+l, testvect, bk->bkFFT, barb, bara, n, ad_bit, bk_params);
        (LargeLWE+l)->b += 1<<(31-(l+1)*PM_param[0]);
        lweKeySwitch(smallLWE+l, bk->ks, LargeLWE+l);
        TLweKeySwitch(result+l, rks, smallLWE+l);
        delete_TorusPolynomial_array(k2, testvect);
    }

    delete[] bara;
    // delete_TorusPolynomial_array(k2, testvect);
    delete_LweSample_array(PM_param[1], smallLWE);
    delete_LweSample_array(PM_param[1], LargeLWE);
}

void FDFB_PubMux(TLweSample* ACC, TLweSample* ACCres, TorusPolynomial* stv, TLweSample* TGSWp, const LweBootstrappingKeyFFT *bk, int32_t ad_bit, int32_t PM_param[2], int32_t msize, int32_t barb) {
    const TGswParams *bk_params = bk->bk_params;
    const TLweParams *accum_params = bk->accum_params;
    const LweParams *in_params = bk->in_out_params;
    const int32_t N = accum_params->N;
    const int32_t n = in_params->n;
    const int32_t k2 = pow(2, ad_bit);
    const int32_t Nk2 = 2*N*k2;
    const int32_t Nk = N*k2;
    const int32_t N2kM = Nk2 / msize; 
    const int32_t base = 1<<PM_param[0];
    const int32_t mask = base - 1;
    TorusPolynomial* tp_0 = new_TorusPolynomial_array(k2, N);  // (p+) - (p-)
    TorusPolynomial* tp_1 = new_TorusPolynomial_array(k2, N);  // (p-)
    TorusPolynomial* p_0 = new_TorusPolynomial_array(k2, N);  // ((p+) - (p-))*X^{-b}
    TorusPolynomial* p_1 = new_TorusPolynomial_array(k2, N);  // (p-)*X^{-b}
    
    auto distr = std::bind(std::uniform_int_distribution<int32_t>{std::numeric_limits<int32_t>::min(),std::numeric_limits<int32_t>::max()}, std::mt19937(std::random_device{}()));
    for (int32_t j = 0; j < N; j++) {
        for (int32_t i = 0; i < k2; i++) {
            int32_t m = (i + k2*j)/N2kM;
            // (tp_1+i)->coefsT[j] = - modSwitchToTorus32(msize/2 + m, msize);
            // (tp_0+i)->coefsT[j] = modSwitchToTorus32(m, msize);
            (tp_1+i)->coefsT[j] = distr();
            (tp_0+i)->coefsT[j] = distr();
        }
    }
    for (int32_t i = 0; i < k2; i++){
        torusPolynomialSubTo(tp_0+i, tp_1+i);
        torusPolynomialCopy(p_0 + i, tp_0 + i);
        torusPolynomialCopy(p_1 + i, tp_1 + i);
    }
    if (barb != 0) torusPolynomialArrayMulByXai(p_0, k2, Nk2 - barb, tp_0);
    if (barb != 0) torusPolynomialArrayMulByXai(p_1, k2, Nk2 - barb, tp_1);
    
    // #pragma omp parallel num_threads(min(k2, 16))
    // #pragma omp for
    for (int32_t i = 0; i < k2; i++){
        IntPolynomial* decomp = new_IntPolynomial_array(PM_param[1], N);
        for (int32_t j = 0; j < PM_param[1]; j++) {
            for (int32_t k = 0; k < N; k++) {
                (decomp+j)->coefs[k] =  (((p_0+i)->coefsT[k])>>(32-(j+1)*PM_param[0])) & mask;
            }
            tLweAddMulRTo(ACC+i, decomp+j, TGSWp+j, accum_params);    
        }
        torusPolynomialAddTo((ACC+i)->b, (p_1+i));
        delete_IntPolynomial_array(PM_param[1], decomp);
    }

    for (int i = 0; i < k2; i++){
        torusPolynomialAddTo(p_0 + i, p_1 + i);
        torusPolynomialCopy(stv + i, p_0 + i);
        // torusPolynomialSubTo((ACC+i)->b, tp_0 + i); //// eliminate this line if not needed 
        tLweCopy(ACCres + i, ACC + i, accum_params);
        torusPolynomialSubTo((ACCres + i)->b, p_0 + i);
    }
    
    
    delete_TorusPolynomial_array(k2, tp_0);
    delete_TorusPolynomial_array(k2, tp_1);
    delete_TorusPolynomial_array(k2, p_0);
    delete_TorusPolynomial_array(k2, p_1);
}

void FDFB_PubMux_Real(TLweSample* ACC, TLweSample* ACCres, TorusPolynomial* stv, TLweSample* TGSWp, const LweBootstrappingKeyFFT *bk, int32_t ad_bit, int32_t PM_param[2], int32_t interval, int32_t barb) {
    const TGswParams *bk_params = bk->bk_params;
    const TLweParams *accum_params = bk->accum_params;
    const LweParams *in_params = bk->in_out_params;
    const int32_t N = accum_params->N;
    const int32_t n = in_params->n;
    const int32_t k2 = pow(2, ad_bit);
    const int32_t Nk2 = 2*N*k2;
    const int32_t Nk = N*k2;
    // const int32_t N2kM = Nk2 / msize; 
    const int32_t base = 1<<PM_param[0];
    const int32_t mask = base - 1;
    double delta = pow(2., 32)/(interval*2);
    TorusPolynomial* tp_0 = new_TorusPolynomial_array(k2, N);  // (p+) - (p-)
    TorusPolynomial* tp_1 = new_TorusPolynomial_array(k2, N);  // (p-)
    TorusPolynomial* p_0 = new_TorusPolynomial_array(k2, N);  // ((p+) - (p-))*X^{-b}
    TorusPolynomial* p_1 = new_TorusPolynomial_array(k2, N);  // (p-)*X^{-b}
    
    for (int32_t j = 0; j < N; j++) {
        for (int32_t i = 0; i < k2; i++) {
            double mp = interval * (i + k2*j)/(double) Nk;
            double mm = interval * (i + k2*j - Nk)/(double) Nk;
            // (tp_0+i)->coefsT[j] = (Torus32) ((double) delta * ((mp+50)*(mp+7)*(mp-50)/(2000)));
            // (tp_0+i)->coefsT[j] = (Torus32) ((double) delta * (43*sin(mp*boost::math::double_constants::pi / 4))); // exp1
            // (tp_0+i)->coefsT[j] = (Torus32) ((double) delta * (40*(exp(mp/24) - exp(-mp/24))/(exp(mp/24)+exp(-mp/24)))); //exp2
            (tp_0+i)->coefsT[j] = (Torus32) ((double) delta * (43*sin(mp*boost::math::double_constants::pi / 32))); // exp3
            // (tp_1+i)->coefsT[j] = - (Torus32) (((double) delta * (mm+50)*(mm+7)*(mm-50)/(2000)));
            // (tp_1+i)->coefsT[j] = - (Torus32) ((double) delta * (43*sin(mm*boost::math::double_constants::pi / 4))); // exp1
            // (tp_1+i)->coefsT[j] = - (Torus32) ((double) delta * (40*(exp(mm/24) - exp(-mm/24))/(exp(mm/24)+exp(-mm/24)))); // exp2
            (tp_1+i)->coefsT[j] = - (Torus32) ((double) delta * (43*sin(mm*boost::math::double_constants::pi / 32))); // exp3
             
        }
    }
    for (int32_t i = 0; i < k2; i++){
        torusPolynomialSubTo(tp_0+i, tp_1+i);
        torusPolynomialCopy(p_0 + i, tp_0 + i);
        torusPolynomialCopy(p_1 + i, tp_1 + i);
    }
    if (barb != 0) torusPolynomialArrayMulByXai(p_0, k2, Nk2 - barb, tp_0);
    if (barb != 0) torusPolynomialArrayMulByXai(p_1, k2, Nk2 - barb, tp_1);
    
    // #pragma omp parallel num_threads(min(k2, 16))
    // #pragma omp for
    for (int32_t i = 0; i < k2; i++){
        IntPolynomial* decomp = new_IntPolynomial_array(PM_param[1], N);
        for (int32_t j = 0; j < PM_param[1]; j++) {
            for (int32_t k = 0; k < N; k++) {
                (decomp+j)->coefs[k] =  (((p_0+i)->coefsT[k])>>(32-(j+1)*PM_param[0])) & mask;
            }
            tLweAddMulRTo(ACC+i, decomp+j, TGSWp+j, accum_params);    
        }
        torusPolynomialAddTo((ACC+i)->b, (p_1+i));
        delete_IntPolynomial_array(PM_param[1], decomp);
    }

    for (int i = 0; i < k2; i++){
        torusPolynomialAddTo(p_0 + i, p_1 + i);
        torusPolynomialCopy(stv + i, p_0 + i);
        // torusPolynomialSubTo((ACC+i)->b, tp_0 + i); //// eliminate this line if not needed 
        tLweCopy(ACCres + i, ACC + i, accum_params);
        torusPolynomialSubTo((ACCres + i)->b, p_0 + i);
    }
    
    
    delete_TorusPolynomial_array(k2, tp_0);
    delete_TorusPolynomial_array(k2, tp_1);
    delete_TorusPolynomial_array(k2, p_0);
    delete_TorusPolynomial_array(k2, p_1);
}

void FDFB_BRnSE(TLweSample* result, TLweSample* ACC, TorusPolynomial* stv, TFheGateBootstrappingSecretKeySet* key, int32_t msize, int32_t ad_bit, const LweSample *x, int32_t barb) {
    const LweBootstrappingKeyFFT *bk = key->cloud.bkFFT;
    const TGswParams *bk_params = bk->bk_params;
    const TLweParams *accum_params = bk->accum_params;
    const LweParams *in_params = bk->in_out_params;
    const int32_t N = accum_params->N;
    const int32_t n = in_params->n;
    const int32_t k2 = pow(2, ad_bit);
    const int32_t Nk2 = 2*N*k2;
    const int32_t Nk = N*k2;
    const int32_t N2kM = Nk2 / msize; 
    const int32_t halfNk = N*k2/2;
    const LweParams *extract_params = &accum_params->extracted_lweparams;

    // TorusPolynomial *testvect = new_TorusPolynomial_array(k2, N);
    int32_t *bara = new int32_t[n];
    int32_t dphase = 0;

    TorusPolynomial* tp_1 = new_TorusPolynomial_array(k2, N); 
    for (int i = 0; i < n; i++){
        bara[i] = modSwitchFromTorus32(x->a[i], Nk2);
        dphase -= bara[i] * key->lwe_key->key[i];
        dphase = (dphase % Nk2 + Nk2)%Nk2;
    }    
    ext_blindRotate_FFT(ACC, bk->bkFFT, bara, n, k2, bk_params);
    torusPolynomialArrayMulByXai(tp_1, k2, Nk2 - dphase, stv);
    for (int i = 0; i < k2; i++){
        torusPolynomialSubTo((ACC+i)->b, tp_1 + i);
        tLweCopy(result + i, ACC + i, accum_params);
    }
    // tLweExtractLweSample(result, ACC+0, extract_params, accum_params);

    delete[] bara;
    delete_TorusPolynomial_array(k2, tp_1);
}

void FDFB_BRnSE_Real(TLweSample* result, TLweSample* ACC, TorusPolynomial* stv, TFheGateBootstrappingSecretKeySet* key, int32_t interval, int32_t ad_bit, const LweSample *x, int32_t barb) {
    const LweBootstrappingKeyFFT *bk = key->cloud.bkFFT;
    const TGswParams *bk_params = bk->bk_params;
    const TLweParams *accum_params = bk->accum_params;
    const LweParams *in_params = bk->in_out_params;
    const int32_t N = accum_params->N;
    const int32_t n = in_params->n;
    const int32_t k2 = pow(2, ad_bit);
    const int32_t Nk2 = 2*N*k2;
    const int32_t Nk = N*k2;
    // const int32_t N2kM = Nk2 / msize;
    double delta = pow(2., 32)/(interval*2); 
    const int32_t halfNk = N*k2/2;
    const LweParams *extract_params = &accum_params->extracted_lweparams;

    // TorusPolynomial *testvect = new_TorusPolynomial_array(k2, N);
    int32_t *bara = new int32_t[n];
    int32_t dphase = 0;

    TorusPolynomial* tp_1 = new_TorusPolynomial_array(k2, N); 
    for (int i = 0; i < n; i++){
        bara[i] = modSwitchFromTorus32(x->a[i], Nk2);
        dphase -= bara[i] * key->lwe_key->key[i];
        dphase = (dphase % Nk2 + Nk2)%Nk2;
    }    
    ext_blindRotate_FFT(ACC, bk->bkFFT, bara, n, k2, bk_params);
    torusPolynomialArrayMulByXai(tp_1, k2, Nk2 - dphase, stv);
    for (int i = 0; i < k2; i++){
        torusPolynomialSubTo((ACC+i)->b, tp_1 + i);
        tLweCopy(result + i, ACC + i, accum_params);
    }
    // tLweExtractLweSample(result, ACC+0, extract_params, accum_params);

    delete[] bara;
    delete_TorusPolynomial_array(k2, tp_1);
}

void FDFB_EBS_Modular(TLweSample *result, TLweSample *accumres, TFheGateBootstrappingSecretKeySet* key, TLweKeySwitchKey* rks,  int32_t ad_bit, const LweSample *x, int32_t PM_param[2], const int32_t msize) {
    const LweBootstrappingKeyFFT *bk = key->cloud.bkFFT;
    const TGswParams *bk_params = bk->bk_params;
    const TLweParams *accum_params = bk->accum_params;
    const LweParams *in_params = bk->in_out_params;
    const LweParams *extract_params = &accum_params->extracted_lweparams;
    const int32_t N = accum_params->N;
    const int32_t k2 = pow(2, ad_bit);
    const int32_t Nk2 = 2*N*k2;
    LweSample* c1 = new_LweSample(&accum_params->extracted_lweparams);
    LweSample* ctemp = new_LweSample(in_params);
    TLweSample* Tgsp = new_TLweSample_array(PM_param[1], accum_params);
    TLweSample* accum = new_TLweSample_array(k2, accum_params);
    TorusPolynomial* sel_tv = new_TorusPolynomial_array(k2, N);
    for (int i = 0; i < k2; i++){
        tLweClear(accum+i, accum_params);
    }
    lweCopy(ctemp, x, bk->in_out_params);
    ctemp->b += modSwitchToTorus32(1, 2*msize);
    int32_t barb = modSwitchFromTorus32(ctemp->b, Nk2);

    FDFB_signeval_FFT(Tgsp, bk, rks, ad_bit, ctemp, PM_param);
    FDFB_PubMux(accum, accumres, sel_tv, Tgsp, bk, ad_bit, PM_param, msize, barb);
    FDFB_BRnSE(result, accum, sel_tv, key, msize, ad_bit, ctemp, barb);
    // lweKeySwitch(result, bk->ks, c1);

    delete_LweSample(c1);
    delete_LweSample(ctemp);
    delete_TLweSample_array(PM_param[1], Tgsp);
    delete_TLweSample_array(k2, accum);
    delete_TorusPolynomial_array(k2, sel_tv);
}

void FDFB_EBS_Real(TLweSample *result, TLweSample *accumres, TFheGateBootstrappingSecretKeySet* key, TLweKeySwitchKey* rks,  int32_t ad_bit, const LweSample *x, int32_t PM_param[2], int32_t interval) {
    const LweBootstrappingKeyFFT *bk = key->cloud.bkFFT;
    const TGswParams *bk_params = bk->bk_params;
    const TLweParams *accum_params = bk->accum_params;
    const LweParams *in_params = bk->in_out_params;
    const LweParams *extract_params = &accum_params->extracted_lweparams;
    const int32_t N = accum_params->N;
    const int32_t k2 = pow(2, ad_bit);
    const int32_t Nk2 = 2*N*k2;
    LweSample* c1 = new_LweSample(&accum_params->extracted_lweparams);
    LweSample* ctemp = new_LweSample(in_params);
    TLweSample* Tgsp = new_TLweSample_array(PM_param[1], accum_params);
    TLweSample* accum = new_TLweSample_array(k2, accum_params);
    TorusPolynomial* sel_tv = new_TorusPolynomial_array(k2, N);
    for (int i = 0; i < k2; i++){
        tLweClear(accum+i, accum_params);
    }
    lweCopy(ctemp, x, bk->in_out_params);
    int32_t barb = modSwitchFromTorus32(ctemp->b, Nk2);

    FDFB_signeval_FFT(Tgsp, bk, rks, ad_bit, ctemp, PM_param);
    FDFB_PubMux_Real(accum, accumres, sel_tv, Tgsp, bk, ad_bit, PM_param, interval, barb);
    FDFB_BRnSE_Real(result, accum, sel_tv, key, interval, ad_bit, ctemp, barb);
    // lweKeySwitch(result, bk->ks, c1);

    delete_LweSample(c1);
    delete_LweSample(ctemp);
    delete_TLweSample_array(PM_param[1], Tgsp);
    delete_TLweSample_array(k2, accum);
    delete_TorusPolynomial_array(k2, sel_tv);
}

///////////////////////////////////////////////////////////////////////////
void Comp_odd_h1(LweSample *result, const LweBootstrappingKeyFFT *bk, int32_t ad_bit, const LweSample *x, const int32_t msize) {
    const TGswParams *bk_params = bk->bk_params;
    const TLweParams *accum_params = bk->accum_params;
    const LweParams *in_params = bk->in_out_params;
    const int32_t N = accum_params->N;
    const int32_t n = in_params->n;
    const int32_t k2 = pow(2, ad_bit);
    const int32_t Nk2 = 2*N*k2;
    const int32_t Nk = N*k2;
    const int32_t N2kM = Nk2 / msize; 
    LweSample* cks = new_LweSample(&accum_params->extracted_lweparams);
    LweSample* ctemp = new_LweSample(in_params);
    lweCopy(ctemp, x, bk->in_out_params);
    ctemp->b += modSwitchToTorus32(1, 2*msize);

    TorusPolynomial *testvect = new_TorusPolynomial_array(k2, N);
    int32_t *bara = new int32_t[N];


    int32_t barb = modSwitchFromTorus32(ctemp->b, Nk2);
    for (int i = 0; i < n; i++){
        bara[i] = modSwitchFromTorus32(ctemp->a[i], Nk2);
    }

    for (int32_t i = 0; i < k2; i++) {
        for (int32_t j = 0; j < N; j++) {
            int32_t m = (i + k2*j)/N2kM;
            (testvect+i)->coefsT[j] =  modSwitchToTorus32(1 + 2*m, 2*msize); // round(m) + 1/2p
        }
    }

    // Bootstrapping rotation and extraction
    ext_blindRotateAndExtract_FFT(cks, testvect, bk->bkFFT, barb, bara, n, ad_bit, bk_params);
    lweKeySwitch(result, bk->ks, cks);
    result->b -= modSwitchToTorus32(1, 2*msize);

    delete[] bara;
    delete_TorusPolynomial_array(k2, testvect);
    delete_LweSample(ctemp);
    delete_LweSample(cks);
}

void Comp_even_h2(LweSample *result, const LweBootstrappingKeyFFT *bk, int32_t ad_bit, const LweSample *x, const int32_t msize) {
    const TGswParams *bk_params = bk->bk_params;
    const TLweParams *accum_params = bk->accum_params;
    const LweParams *in_params = bk->in_out_params;
    const int32_t N = accum_params->N;
    const int32_t n = in_params->n;
    const int32_t k2 = pow(2, ad_bit);
    const int32_t Nk2 = 2*N*k2;
    const int32_t Nk = N*k2;
    const int32_t N2kM = Nk2 / msize; 
    LweSample* cks = new_LweSample(&accum_params->extracted_lweparams);
    LweSample* ctemp = new_LweSample(in_params);
    lweCopy(ctemp, x, bk->in_out_params);
    ctemp->b += modSwitchToTorus32(1, 2*msize);

    TorusPolynomial *testvect = new_TorusPolynomial_array(k2, N);
    int32_t *bara = new int32_t[N];


    int32_t barb = modSwitchFromTorus32(ctemp->b, Nk2);
    for (int i = 0; i < n; i++){
        bara[i] = modSwitchFromTorus32(ctemp->a[i], Nk2);
    }
    
    // abs(x) - 1/4
    for (int32_t i = 0; i < k2; i++) {
        for (int32_t j = 0; j < N; j++) {
            int32_t m = (i + k2*j)/N2kM;
            (testvect+i)->coefsT[j] =  modSwitchToTorus32(1 + 2*m + msize/2, 2*msize); // 1/2p + 1/4 + round(m)
        }
    }

    // Bootstrapping rotation and extraction
    ext_blindRotateAndExtract_FFT(cks, testvect, bk->bkFFT, barb, bara, n, ad_bit, bk_params);
    lweKeySwitch(result, bk->ks, cks);
    // abs(x) - 1/4 + 1/4
    result->b -= modSwitchToTorus32(1 + msize/2, 2*msize);
    
    delete[] bara;
    delete_TorusPolynomial_array(k2, testvect);
    delete_LweSample(ctemp);
    delete_LweSample(cks);
}

void Comp_odd_Rh1(LweSample *result, const LweBootstrappingKeyFFT *bk, int32_t ad_bit, const LweSample *x, const int32_t interval) {
    const TGswParams *bk_params = bk->bk_params;
    const TLweParams *accum_params = bk->accum_params;
    const LweParams *in_params = bk->in_out_params;
    const int32_t N = accum_params->N;
    const int32_t n = in_params->n;
    const int32_t k2 = pow(2, ad_bit);
    const int32_t Nk2 = 2*N*k2;
    const int32_t Nk = N*k2;
    LweSample* cks = new_LweSample(&accum_params->extracted_lweparams);


    TorusPolynomial *testvect = new_TorusPolynomial_array(k2, N);
    int32_t *bara = new int32_t[N];


    int32_t barb = modSwitchFromTorus32(x->b, Nk2);
    for (int i = 0; i < n; i++){
        bara[i] = modSwitchFromTorus32(x->a[i], Nk2);
    }

    for (int32_t i = 0; i < k2; i++) {
        for (int32_t j = 0; j < N; j++) {
            (testvect+i)->coefsT[j] =  modSwitchToTorus32(k2*j + i, Nk2); // identity on [0, 1/2)
        }
    }

    // Bootstrapping rotation and extraction
    ext_blindRotateAndExtract_FFT(cks, testvect, bk->bkFFT, barb, bara, n, ad_bit, bk_params);
    lweKeySwitch(result, bk->ks, cks);

    delete[] bara;
    delete_TorusPolynomial_array(k2, testvect);
    delete_LweSample(cks);
}

void Comp_even_Rh2(LweSample *result, const LweBootstrappingKeyFFT *bk, int32_t ad_bit, const LweSample *x, const int32_t interval) {
    const TGswParams *bk_params = bk->bk_params;
    const TLweParams *accum_params = bk->accum_params;
    const LweParams *in_params = bk->in_out_params;
    const int32_t N = accum_params->N;
    const int32_t n = in_params->n;
    const int32_t k2 = pow(2, ad_bit);
    const int32_t Nk2 = 2*N*k2;
    const int32_t Nk = N*k2;
    LweSample* cks = new_LweSample(&accum_params->extracted_lweparams);

    TorusPolynomial *testvect = new_TorusPolynomial_array(k2, N);
    int32_t *bara = new int32_t[N];


    int32_t barb = modSwitchFromTorus32(x->b, Nk2);
    for (int i = 0; i < n; i++){
        bara[i] = modSwitchFromTorus32(x->a[i], Nk2);
    }
    
    // abs(x) - 1/4
    for (int32_t i = 0; i < k2; i++) {
        for (int32_t j = 0; j < N; j++) {
            (testvect+i)->coefsT[j] = modSwitchToTorus32(k2*j + i, Nk2) - modSwitchToTorus32(1, 4); // abs(x) - 1/4
        }
    }

    // Bootstrapping rotation and extraction
    ext_blindRotateAndExtract_FFT(cks, testvect, bk->bkFFT, barb, bara, n, ad_bit, bk_params);
    lweKeySwitch(result, bk->ks, cks);
    // abs(x) - 1/4 + 1/4
    result->b += modSwitchToTorus32(1, 4);
    
    delete[] bara;
    delete_TorusPolynomial_array(k2, testvect);
    delete_LweSample(cks);
}


void Comp_funceval(TLweSample *result, TFheGateBootstrappingSecretKeySet* key, TorusPolynomial* testvect, int32_t ad_bit, const LweSample *x, const int32_t msize) {
    const LweBootstrappingKeyFFT *bk = key->cloud.bkFFT;
    const TGswParams *bk_params = bk->bk_params;
    const TLweParams *accum_params = bk->accum_params;
    const LweParams *in_params = bk->in_out_params;
    const int32_t N = accum_params->N;
    const int32_t n = in_params->n;
    const int32_t k2 = pow(2, ad_bit);
    const int32_t Nk2 = 2*N*k2;
    const int32_t Nk = N*k2;
    const int32_t N2kM = Nk2 / msize; 
    TorusPolynomial* tempvect = new_TorusPolynomial_array(k2, N);
    LweSample* ctemp = new_LweSample(in_params);
    lweCopy(ctemp, x, bk->in_out_params);
    ctemp->b += modSwitchToTorus32(1, 2*msize);

    int32_t *bara = new int32_t[N];


    int32_t barb = modSwitchFromTorus32(ctemp->b, Nk2);
    int32_t dphase = barb;
    for (int i = 0; i < n; i++){
        bara[i] = modSwitchFromTorus32(ctemp->a[i], Nk2);
        dphase -= bara[i] * key->lwe_key->key[i];
    }
    dphase = (dphase%Nk2 + Nk2)%Nk2;

    ext_blindRotate_woExtract_FFT(result, testvect, bk->bkFFT, barb, bara, n, ad_bit, bk_params);
    torusPolynomialArrayMulByXai(tempvect, k2, Nk2 - dphase, testvect);

    for (int i = 0; i < k2; i++){
        torusPolynomialSubTo((result + i )->b, (tempvect + i));
    }


    // ext_blindRotateAndExtract_FFT(cks, testvect, bk->bkFFT, barb, bara, n, ad_bit, bk_params);
    // lweKeySwitch(result, bk->ks, cks);

    delete[] bara;
    delete_TorusPolynomial_array(k2, tempvect);
    delete_LweSample(ctemp);
}

void Comp_EBS_Modular(TLweSample *result, TFheGateBootstrappingSecretKeySet* key, int32_t ad_bit, const LweSample *x, const int32_t msize) {
    const LweBootstrappingKeyFFT *bk = key->cloud.bkFFT;
    const TGswParams *bk_params = bk->bk_params;
    const TLweParams *accum_params = bk->accum_params;
    const LweParams *in_params = bk->in_out_params;
    const int32_t N = accum_params->N;
    const int32_t n = in_params->n;
    const int32_t k2 = pow(2, ad_bit);
    const int32_t Nk2 = 2*N*k2;
    const int32_t Nk = N*k2;
    const int32_t N2kM = Nk2 / msize;    

    TorusPolynomial* fpm_LUT = new_TorusPolynomial_array(k2, N); // f(m)
    TorusPolynomial* fmm_LUT = new_TorusPolynomial_array(k2, N); // f(-m)
    TorusPolynomial* odd_LUT = new_TorusPolynomial_array(k2, N);
    TorusPolynomial* even_LUT = new_TorusPolynomial_array(k2, N);

    LweSample* ch1 = new_LweSample(in_params);
    LweSample* ch2 = new_LweSample(in_params);
    TLweSample* cGeh2 = new_TLweSample_array(k2, accum_params);
    auto distr = std::bind(std::uniform_int_distribution<int32_t>{std::numeric_limits<int32_t>::min(),std::numeric_limits<int32_t>::max()}, std::mt19937(std::random_device{}()));

    for (int32_t i = 0; i < k2; i++) {
        for (int32_t j = 0; j < N; j++) {
            int32_t m = (i + k2*j)/N2kM;
            // (fpm_LUT+i)->coefsT[j] = modSwitchToTorus32(1 + 2*m, 2*msize);
            // (fmm_LUT+i)->coefsT[j] = modSwitchToTorus32(2*msize - 1, 2*msize);

            (fpm_LUT+i)->coefsT[j] = distr();
            (fmm_LUT+i)->coefsT[j] = distr();

            // (odd_LUT+i)->coefsT[j] =  (fpm_LUT+i)->coefsT[j] - (fmm_LUT+i)->coefsT[j];
            // (even_LUT+i)->coefsT[j] =  (fpm_LUT+i)->coefsT[j] + (fmm_LUT+i)->coefsT[j];
            (odd_LUT+i)->coefsT[j] =  (fpm_LUT+i)->coefsT[j];
            (even_LUT+i)->coefsT[j] = (fmm_LUT+i)->coefsT[j];
        }
        // torusPolynomialSubTo(odd_LUT+i, fmm_LUT+i);
        // torusPolynomialAddTo(even_LUT+i, fmm_LUT+i);
    }

    Comp_odd_h1(ch1, bk, ad_bit, x, msize);
    Comp_funceval(result, key, odd_LUT, ad_bit, ch1, msize);
    Comp_even_h2(ch2, bk, ad_bit, x, msize);
    Comp_funceval(cGeh2, key, even_LUT, ad_bit, ch2, msize);
    for (int i = 0; i < k2; i++){
        tLweAddTo(result + i, cGeh2 + i, accum_params);
    }
    // lweAddTo(result, cGeh2, in_params);


    delete_LweSample(ch1);
    delete_LweSample(ch2);
    delete_TLweSample_array(k2, cGeh2);
    delete_TorusPolynomial_array(k2, fpm_LUT);
    delete_TorusPolynomial_array(k2, fmm_LUT);
    delete_TorusPolynomial_array(k2, odd_LUT);
    delete_TorusPolynomial_array(k2, even_LUT);
}

void Comp_Rfunceval(TLweSample *result, TFheGateBootstrappingSecretKeySet* key, TorusPolynomial* testvect, int32_t ad_bit, const LweSample *x, int32_t interval) {
    const LweBootstrappingKeyFFT *bk = key->cloud.bkFFT;
    const TGswParams *bk_params = bk->bk_params;
    const TLweParams *accum_params = bk->accum_params;
    const LweParams *in_params = bk->in_out_params;
    const int32_t N = accum_params->N;
    const int32_t n = in_params->n;
    const int32_t k2 = pow(2, ad_bit);
    const int32_t Nk2 = 2*N*k2;
    const int32_t Nk = N*k2;
    // const int32_t N2kM = Nk2 / msize; 
    TorusPolynomial* tempvect = new_TorusPolynomial_array(k2, N);
    LweSample* ctemp = new_LweSample(in_params);
    lweCopy(ctemp, x, bk->in_out_params);
    // ctemp->b += modSwitchToTorus32(1, 2*msize);

    int32_t *bara = new int32_t[N];

    int32_t barb = modSwitchFromTorus32(ctemp->b, Nk2);
    int32_t dphase = barb;
    for (int i = 0; i < n; i++){
        bara[i] = modSwitchFromTorus32(ctemp->a[i], Nk2);
        dphase -= bara[i] * key->lwe_key->key[i];
    }
    dphase = (dphase%Nk2 + Nk2)%Nk2;

    ext_blindRotate_woExtract_FFT(result, testvect, bk->bkFFT, barb, bara, n, ad_bit, bk_params);
    torusPolynomialArrayMulByXai(tempvect, k2, Nk2 - dphase, testvect);

    for (int i = 0; i < k2; i++){
        torusPolynomialSubTo((result + i )->b, (tempvect + i));
    }


    // ext_blindRotateAndExtract_FFT(cks, testvect, bk->bkFFT, barb, bara, n, ad_bit, bk_params);
    // lweKeySwitch(result, bk->ks, cks);

    delete[] bara;
    delete_TorusPolynomial_array(k2, tempvect);
    delete_LweSample(ctemp);
}

void Comp_EBS_Real(TLweSample *result, TFheGateBootstrappingSecretKeySet* key, int32_t ad_bit, const LweSample *x, int32_t interval) {
    const LweBootstrappingKeyFFT *bk = key->cloud.bkFFT;
    const TGswParams *bk_params = bk->bk_params;
    const TLweParams *accum_params = bk->accum_params;
    const LweParams *in_params = bk->in_out_params;
    const int32_t N = accum_params->N;
    const int32_t n = in_params->n;
    const int32_t k2 = pow(2, ad_bit);
    const int32_t Nk2 = 2*N*k2;
    const int32_t Nk = N*k2;
    // const int32_t N2kM = Nk2 / msize;    
    double delta = pow(2., 32)/(interval*4);

    TorusPolynomial* fpm_LUT = new_TorusPolynomial_array(k2, N); // f(m)
    TorusPolynomial* fmm_LUT = new_TorusPolynomial_array(k2, N); // f(-m)
    TorusPolynomial* odd_LUT = new_TorusPolynomial_array(k2, N);
    TorusPolynomial* even_LUT = new_TorusPolynomial_array(k2, N);

    LweSample* ch1 = new_LweSample(in_params);
    LweSample* ch2 = new_LweSample(in_params);
    TLweSample* cGeh2 = new_TLweSample_array(k2, accum_params);
    auto distr = std::bind(std::uniform_int_distribution<int32_t>{std::numeric_limits<int32_t>::min(),std::numeric_limits<int32_t>::max()}, std::mt19937(std::random_device{}()));

    for (int32_t i = 0; i < k2; i++) {
        for (int32_t j = 0; j < N; j++) {
            double mp = interval * (i + k2*j)/(double) Nk;
            double mm = - mp;
            // (fpm_LUT+i)->coefsT[j] = (Torus32) ((double) delta * (43*sin(mp*boost::math::double_constants::pi / 4))); // exp1
            // (fpm_LUT+i)->coefsT[j] = (Torus32) ((double) delta * (40*(exp(mp/24) - exp(-mp/24))/(exp(mp/24)+exp(-mp/24)))); // exp2
            (fpm_LUT+i)->coefsT[j] = (Torus32) ((double) delta * (43*sin(mp*boost::math::double_constants::pi / 32))); // exp3
            // (fmm_LUT+i)->coefsT[j] = (Torus32) ((double) delta * (43*sin(mm*boost::math::double_constants::pi / 4))); // exp1
            // (fmm_LUT+i)->coefsT[j] = (Torus32) ((double) delta * (40*(exp(mm/24) - exp(-mm/24))/(exp(mm/24)+exp(-mm/24))));
            (fmm_LUT+i)->coefsT[j] = (Torus32) ((double) delta * (43*sin(mm*boost::math::double_constants::pi / 32))); // exp3

            // (fpm_LUT+i)->coefsT[j] = 0;
            // (fmm_LUT+i)->coefsT[j] = 0;

            (odd_LUT+i)->coefsT[j] =  (fpm_LUT+i)->coefsT[j] - (fmm_LUT+i)->coefsT[j];
            (even_LUT+i)->coefsT[j] =  (fpm_LUT+i)->coefsT[j] + (fmm_LUT+i)->coefsT[j];
            // (odd_LUT+i)->coefsT[j] =  (fpm_LUT+i)->coefsT[j];
            // (even_LUT+i)->coefsT[j] = (fmm_LUT+i)->coefsT[j];
        }
        // torusPolynomialSubTo(odd_LUT+i, fmm_LUT+i);
        // torusPolynomialAddTo(even_LUT+i, fmm_LUT+i);
    }

    Comp_odd_Rh1(ch1, bk, ad_bit, x, interval);
    Comp_funceval(result, key, odd_LUT, ad_bit, ch1, interval);
    Comp_even_Rh2(ch2, bk, ad_bit, x, interval);
    Comp_funceval(cGeh2, key, even_LUT, ad_bit, ch2, interval);
    for (int i = 0; i < k2; i++){
        tLweAddTo(result + i, cGeh2 + i, accum_params);
    }
    // lweAddTo(result, cGeh2, in_params);


    delete_LweSample(ch1);
    delete_LweSample(ch2);
    delete_TLweSample_array(k2, cGeh2);
    delete_TorusPolynomial_array(k2, fpm_LUT);
    delete_TorusPolynomial_array(k2, fmm_LUT);
    delete_TorusPolynomial_array(k2, odd_LUT);
    delete_TorusPolynomial_array(k2, even_LUT);
}

///////////////////////////////////////////////////////////////////////////


void discretize_LWE(LweSample* c, LweSample* sample, int32_t N, const LweParams* lweparam){
    int32_t n = lweparam->n;
    c->b = modSwitchFromTorus32(sample->b, 2*N);
    for (int i = 0; i < n; i++){
        c->a[i] = modSwitchFromTorus32(sample->a[i], 2*N);
    }
}

void prec_est(TFheGateBootstrappingSecretKeySet* key) {
    const LweParams* lweparam = key->params->in_out_params;
    int32_t n = key->params->in_out_params->n;
    int32_t N = key->params->tgsw_params->tlwe_params->N;
    int32_t ham = 0;
    const LweKey *lwekey = key->lwe_key;
    for (int i = 0; i < n; i++){
        ham += lwekey->key[i];
    }
    int32_t enu = pow(2, 16);
    accumulator_set<double, stats<tag::max, tag::min, tag::mean, tag::variance> > acc;
    LweSample* ca = new_LweSample(lweparam);
    lweClear(ca, lweparam);

    for (int i = 0; i < enu; i++){
        lweSymEncrypt(ca, 0, lweparam->alpha_min, lwekey);
        LweSample* DisLWE = new_LweSample(lweparam);
        lweClear(DisLWE, lweparam);
        discretize_LWE(DisLWE, ca, N, lweparam);
        int32_t m = lwePhase(DisLWE, lwekey);
        Torus32 T = modSwitchToTorus32(m, 2*N);
        double e = T / pow(2., 32);
        acc(e);
    }

    std::cout << "Estimated Variance (n+1)/48N^2  (log2) : " << log2( (double) (n+1)/(48*(N*N))) << "\n";
    std::cout << "Hamming Weight Ham(s) : " << ham << "\n";
    std::cout << "Estimated with hamming weight (h+1)/48N^2 (log2) : " << log2( (double) (ham+1) / (48*N*N)) << "\n \n";

    std::cout << "Estimated Stdev (log2) : " << log2( (double) (n+1)/(48*(N*N)))/2  << "\n";
    std::cout << "Hamming Weight Ham(s) : " << ham << "\n";
    std::cout << "Estimated with hamming weight (h+1)/48N^2 (log2) : " << log2( (double) (ham+1) / (48*N*N))/2  << "\n";
    std::cout << "============================================================================== \n";

    std::cout << "MS Max :   " << boost::accumulators::max(acc) << std::endl;
    std::cout << "MS min :   " << boost::accumulators::min(acc) << std::endl;
    std::cout << "MS Mean :   " << boost::accumulators::mean(acc) << std::endl;
    std::cout << "MS Variance (log2) :   " << log2(boost::accumulators::variance(acc)) << std::endl;
    std::cout << "MS Stdev (log2) :   " << log2(boost::accumulators::variance(acc))/2 << std::endl;
    std::cout << "============================================================================== \n";
    
}

void std_est(TFheGateBootstrappingSecretKeySet* key, TLweKeySwitchKey* rsk, int32_t* RS, int32_t adbit) {
    const LweParams* lweparam = key->params->in_out_params;
    const TLweParams* tlweparam = key->params->tgsw_params->tlwe_params;
    int32_t n = key->params->in_out_params->n;
    int32_t N = key->params->tgsw_params->tlwe_params->N;
    int32_t nu2 = std::pow(2, adbit);
    int32_t nuN = nu2*N;
    int32_t nu2N = nu2*2*N; 
    const LweKey *lwekey = key->lwe_key; 
    const TLweKey* tlwe_key = &key->tgsw_key->tlwe_key;  

    accumulator_set<double, stats<tag::max, tag::min, tag::mean, tag::variance> > acc1, acc2, acc3, acc4;
    LweSample* ca = new_LweSample(lweparam);
    lweClear(ca, lweparam);
    LweSample* DisLWE = new_LweSample(lweparam);
    lweClear(DisLWE, lweparam);
    TLweSample* switchlwe = new_TLweSample(tlweparam);
    TorusPolynomial* dectlwe = new_TorusPolynomial(N);
    TLweSample* tmptlwe = new_TLweSample_array(nu2, tlweparam);
    LweSample* templwe = new_LweSample_array(nuN, &tlweparam->extracted_lweparams);
    LweSample* reslwe = new_LweSample_array(nuN, lweparam);
    TorusPolynomial* tmp_poly = new_TorusPolynomial_array(nu2, N);
    int32_t enu = pow(2, 16)/nuN;

    for (int i = 0; i < enu; i++){
        lweSymEncrypt(ca, 0, lweparam->alpha_min, lwekey);
        HDEBS_varest(tmptlwe, key, adbit, ca, nuN);
        for(int j = 0; j < nu2; j++){
            tLwePhase(tmp_poly + j, tmptlwe + j, tlwe_key);
            for (int k = 0; k < N; k++){
                acc1(((tmp_poly+j)->coefsT[k])/pow(2., 32));
                tLweExtractLweSampleIndex(templwe + (j*N + k),tmptlwe + j, k, lweparam, tlweparam);
                lweKeySwitch(reslwe + (N * j + k), key->cloud.bkFFT->ks, templwe + (j * N + k)); 
                int32_t phs = lwePhase(reslwe + (N * j + k), lwekey);
                acc2(((tmp_poly+j)->coefsT[k] - phs)/pow(2., 32));
                acc3(phs/pow(2., 32));
            }
        }
        lweSymEncrypt(ca, 0, 0, lwekey);
        TLweKeySwitch(switchlwe, rsk, ca);
        tLwePhase(dectlwe, switchlwe, tlwe_key);
        for (int k = 0; k < N; k++){
            acc4((dectlwe->coefsT[i])/pow(2., 32));
        }
        
    }    

    std::cout << "BlindRotate Max :   " << boost::accumulators::max(acc1) << std::endl;
    std::cout << "BlindRotate min :   " << boost::accumulators::min(acc1) << std::endl;
    std::cout << "BlindRotate Mean :   " << boost::accumulators::mean(acc1) << std::endl;
    std::cout << "BlindRotate Variance (log2) :   " << log2(boost::accumulators::variance(acc1)) << std::endl;
    std::cout << "BlindRotate Stdev (log2) :   " << log2(boost::accumulators::variance(acc1))/2 << std::endl;
    std::cout << "============================================================================== \n";

    std::cout << "KeySwitch Max :   " << boost::accumulators::max(acc2) << std::endl;
    std::cout << "KeySwitch min :   " << boost::accumulators::min(acc2) << std::endl;
    std::cout << "KeySwitch Mean :   " << boost::accumulators::mean(acc2) << std::endl;
    std::cout << "KeySwitch Variance (log2) :   " << log2(boost::accumulators::variance(acc2)) << std::endl;
    std::cout << "KeySwitch Stdev (log2) :   " << log2(boost::accumulators::variance(acc2))/2 << std::endl;
    std::cout << "============================================================================== \n";

    std::cout << "Bootstrapping Max :   " << boost::accumulators::max(acc3) << std::endl;
    std::cout << "Bootstrapping min :   " << boost::accumulators::min(acc3) << std::endl;
    std::cout << "Bootstrapping Mean :   " << boost::accumulators::mean(acc3) << std::endl;
    std::cout << "Bootstrapping Variance (log2) :   " << log2(boost::accumulators::variance(acc3)) << std::endl;
    std::cout << "Bootstrapping Stdev (log2) :   " << log2(boost::accumulators::variance(acc3))/2 << std::endl;
    std::cout << "============================================================================== \n";

    std::cout << "LWE-to-TRLWE switch Max :   " << boost::accumulators::max(acc4) << std::endl;
    std::cout << "LWE-to-TRLWE switch min :   " << boost::accumulators::min(acc4) << std::endl;
    std::cout << "LWE-to-TRLWE switch Mean :   " << boost::accumulators::mean(acc4) << std::endl;
    std::cout << "LWE-to-TRLWE switch Variance (log2) :   " << log2(boost::accumulators::variance(acc4)) << std::endl;
    std::cout << "LWE-to-TRLWE switch Stdev(log2) :   " << log2(boost::accumulators::variance(acc4))/2 << std::endl;
    std::cout << "============================================================================== \n";
}

void fd_est(TFheGateBootstrappingSecretKeySet* key, TLweKeySwitchKey* rsk, int32_t* PM, int32_t adbit) {
    const LweParams* lweparam = key->params->in_out_params;
    const TLweParams* tlweparam = key->params->tgsw_params->tlwe_params;
    int32_t n = key->params->in_out_params->n;
    int32_t N = key->params->tgsw_params->tlwe_params->N;
    int32_t nu2 = std::pow(2, adbit);
    int32_t nuN = nu2*N;
    int32_t nu2N = nu2*2*N; 
    const LweKey *lwekey = key->lwe_key; 
    const TLweKey* tlwe_key = &key->tgsw_key->tlwe_key;  

    accumulator_set<double, stats<tag::max, tag::min, tag::mean, tag::variance> > acc1, acc2, acc3, acc4;
    Torus32 mu = modSwitchToTorus32(1, 4); 
    LweSample* ca = new_LweSample(lweparam);
    lweClear(ca, lweparam);
    LweSample* DisLWE = new_LweSample(lweparam);
    lweClear(DisLWE, lweparam);
    TLweSample* tmptlwe = new_TLweSample_array(nu2, tlweparam);
    TLweSample* restlwe = new_TLweSample_array(nu2, tlweparam);
    LweSample* templwe = new_LweSample_array(nuN, &tlweparam->extracted_lweparams);
    LweSample* reslwe = new_LweSample_array(nuN, lweparam);
    TorusPolynomial* tmp_poly = new_TorusPolynomial_array(nu2, N);
    TorusPolynomial* tmp_poly2 = new_TorusPolynomial_array(nu2, N);
    int32_t enu = pow(2, 16)/nuN;

    for (int i = 0; i < enu; i++){
        lweSymEncrypt(ca, mu, lweparam->alpha_min, lwekey);

        ///// FDFB ////
        FDFB_EBS_Modular(tmptlwe, restlwe, key, rsk, adbit, ca, PM, nu2N);
        for(int j = 0; j < nu2; j++){
            // tLwePhase(tmp_poly + j, tmptlwe + j, tlwe_key);
            tLwePhase(tmp_poly2 + j, restlwe + j, tlwe_key);
            for (int k = 0; k < N; k++){
                tLweExtractLweSampleIndex(templwe + (j*N + k),tmptlwe + j, k, lweparam, tlweparam);
                lweKeySwitch(reslwe + (N * j + k), key->cloud.bkFFT->ks, templwe + (j * N + k)); 
                int32_t phs = lwePhase(reslwe + (N * j + k), lwekey);
                acc1(phs/pow(2., 32));

                int32_t accum_err = (tmp_poly2 + j) -> coefsT[k];
                acc2(accum_err/pow(2., 32));
            }
        }
        ///// TOTA ////
        TOTA_EBS_Modular(tmptlwe, key, adbit, ca, nuN);
        for(int j = 0; j < nu2; j++){
            // tLwePhase(tmp_poly + j, tmptlwe + j, tlwe_key);
            for (int k = 0; k < N; k++){
                tLweExtractLweSampleIndex(templwe + (j*N + k),tmptlwe + j, k, lweparam, tlweparam);
                lweKeySwitch(reslwe + (N * j + k), key->cloud.bkFFT->ks, templwe + (j * N + k)); 
                int32_t phs = lwePhase(reslwe + (N * j + k), lwekey);
                acc3(phs/pow(2., 32));
            }
        }
        ///// COMP ////
        Comp_EBS_Modular(tmptlwe, key, adbit, ca, nuN);
        for(int j = 0; j < nu2; j++){
            // tLwePhase(tmp_poly + j, tmptlwe + j, tlwe_key);
            for (int k = 0; k < N; k++){
                tLweExtractLweSampleIndex(templwe + (j*N + k),tmptlwe + j, k, lweparam, tlweparam);
                lweKeySwitch(reslwe + (N * j + k), key->cloud.bkFFT->ks, templwe + (j * N + k)); 
                int32_t phs = lwePhase(reslwe + (N * j + k), lwekey);
                acc4(phs/pow(2., 32));
            }
        }
    }

    std::cout << "FDFB Max :   " << boost::accumulators::max(acc1) << std::endl;
    std::cout << "FDFB min :   " << boost::accumulators::min(acc1) << std::endl;
    std::cout << "FDFB Mean :   " << boost::accumulators::mean(acc1) << std::endl;
    std::cout << "FDFB Variance (log2) :   " << log2(boost::accumulators::variance(acc1)) << std::endl;
    std::cout << "FDFB stdev (log2) :   " << log2(boost::accumulators::variance(acc1))/2 << std::endl;
    std::cout << "============================================================================== \n";

    std::cout << "FDFB ACCUM Max :   " << boost::accumulators::max(acc2) << std::endl;
    std::cout << "FDFB ACCUM min :   " << boost::accumulators::min(acc2) << std::endl;
    std::cout << "FDFB ACCUM  Mean :   " << boost::accumulators::mean(acc2) << std::endl;
    std::cout << "FDFB ACCUM Variance (log2) :   " << log2(boost::accumulators::variance(acc2)) << std::endl;
    std::cout << "FDFB ACCUM stdev (log2) :   " << log2(boost::accumulators::variance(acc2))/2 << std::endl;
    std::cout << "============================================================================== \n";

    std::cout << "TOTA Max :   " << boost::accumulators::max(acc3) << std::endl;
    std::cout << "TOTA min :   " << boost::accumulators::min(acc3) << std::endl;
    std::cout << "TOTA Mean :   " << boost::accumulators::mean(acc3) << std::endl;
    std::cout << "TOTA Variance (log2) :   " << log2(boost::accumulators::variance(acc3)) << std::endl;
    std::cout << "TOTA stdev (log2) :   " << log2(boost::accumulators::variance(acc3))/2 << std::endl;
    std::cout << "============================================================================== \n";

    std::cout << "Comp Max :   " << boost::accumulators::max(acc4) << std::endl;
    std::cout << "Comp min :   " << boost::accumulators::min(acc4) << std::endl;
    std::cout << "Comp Mean :   " << boost::accumulators::mean(acc4) << std::endl;
    std::cout << "Comp Variance (log2) :   " << log2(boost::accumulators::variance(acc4)) << std::endl;
    std::cout << "Comp stdev (log2) :   " << log2(boost::accumulators::variance(acc4))/2 << std::endl;
    std::cout << "============================================================================== \n";


}

void fd_est_real(TFheGateBootstrappingSecretKeySet* key, TLweKeySwitchKey* rsk, int32_t* PM, int32_t adbit, int32_t interval) {
    const LweParams* lweparam = key->params->in_out_params;
    const TLweParams* tlweparam = key->params->tgsw_params->tlwe_params;
    int32_t n = key->params->in_out_params->n;
    int32_t N = key->params->tgsw_params->tlwe_params->N;
    int32_t nu2 = std::pow(2, adbit);
    int32_t nuN = nu2*N;
    int32_t nu2N = nu2*2*N; 
    const LweKey *lwekey = key->lwe_key; 
    const TLweKey* tlwe_key = &key->tgsw_key->tlwe_key;  

    accumulator_set<double, stats<tag::max, tag::min, tag::mean, tag::variance> > acc1, acc2, acc3, acc4;
    Torus32 mu = modSwitchToTorus32(1, 4); 
    LweSample* ca = new_LweSample(lweparam);
    lweClear(ca, lweparam);
    LweSample* DisLWE = new_LweSample(lweparam);
    lweClear(DisLWE, lweparam);
    TLweSample* tmptlwe = new_TLweSample_array(nu2, tlweparam);
    TLweSample* restlwe = new_TLweSample_array(nu2, tlweparam);
    LweSample* templwe = new_LweSample_array(nuN, &tlweparam->extracted_lweparams);
    LweSample* reslwe = new_LweSample_array(nuN, lweparam);
    TorusPolynomial* tmp_poly = new_TorusPolynomial_array(nu2, N);
    TorusPolynomial* tmp_poly2 = new_TorusPolynomial_array(nu2, N);
    int32_t enu = pow(2, 16)/nuN;
    double delta = pow(2., 32)/(interval*2);

    for (int i = 0; i < enu; i++){
        lweSymEncrypt(ca, mu, lweparam->alpha_min, lwekey);

        ///// FDFB ////
        FDFB_EBS_Real(tmptlwe, restlwe, key, rsk, adbit, ca, PM, interval);
        for(int j = 0; j < nu2; j++){
            // tLwePhase(tmp_poly + j, tmptlwe + j, tlwe_key);
            tLwePhase(tmp_poly2 + j, restlwe + j, tlwe_key);
            for (int k = 0; k < N; k++){
                tLweExtractLweSampleIndex(templwe + (j*N + k),tmptlwe + j, k, lweparam, tlweparam);
                lweKeySwitch(reslwe + (N * j + k), key->cloud.bkFFT->ks, templwe + (j * N + k)); 
                int32_t phs = lwePhase(reslwe + (N * j + k), lwekey);
                acc1((phs/delta)/(interval*2));

                int32_t accum_err = (tmp_poly2 + j) -> coefsT[k];
                acc2((accum_err/delta)/(interval*2));
            }
        }
        ///// TOTA ////
        TOTA_EBS_Real(tmptlwe, key, adbit, ca, interval);
        for(int j = 0; j < nu2; j++){
            // tLwePhase(tmp_poly + j, tmptlwe + j, tlwe_key);
            for (int k = 0; k < N; k++){
                tLweExtractLweSampleIndex(templwe + (j*N + k),tmptlwe + j, k, lweparam, tlweparam);
                lweKeySwitch(reslwe + (N * j + k), key->cloud.bkFFT->ks, templwe + (j * N + k)); 
                int32_t phs = lwePhase(reslwe + (N * j + k), lwekey);
                acc3((phs/delta)/(interval*2));
            }
        }
        ///// COMP ////
        Comp_EBS_Real(tmptlwe, key, adbit, ca, interval);
        for(int j = 0; j < nu2; j++){
            // tLwePhase(tmp_poly + j, tmptlwe + j, tlwe_key);
            for (int k = 0; k < N; k++){
                tLweExtractLweSampleIndex(templwe + (j*N + k),tmptlwe + j, k, lweparam, tlweparam);
                lweKeySwitch(reslwe + (N * j + k), key->cloud.bkFFT->ks, templwe + (j * N + k)); 
                int32_t phs = lwePhase(reslwe + (N * j + k), lwekey);
                acc4((phs/delta)/(interval*2));
            }
        }
    }

    std::cout << "FDFB Max :   " << boost::accumulators::max(acc1) << std::endl;
    std::cout << "FDFB min :   " << boost::accumulators::min(acc1) << std::endl;
    std::cout << "FDFB Mean :   " << boost::accumulators::mean(acc1) << std::endl;
    std::cout << "FDFB Variance (log2) :   " << log2(boost::accumulators::variance(acc1)) << std::endl;
    std::cout << "FDFB stdev (log2) :   " << log2(boost::accumulators::variance(acc1))/2 << std::endl;
    std::cout << "============================================================================== \n";

    std::cout << "FDFB ACCUM Max :   " << boost::accumulators::max(acc2) << std::endl;
    std::cout << "FDFB ACCUM min :   " << boost::accumulators::min(acc2) << std::endl;
    std::cout << "FDFB ACCUM  Mean :   " << boost::accumulators::mean(acc2) << std::endl;
    std::cout << "FDFB ACCUM Variance (log2) :   " << log2(boost::accumulators::variance(acc2)) << std::endl;
    std::cout << "FDFB ACCUM stdev (log2) :   " << log2(boost::accumulators::variance(acc2))/2 << std::endl;
    std::cout << "============================================================================== \n";

    std::cout << "TOTA Max :   " << boost::accumulators::max(acc3) << std::endl;
    std::cout << "TOTA min :   " << boost::accumulators::min(acc3) << std::endl;
    std::cout << "TOTA Mean :   " << boost::accumulators::mean(acc3) << std::endl;
    std::cout << "TOTA Variance (log2) :   " << log2(boost::accumulators::variance(acc3)) << std::endl;
    std::cout << "TOTA stdev (log2) :   " << log2(boost::accumulators::variance(acc3))/2 << std::endl;
    std::cout << "============================================================================== \n";

    std::cout << "Comp Max :   " << boost::accumulators::max(acc4) << std::endl;
    std::cout << "Comp min :   " << boost::accumulators::min(acc4) << std::endl;
    std::cout << "Comp Mean :   " << boost::accumulators::mean(acc4) << std::endl;
    std::cout << "Comp Variance (log2) :   " << log2(boost::accumulators::variance(acc4)) << std::endl;
    std::cout << "Comp stdev (log2) :   " << log2(boost::accumulators::variance(acc4))/2 << std::endl;
    std::cout << "============================================================================== \n";


}

