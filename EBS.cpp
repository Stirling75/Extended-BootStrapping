#include "EBS.h"
#include "tlwekeyswitch.h"
#include <bits/c++config.h>
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

using namespace std;
using namespace boost::multiprecision;
using boost::multiprecision::cpp_dec_float_100;


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
    #pragma omp parallel for num_threads(min(k2, 24))
    // #pragma omp parallel num_threads(min(k2/2, 32))
    // #pragma omp parallel for
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

/////////////////////////////////////////////////////////// TOTA ///////////////////////////////////////////////////////////

// TOTA - Modswitch to Z_{2^kN}, BlindRotate in Z_{2^(k+1)N}
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

void TOTA_Rfunceval_woKS_FFT(LweSample *result, const LweBootstrappingKeyFFT *bk, int32_t ad_bit, const LweSample *x, const LweSample *signx, int32_t interval) {
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

    TorusPolynomial *testvect = new_TorusPolynomial_array(k2, N);
    int32_t *bara = new int32_t[N];


    int32_t barb = (modSwitchFromTorus32(x->b, Nk) + modSwitchFromTorus32(signx->b, Nk2)) % Nk2; 
    for (int i = 0; i < n; i++){
        bara[i] = (modSwitchFromTorus32(x->a[i], Nk) + modSwitchFromTorus32(signx->a[i], Nk2)) % Nk2;
    }

    // From -1/2 -> 0
    for (int32_t j = 0; j < N/2; j++) {
        for (int32_t i = 0; i < k2; i++) {
            double m = int_length*(j*k2 + i - halfNk)/(double)Nk;
            // (testvect+i)->coefsT[j] = (Torus32) ((double) delta * (m+50)*(m+7)*(m-50)/(2000)); 
            // (testvect+i)->coefsT[j] = (Torus32) ((double) delta * (43*sin(m*boost::math::double_constants::pi / 4))); 
            // (testvect+i)->coefsT[j] = (Torus32) ((double) delta * (40*(exp(m/24) - exp(-m/24))/(exp(m/24)+exp(-m/24))));
            (testvect+i)->coefsT[j] = (Torus32) ((double) delta * (43*sin(m*boost::math::double_constants::pi / 32))); 
        }
    }

    // 0 -> 1/2
    for (int32_t j = N/2; j < N; j++) {
        for (int32_t i = 0; i < k2; i++) {
            double m = int_length*(j*k2 + i - halfNk)/(double)Nk;
            // (testvect+i)->coefsT[j] = (Torus32) ((double) delta * ((m+50)*(m+7)*(m-50)/(2000))); 
            // (testvect+i)->coefsT[j] = (Torus32) ((double) delta * (43*sin(m*boost::math::double_constants::pi / 4))); 
            // (testvect+i)->coefsT[j] = (Torus32) ((double) delta * (40*(exp(m/24) - exp(-m/24))/(exp(m/24)+exp(-m/24))));
            (testvect+i)->coefsT[j] = (Torus32) ((double) delta * (43*sin(m*boost::math::double_constants::pi / 32))); 
        }
    }

    // Bootstrapping rotation and extraction
    ext_blindRotateAndExtract_FFT(result, testvect, bk->bkFFT, barb, bara, n, ad_bit, bk_params);


    delete[] bara;
    delete_TorusPolynomial_array(k2, testvect);
}

void TOTA_Mfunceval_woKS_FFT(LweSample *result, const LweBootstrappingKeyFFT *bk, int32_t ad_bit, const LweSample *x, const LweSample *signx, const int32_t msize) {

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


    int32_t barb = (modSwitchFromTorus32(x->b, Nk) + modSwitchFromTorus32(signx->b, Nk2)) % Nk2; 
    for (int i = 0; i < n; i++){
        bara[i] = (modSwitchFromTorus32(x->a[i], Nk) + modSwitchFromTorus32(signx->a[i], Nk2)) % Nk2;
    }

    // From 0 -> q-1 
    for (int32_t j = 0; j < N; j++) {
        for (int32_t i = 0; i < k2; i++) {
            int32_t m = (i + k2*j)/NkM;
            // int32_t m1 = m / sqrt(msize);
            // int32_t m2 = m - m1*sqrt(msize); 
            // (testvect+i)->coefsT[j] = modSwitchToTorus32((m1*m2)%((int32_t) M), M);
            // (testvect+i)->coefsT[j] = modSwitchToTorus32((m1*m2)/((int32_t) sqrt(msize)), msize);
            (testvect+i)->coefsT[j] = modSwitchToTorus32(m, msize);
        }
    }

    // Bootstrapping rotation and extraction
    ext_blindRotateAndExtract_FFT(result, testvect, bk->bkFFT, barb, bara, n, ad_bit, bk_params);


    delete[] bara;
    delete_TorusPolynomial_array(k2, testvect);
}

void TOTA_woKS_Real(LweSample *result, const LweBootstrappingKeyFFT *bk, int32_t ad_bit, const LweSample *x, int32_t interval ) {
    LweSample* c1 = new_LweSample(&bk->accum_params->extracted_lweparams);
    LweSample* c1ks = new_LweSample(bk->in_out_params);
    LweSample* ctemp = new_LweSample(bk->in_out_params);
    lweCopy(ctemp, x, bk->in_out_params);
    ctemp->b += modSwitchToTorus32(1, 2);

    TOTA_signeval_woKS_FFT(c1, bk, ad_bit, ctemp); // c1 = TLWE_{K}(1/2 * sign(?*2^k N))
    lweKeySwitch(c1ks, bk->ks, c1); 

    TOTA_Rfunceval_woKS_FFT(result, bk, ad_bit, ctemp, c1ks, interval);

    delete_LweSample(c1);
    delete_LweSample(c1ks);
    delete_LweSample(ctemp);
}

void TOTA_EBS_Real(LweSample *result, const LweBootstrappingKeyFFT *bk, int32_t ad_bit, const LweSample *x, int32_t interval) {
    LweSample* c1 = new_LweSample(&bk->accum_params->extracted_lweparams);
    LweSample* c1ks = new_LweSample(bk->in_out_params);
    LweSample* ctemp = new_LweSample(bk->in_out_params);
    lweCopy(ctemp, x, bk->in_out_params);
    ctemp->b += modSwitchToTorus32(1, 2);

    TOTA_signeval_woKS_FFT(c1, bk, ad_bit, ctemp); // c1 = TLWE_{K}(1/2 * sign(?*2^k N))
    lweKeySwitch(c1ks, bk->ks, c1); 

    TOTA_Rfunceval_woKS_FFT(c1, bk, ad_bit, ctemp, c1ks, interval);
    lweKeySwitch(result, bk->ks, c1);

    delete_LweSample(c1);
    delete_LweSample(c1ks);
    delete_LweSample(ctemp);
}

void TOTA_EBS_Modular(LweSample *result, const LweBootstrappingKeyFFT *bk, int32_t ad_bit, const LweSample *x, const int32_t msize) {
    LweSample* c1 = new_LweSample(&bk->accum_params->extracted_lweparams);
    LweSample* c1ks = new_LweSample(bk->in_out_params);
    LweSample* ctemp = new_LweSample(bk->in_out_params);
    lweCopy(ctemp, x, bk->in_out_params);
    ctemp->b += modSwitchToTorus32(1, 2*msize);

    TOTA_signeval_woKS_FFT(c1, bk, ad_bit, ctemp); // c1 = TLWE_{K}(1/2 * sign(?*2^k N))
    lweKeySwitch(c1ks, bk->ks, c1); 

    TOTA_Mfunceval_woKS_FFT(c1, bk, ad_bit, ctemp, c1ks, msize);
    lweKeySwitch(result, bk->ks, c1);

    delete_LweSample(c1);
    delete_LweSample(c1ks);
    delete_LweSample(ctemp);
}

void TOTA_woKS_Modular(LweSample *result, const LweBootstrappingKeyFFT *bk, int32_t ad_bit, const LweSample *x, const int32_t msize) {
    LweSample* c1 = new_LweSample(&bk->accum_params->extracted_lweparams);
    LweSample* c1ks = new_LweSample(bk->in_out_params);
    LweSample* ctemp = new_LweSample(bk->in_out_params);
    lweCopy(ctemp, x, bk->in_out_params);
    ctemp->b += modSwitchToTorus32(1, 2*msize);

    TOTA_signeval_woKS_FFT(c1, bk, ad_bit, ctemp); // c1 = TLWE_{K}(1/2 * sign(?*2^k N))
    lweKeySwitch(c1ks, bk->ks, c1); 

    TOTA_Mfunceval_woKS_FFT(result, bk, ad_bit, ctemp, c1ks, msize);

    delete_LweSample(c1);
    delete_LweSample(c1ks);
    delete_LweSample(ctemp);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void param_quality(TFheGateBootstrappingParameterSet* param, int32_t RS_param[2], int32_t PM_param[2], int32_t msize) {
    float128 n = param->in_out_params->n;
    float128 N = param->tgsw_params->tlwe_params->N;
    float128 k = param->tgsw_params->tlwe_params->k;
    float128 p = msize;

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

    float128 log2_VMS = log2(n+1) - 4 * log2(3) - 2 * log2(N);
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
    float128 log2_HDFB = log2(erf(1/(4*p*sqrt(2)*sqrt(V_MS+V_ct)))) + log2(erf(1/(4*p*sqrt(2)*sqrt(V_BS))));
    float128 log2err_HDFB = log2(1 - pow(2., log2_HDFB));

    float128 log2_FDFB = log2(erf(1/(2*p*sqrt(2)*sqrt(V_ct+V_MS)))) + log2(erf(1/(2*p*sqrt(2)*sqrt(V_FDFBACC + V_BS))));
    float128 log2err_FDFB = log2(1 - pow(2., log2_FDFB));

    float128 log2_TOTA = log2(erf(1/(2*p*sqrt(2)*sqrt(V_ct + 4*V_MS)))) + log2(erf(1/(2*p*sqrt(2)*sqrt(V_ct + V_BS + 5*V_MS)))) + log2(erf(1/(2*p*sqrt(2)*sqrt(V_BS))));
    float128 log2err_TOTA = log2(1 - pow(2., log2_TOTA));

    float128 log2_Comp = log2(erf(1/(2*p*sqrt(2)*sqrt(V_ct + V_MS)))) + 2*log2(erf(1/(2*p*sqrt(2)*sqrt(V_BS + V_MS)))) + log2(erf(1/(2*p*sqrt(2)*sqrt(2*V_BS))));
    float128 log2err_Comp = log2(1 - pow(2., log2_Comp));

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
    std::cout << BOLDWHITE "========================================================== \n" RESET;
    std::cout << "       Precision estimated with 4-sigma (99.9936%) \n";
    std::cout << "    Precision of output after BlindRotate : " << BOLDGREEN <<(int) floor(32 - (32 + log2_VBR/2 + 3)) << " Bits \n" RESET;
    std::cout << "      Precision of output after Bootstrap : " << BOLDRED << (int) floor(32 - (32 + log2_VBS/2 + 3)) << " Bits \n" RESET; 
    std::cout << "                      Desired value for k : " << BOLDCYAN << max(0, (int) ceil(log2_stdMS - log2_VBS/2)) << " ~ " << max(0, (int) ceil(log2_stdMS - log2_VBR/2)) << "\n" RESET;
    std::cout << BOLDWHITE "========================================================== \n" RESET;
    std::cout << BOLDWHITE "With Message space p = " << msize << " and extension factor nu = 0, \n" RESET;
    std::cout << BOLDRED "Log2 error rate for HDFB <= " << log2err_HDFB << "\n" RESET;
    std::cout << BOLDGREEN "Log2 error rate for FDFB <= " << log2err_FDFB << "\n" RESET;
    std::cout << BOLDYELLOW "Log2 error rate for TOTA <= " << log2err_TOTA << "\n" RESET;
    std::cout << BOLDBLUE "Log2 error rate for Comp <= " << log2err_Comp << "\n" RESET; 
    std::cout << BOLDWHITE "========================================================== \n" RESET;
    FDB_quality(V_ct, V_MS, V_BS, V_FDFBACC, p, 1);
    FDB_quality(V_ct, V_MS, V_BS, V_FDFBACC, p, 2);
    FDB_quality(V_ct, V_MS, V_BS, V_FDFBACC, p, 3);
    FDB_quality(V_ct, V_MS, V_BS, V_FDFBACC, p, 4);
    FDB_quality(V_ct, V_MS, V_BS, V_FDFBACC, p, 5);
    FDB_quality(V_ct, V_MS, V_BS, V_FDFBACC, p, 6);
    FDB_quality(V_ct, V_MS, V_BS, V_FDFBACC, p, 7);
}

void FDB_quality(float128 V_ct, float128 V_MS, float128 V_BS, float128 V_FDFBACC, float128 p, int32_t ad_bit) {
    float128 nu2 = 1<<(2*ad_bit);
    
    float128 log2_HDFB = log2(erf(1/(4*p*sqrt(2)*sqrt((V_MS/nu2) + V_ct)))) + log2(erf(1/(4*p*sqrt(2)*sqrt(V_BS))));
    float128 log2err_HDFB = log2(1 - pow(2., log2_HDFB));

    float128 log2_FDFB = log2(erf(1/(2*p*sqrt(2)*sqrt(V_ct + (V_MS/nu2))))) + log2(erf(1/(2*p*sqrt(2)*sqrt(V_FDFBACC + V_BS))));
    float128 log2err_FDFB = log2(1 - pow(2., log2_FDFB));

    float128 log2_TOTA = log2(erf(1/(2*p*sqrt(2)*sqrt(V_ct + (4*V_MS/nu2))))) + log2(erf(1/(2*p*sqrt(2)*sqrt(V_ct + V_BS + (5*V_MS/nu2))))) + log2(erf(1/(2*p*sqrt(2)*sqrt(V_BS))));
    float128 log2err_TOTA = log2(1 - pow(2., log2_TOTA));

    float128 log2_Comp = log2(erf(1/(2*p*sqrt(2)*sqrt(V_ct + (V_MS/nu2))))) + 2*log2(erf(1/(2*p*sqrt(2)*sqrt(V_BS + (V_MS/nu2))))) + log2(erf(1/(2*p*sqrt(2)*sqrt(2*V_BS))));
    float128 log2err_Comp = log2(1 - pow(2., log2_Comp));

    std::cout << BOLDWHITE "With Message space p = " << p << " and extension factor nu = "<< ad_bit <<",\n" RESET;
    std::cout << BOLDRED "Log2 error rate for HDFB <= " << log2err_HDFB << "\n" RESET;
    std::cout << BOLDGREEN "Log2 error rate for FDFB <= " << log2err_FDFB << "\n" RESET;
    std::cout << BOLDYELLOW "Log2 error rate for TOTA <= " << log2err_TOTA << "\n" RESET;
    std::cout << BOLDBLUE "Log2 error rate for Comp <= " << log2err_Comp << "\n" RESET; 
    std::cout << BOLDWHITE "========================================================== \n" RESET;
}

void param_quality_f100(TFheGateBootstrappingParameterSet* param, int32_t RS_param[2], int32_t PM_param[2], int32_t msize) {
    cpp_dec_float_100 n = param->in_out_params->n;
    cpp_dec_float_100 N = param->tgsw_params->tlwe_params->N;
    cpp_dec_float_100 k = param->tgsw_params->tlwe_params->k;
    cpp_dec_float_100 p = msize;

    cpp_dec_float_100 bk_Bg = param->tgsw_params->Bg;
    cpp_dec_float_100 bk_halfBg = param->tgsw_params->halfBg;  
    cpp_dec_float_100 bk_l = param->tgsw_params->l;
    cpp_dec_float_100 bk_stdev = param->tgsw_params->tlwe_params->alpha_min; 
    // static const double bk_stdev = 0; 

    cpp_dec_float_100 ks_Bg = 1<< param->ks_basebit; 
    cpp_dec_float_100 ks_l  = param->ks_t;
    cpp_dec_float_100 ks_stdev = param->in_out_params->alpha_min;

    cpp_dec_float_100 rs_Bg = 1<<RS_param[0];
    cpp_dec_float_100 rs_l = RS_param[1];

    cpp_dec_float_100 pm_Bg = 1<<PM_param[0];
    cpp_dec_float_100 pm_l = PM_param[1];

    cpp_dec_float_100 log2_VMS = log2(n+1) - 4 * log2(3) - 2 * log2(N);
    cpp_dec_float_100 V_MS = pow(2., log2_VMS);
    cpp_dec_float_100 log2_stdMS = log2_VMS/2; 

    cpp_dec_float_100 BR_1 = log2(n) + log2(N) + log2(k+1) + log2(bk_l) + 2*log2(bk_halfBg) + 2*log2(bk_stdev);
    cpp_dec_float_100 BR_2 = log2(n) + log2(1 + k*N) - 2 * bk_l * log2(bk_Bg) - 2;
    cpp_dec_float_100 V_BR = pow(2., BR_1) + pow(2., BR_2);
    cpp_dec_float_100 log2_VBR = log2(V_BR); 
    cpp_dec_float_100 KS_1 = log2(k) + log2(N) + log2(ks_l) + 2 * log2(ks_stdev);
    cpp_dec_float_100 KS_2 = log2(k) -2 * ks_l * log2(ks_Bg) + log2(N) - 2 - log2(3);
    cpp_dec_float_100 V_KS = pow(2., KS_1) + pow(2., KS_2);
    cpp_dec_float_100 log2_VKS = log2(V_KS);
    cpp_dec_float_100 V_BS = V_BR + V_KS;
    cpp_dec_float_100 log2_VBS = log2(V_BS);
    cpp_dec_float_100 RS_1 = log2(n) + log2(rs_l) + 2 * log2(bk_stdev);
    cpp_dec_float_100 RS_2 = -2 * rs_l * log2(rs_Bg) + log2(n) - 2 - log2(3);
    cpp_dec_float_100 V_RS = pow(2., RS_1) + pow(2., RS_2);
    cpp_dec_float_100 log2_VRS = log2(V_RS);
    cpp_dec_float_100 FDFBACC_1 = log2(N) + log2(pm_l) + 2*log2(pm_Bg) + log2(V_RS + V_BS) - 2;
    cpp_dec_float_100 FDFBACC_2 = log2(1 + k*N) - 2*pm_l*log2(pm_Bg)- 2;
    cpp_dec_float_100 V_FDFBACC = pow(2., FDFBACC_1) + pow(2., FDFBACC_2);
    cpp_dec_float_100 log2_VFDFBACC = log2(V_FDFBACC);

    cpp_dec_float_100 V_ct = pow(ks_stdev, 2);
    // cpp_dec_float_100 V_ct = 2*V_BS;
    
    // We assume V_ct = V_TLWE
    cpp_dec_float_100 log2_HDFB = log2(erf(1/(4*p*sqrt(2)*sqrt(V_MS+V_ct)))) + log2(erf(1/(4*p*sqrt(2)*sqrt(V_BS))));
    cpp_dec_float_100 log2err_HDFB = log2(1 - pow(2., log2_HDFB));

    cpp_dec_float_100 log2_FDFB = log2(erf(1/(2*p*sqrt(2)*sqrt(V_ct+V_MS)))) + log2(erf(1/(2*p*sqrt(2)*sqrt(V_FDFBACC + V_BS))));
    cpp_dec_float_100 log2err_FDFB = log2(1 - pow(2., log2_FDFB));

    cpp_dec_float_100 log2_TOTA = log2(erf(1/(2*p*sqrt(2)*sqrt(V_ct + 4*V_MS)))) + log2(erf(1/(2*p*sqrt(2)*sqrt(V_ct + V_BS + 5*V_MS)))) + log2(erf(1/(2*p*sqrt(2)*sqrt(V_BS))));
    cpp_dec_float_100 log2err_TOTA = log2(1 - pow(2., log2_TOTA));

    cpp_dec_float_100 log2_Comp = log2(erf(1/(2*p*sqrt(2)*sqrt(V_ct + V_MS)))) + 2*log2(erf(1/(2*p*sqrt(2)*sqrt(V_BS + V_MS)))) + log2(erf(1/(2*p*sqrt(2)*sqrt(2*V_BS))));
    cpp_dec_float_100 log2err_Comp = log2(1 - pow(2., log2_Comp));

    // std::cout << BOLDWHITE "========================================================== \n" RESET;
    // std::cout << "    Variance of BlindRotate : 2^{" << setprecision(numeric_limits<float>::digits10) << log2_VBR <<"} \n";
    // std::cout << "      Variance of KeySwitch : 2^{" << setprecision(numeric_limits<float>::digits10) << log2_VKS <<"} \n";
    // std::cout << "  Variance of Bootstrapping : 2^{" << setprecision(numeric_limits<float>::digits10) << log2_VBS << "} \n";
    // std::cout << " Variance of LWE-to-TLWE KS : 2^{" << setprecision(numeric_limits<float>::digits10) << log2_VRS << "} \n";
    // std::cout << "        Variance of FDFBACC : 2^{" << setprecision(numeric_limits<float>::digits10) << log2_VFDFBACC << "} \n";
    std::cout << BOLDWHITE "========================================================== \n" RESET;
    std::cout << BOLDYELLOW "    Stdev of Discretization : 2^{" << setprecision(numeric_limits<float>::digits10) << log2_stdMS <<"} \n" RESET;
    std::cout << "       Stdev of BlindRotate : 2^{" << setprecision(numeric_limits<float>::digits10) << log2_VBR/2 <<"} \n";
    std::cout << "         Stdev of KeySwitch : 2^{" << setprecision(numeric_limits<float>::digits10) << log2_VKS/2 <<"} \n";
    std::cout << "     Stdev of Bootstrapping : 2^{" << setprecision(numeric_limits<float>::digits10) << log2_VBS/2 << "} \n";
    std::cout << "   Stdev of TRLWE Keyswitch : 2^{" << setprecision(numeric_limits<float>::digits10) << log2_VRS/2 << "} \n";
    std::cout << "          Stdev of FDFB-ACC : 2^{" << setprecision(numeric_limits<float>::digits10) << log2_VFDFBACC/2 << "} \n";
    std::cout << BOLDWHITE "========================================================== \n" RESET;
    std::cout << "       Precision estimated with 4-sigma (99.9936%) \n";
    std::cout << "    Precision of output after BlindRotate : " << BOLDGREEN << setprecision(numeric_limits<float>::digits10) << (int) floor(32 - (32 + log2_VBR/2 + 3)) << " Bits \n" RESET;
    std::cout << "      Precision of output after Bootstrap : " << BOLDRED << setprecision(numeric_limits<float>::digits10) << (int) floor(32 - (32 + log2_VBS/2 + 3)) << " Bits \n" RESET; 
    std::cout << "                      Desired value for k : " << BOLDCYAN << setprecision(numeric_limits<float>::digits10) << max(0, (int) ceil(log2_stdMS - log2_VBS/2)) << " ~ " << max(0, (int) ceil(log2_stdMS - log2_VBR/2)) << "\n" RESET;
    std::cout << BOLDWHITE "========================================================== \n" RESET;
    std::cout << BOLDWHITE "With Message space p = " << msize << " and extension factor nu = 0, \n" RESET;
    std::cout << BOLDRED "Log2 error rate for HDFB <= " << setprecision(numeric_limits<float>::digits10) << log2err_HDFB << "\n" RESET;
    std::cout << BOLDGREEN "Log2 error rate for FDFB <= " << setprecision(numeric_limits<float>::digits10) << log2err_FDFB << "\n" RESET;
    std::cout << BOLDYELLOW "Log2 error rate for TOTA <= " << setprecision(numeric_limits<float>::digits10) << log2err_TOTA << "\n" RESET;
    std::cout << BOLDBLUE "Log2 error rate for Comp <= " << setprecision(numeric_limits<float>::digits10) << log2err_Comp << "\n" RESET; 
    std::cout << BOLDWHITE "========================================================== \n" RESET;
    FDB_quality_f100(V_ct, V_MS, V_BS, V_FDFBACC, p, 1);
    FDB_quality_f100(V_ct, V_MS, V_BS, V_FDFBACC, p, 2);
    FDB_quality_f100(V_ct, V_MS, V_BS, V_FDFBACC, p, 3);
    FDB_quality_f100(V_ct, V_MS, V_BS, V_FDFBACC, p, 4);
    // FDB_quality_f100(V_ct, V_MS, V_BS, V_FDFBACC, p, 5);
    // FDB_quality_f100(V_ct, V_MS, V_BS, V_FDFBACC, p, 6);
    // FDB_quality_f100(V_ct, V_MS, V_BS, V_FDFBACC, p, 7);
}

void FDB_quality_f100(cpp_dec_float_100 V_ct, cpp_dec_float_100 V_MS, cpp_dec_float_100 V_BS, cpp_dec_float_100 V_FDFBACC, cpp_dec_float_100 p, int32_t ad_bit) {
    cpp_dec_float_100 nu2 = 1<<(2*ad_bit);
    
    cpp_dec_float_100 log2_HDFB = log2(erf(1/(4*p*sqrt(2)*sqrt((V_MS/nu2) + V_ct)))) + log2(erf(1/(4*p*sqrt(2)*sqrt(V_BS))));
    cpp_dec_float_100 log2err_HDFB = log2(1 - pow(2., log2_HDFB));

    cpp_dec_float_100 log2_FDFB = log2(erf(1/(2*p*sqrt(2)*sqrt(V_ct + (V_MS/nu2))))) + log2(erf(1/(2*p*sqrt(2)*sqrt(V_FDFBACC + V_BS))));
    cpp_dec_float_100 log2err_FDFB = log2(1 - pow(2., log2_FDFB));

    cpp_dec_float_100 log2_TOTA = log2(erf(1/(2*p*sqrt(2)*sqrt(V_ct + (4*V_MS/nu2))))) + log2(erf(1/(2*p*sqrt(2)*sqrt(V_ct + V_BS + (5*V_MS/nu2))))) + log2(erf(1/(2*p*sqrt(2)*sqrt(V_BS))));
    cpp_dec_float_100 log2err_TOTA = log2(1 - pow(2., log2_TOTA));

    cpp_dec_float_100 log2_Comp = log2(erf(1/(2*p*sqrt(2)*sqrt(V_ct + (V_MS/nu2))))) + 2*log2(erf(1/(2*p*sqrt(2)*sqrt(V_BS + (V_MS/nu2))))) + log2(erf(1/(2*p*sqrt(2)*sqrt(2*V_BS))));
    cpp_dec_float_100 log2err_Comp = log2(1 - pow(2., log2_Comp));

    std::cout << BOLDWHITE "With Message space p = " << setprecision(numeric_limits<float>::digits10) << p << " and extension factor nu = "<< ad_bit <<",\n" RESET;
    std::cout << BOLDRED "Log2 error rate for HDFB <= " << setprecision(numeric_limits<float>::digits10) << log2err_HDFB << "\n" RESET;
    std::cout << BOLDGREEN "Log2 error rate for FDFB <= " << setprecision(numeric_limits<float>::digits10) << log2err_FDFB << "\n" RESET;
    std::cout << BOLDYELLOW "Log2 error rate for TOTA <= " << setprecision(numeric_limits<float>::digits10) << log2err_TOTA << "\n" RESET;
    std::cout << BOLDBLUE "Log2 error rate for Comp <= " << setprecision(numeric_limits<float>::digits10) << log2err_Comp << "\n" RESET; 
    std::cout << BOLDWHITE "========================================================== \n" RESET;
}

void param_quality_FDB(TFheGateBootstrappingParameterSet* param, int32_t RS_param[2], int32_t PM_param[2], int32_t msize, int32_t ad_bit) {
    cpp_dec_float_100 n = param->in_out_params->n;
    cpp_dec_float_100 N = param->tgsw_params->tlwe_params->N;
    cpp_dec_float_100 k = param->tgsw_params->tlwe_params->k;
    cpp_dec_float_100 p = msize;

    cpp_dec_float_100 bk_Bg = param->tgsw_params->Bg;
    cpp_dec_float_100 bk_halfBg = param->tgsw_params->halfBg;  
    cpp_dec_float_100 bk_l = param->tgsw_params->l;
    cpp_dec_float_100 bk_stdev = param->tgsw_params->tlwe_params->alpha_min; 
    // static const double bk_stdev = 0; 

    cpp_dec_float_100 ks_Bg = 1<< param->ks_basebit; 
    cpp_dec_float_100 ks_l  = param->ks_t;
    cpp_dec_float_100 ks_stdev = param->in_out_params->alpha_min;

    cpp_dec_float_100 rs_Bg = 1<<RS_param[0];
    cpp_dec_float_100 rs_l = RS_param[1];

    cpp_dec_float_100 pm_Bg = 1<<PM_param[0];
    cpp_dec_float_100 pm_l = PM_param[1];

    cpp_dec_float_100 log2_VMS = log2(n+1) - 4 * log2(3) - 2 * log2(N);
    cpp_dec_float_100 V_MS = pow(2., log2_VMS);
    cpp_dec_float_100 log2_stdMS = log2_VMS/2; 

    cpp_dec_float_100 BR_1 = log2(n) + log2(N) + log2(k+1) + log2(bk_l) + 2*log2(bk_halfBg) + 2*log2(bk_stdev);
    cpp_dec_float_100 BR_2 = log2(n) + log2(1 + k*N) - 2 * bk_l * log2(bk_Bg) - 2;
    cpp_dec_float_100 V_BR = pow(2., BR_1) + pow(2., BR_2);
    cpp_dec_float_100 log2_VBR = log2(V_BR); 
    cpp_dec_float_100 KS_1 = log2(k) +log2(N) + log2(ks_l) + 2 * log2(ks_stdev);
    cpp_dec_float_100 KS_2 = log2(k) -2 * ks_l * log2(ks_Bg) + log2(N) - 2 - log2(3);
    cpp_dec_float_100 V_KS = pow(2., KS_1) + pow(2., KS_2);
    cpp_dec_float_100 log2_VKS = log2(V_KS);
    cpp_dec_float_100 V_BS = V_BR + V_KS;
    cpp_dec_float_100 log2_VBS = log2(V_BS);
    cpp_dec_float_100 RS_1 = log2(n) + log2(rs_l) + 2 * log2(bk_stdev);
    cpp_dec_float_100 RS_2 = -2 * rs_l * log2(rs_Bg) + log2(n) - 2 - log2(3);
    cpp_dec_float_100 V_RS = pow(2., RS_1) + pow(2., RS_2);
    cpp_dec_float_100 log2_VRS = log2(V_RS);
    cpp_dec_float_100 FDFBACC_1 = log2(N) + log2(pm_l) + 2*log2(pm_Bg) + log2(V_RS + V_BS) - 2;
    cpp_dec_float_100 FDFBACC_2 = log2(1 + k*N) - 2*pm_l*log2(pm_Bg)- 2;
    cpp_dec_float_100 V_FDFBACC = pow(2., FDFBACC_1) + pow(2., FDFBACC_2);
    cpp_dec_float_100 log2_VFDFBACC = log2(V_FDFBACC);

    cpp_dec_float_100 V_ct = pow(ks_stdev, 2);
    // cpp_dec_float_100 V_ct = 2*V_BS;
    
    // We assume V_ct = V_TLWE

    std::cout << BOLDWHITE "========================================================== \n" RESET;
    FDB_quality_f100(V_ct, V_MS, V_BS, V_FDFBACC, p, ad_bit);
}

void print_result(int32_t msize, int32_t ad_bit, int32_t run, int32_t hdebs, int32_t fdfb, int32_t tota, int32_t comp) {

    std::cout << BOLDWHITE "With Message space p = " <<  msize << " and extension factor nu = "<< ad_bit <<",\n" RESET;
    std::cout << BOLDRED "Wrong Boostrappings in HDFB : " << hdebs << "\n" RESET;
    std::cout << BOLDGREEN "Wrong Boostrappings in FDFB : " <<  fdfb << "\n" RESET;
    std::cout << BOLDYELLOW "Wrong Boostrappings in TOTA : " <<  tota << "\n" RESET;
    std::cout << BOLDBLUE "Wrong Boostrappings in Comp : " << comp << "\n" RESET; 
    std::cout << BOLDWHITE "    During " << run << " executions! \n" RESET; 
    std::cout << BOLDWHITE "========================================================== \n" RESET;    
}

/////////////////////////////////////////////////////////// HDEBS //////////////////////////////////////////////////////////

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

void HDEBS_Rfunceval_woKS_FFT(LweSample *result, const LweBootstrappingKeyFFT *bk, int32_t ad_bit, const LweSample *x, int32_t interval) {

    const TGswParams *bk_params = bk->bk_params;
    const TLweParams *accum_params = bk->accum_params;
    const LweParams *in_params = bk->in_out_params;
    const int32_t N = accum_params->N;
    const int32_t n = in_params->n;
    const int32_t k2 = pow(2, ad_bit);
    const int32_t Nk2 = 2*N*k2;
    const int32_t Nk = N*k2;
    const int32_t halfNk = N*k2/2;
    int32_t int_length = 2*interval;
    double delta = pow(2., 32)/(interval*4);

    TorusPolynomial *testvect = new_TorusPolynomial_array(k2, N);
    int32_t *bara = new int32_t[N];


    int32_t barb = modSwitchFromTorus32(x->b, Nk2); 
    for (int i = 0; i < n; i++){
        bara[i] = modSwitchFromTorus32(x->a[i], Nk2);
    }

    // From 0 -> q-1 
    for (int32_t j = 0; j < N; j++) {
        for (int32_t i = 0; i < k2; i++) {
            double m = int_length * (i + k2*j - halfNk)/(double) Nk;
            // (testvect+i)->coefsT[j] = (Torus32) ((double) delta *((m+50)*(m+7)*(m-50)/(2000))); 
            // (testvect+i)->coefsT[j] = (Torus32) ((double) delta * (43*sin(m*boost::math::double_constants::pi / 4))) + modSwitchToTorus32(1, 4); 
            // (testvect+i)->coefsT[j] = (Torus32) ((double) delta * (40*(exp(m/24) - exp(-m/24))/(exp(m/24)+exp(-m/24)))) + modSwitchToTorus32(1, 4);
            (testvect+i)->coefsT[j] = (Torus32) ((double) delta * (43*sin(m*boost::math::double_constants::pi / 32))) + modSwitchToTorus32(1, 4); 
        }
    }

    // Bootstrapping rotation and extraction
    ext_blindRotateAndExtract_FFT(result, testvect, bk->bkFFT, barb, bara, n, ad_bit, bk_params);


    delete[] bara;
    delete_TorusPolynomial_array(k2, testvect);
}

void HDEBS_R(LweSample *result, const LweBootstrappingKeyFFT *bk, int32_t ad_bit, const LweSample *x, const int32_t msize) {
    LweSample* c1 = new_LweSample(&bk->accum_params->extracted_lweparams);
    LweSample* ctemp = new_LweSample(bk->in_out_params);
    lweCopy(ctemp, x, bk->in_out_params);
    
    HDEBS_Rfunceval_woKS_FFT(c1, bk, ad_bit, ctemp, msize);
    lweKeySwitch(result, bk->ks, c1);

    delete_LweSample(c1);
    delete_LweSample(ctemp);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////// FDFB ///////////////////////////////////////////////////////////

// TLWE -> l_PM TLWEs -> (RKS) l_PM TRLWEs 
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

void FDFB_PubMux(TLweSample* ACC, TLweSample* TGSWp, const LweBootstrappingKeyFFT *bk, int32_t ad_bit, int32_t PM_param[2], int32_t msize, int32_t barb) {
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
    
    
    for (int32_t j = 0; j < N; j++) {
        for (int32_t i = 0; i < k2; i++) {
            int32_t m = (i + k2*j)/N2kM;
            // int32_t m1 = m / sqrt(msize);
            // int32_t m2 = m - m1*sqrt(msize); 
            (tp_1+i)->coefsT[j] = - modSwitchToTorus32(msize/2 + m, msize);
            (tp_0+i)->coefsT[j] = modSwitchToTorus32(m, msize);
        }
    }
    for (int32_t i = 0; i < k2; i++){
        torusPolynomialSubTo(tp_0+i, tp_1+i);
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
    
    delete_TorusPolynomial_array(k2, tp_0);
    delete_TorusPolynomial_array(k2, tp_1);
    delete_TorusPolynomial_array(k2, p_0);
    delete_TorusPolynomial_array(k2, p_1);
}

void FDFB_PubMux_Real(TLweSample* ACC, TLweSample* TGSWp, const LweBootstrappingKeyFFT *bk, int32_t ad_bit, int32_t PM_param[2], int32_t interval, int32_t barb) {
    const TGswParams *bk_params = bk->bk_params;
    const TLweParams *accum_params = bk->accum_params;
    const LweParams *in_params = bk->in_out_params;
    const int32_t N = accum_params->N;
    const int32_t n = in_params->n;
    const int32_t k2 = pow(2, ad_bit);
    const int32_t Nk2 = 2*N*k2;
    const int32_t Nk = N*k2;
    const int32_t base = 1<<PM_param[0];
    const int32_t mask = base - 1;
    const int32_t halfNk = N*k2/2;
    int32_t int_length = 2*interval;
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
    }
    if (barb != 0) torusPolynomialArrayMulByXai(p_0, k2, Nk2 - barb, tp_0);
    if (barb != 0) torusPolynomialArrayMulByXai(p_1, k2, Nk2 - barb, tp_1);
    
    #pragma omp parallel num_threads(min(k2, 16))
    #pragma omp for
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
    
    delete_TorusPolynomial_array(k2, tp_0);
    delete_TorusPolynomial_array(k2, tp_1);
    delete_TorusPolynomial_array(k2, p_0);
    delete_TorusPolynomial_array(k2, p_1);
}

void FDFB_BRnSE(LweSample* result, TLweSample* ACC, const LweBootstrappingKeyFFT *bk, int32_t ad_bit, const LweSample *x) {
    const TGswParams *bk_params = bk->bk_params;
    const TLweParams *accum_params = bk->accum_params;
    const LweParams *in_params = bk->in_out_params;
    const int32_t N = accum_params->N;
    const int32_t n = in_params->n;
    const int32_t k2 = pow(2, ad_bit);
    const int32_t Nk2 = 2*N*k2;
    const int32_t Nk = N*k2;
    const int32_t halfNk = N*k2/2;
    const LweParams *extract_params = &accum_params->extracted_lweparams;

    TorusPolynomial *testvect = new_TorusPolynomial_array(k2, N);
    int32_t *bara = new int32_t[n];

    for (int i = 0; i < n; i++){
        bara[i] = modSwitchFromTorus32(x->a[i], Nk2);
    }    
    ext_blindRotate_FFT(ACC, bk->bkFFT, bara, n, k2, bk_params);
    tLweExtractLweSample(result, ACC+0, extract_params, accum_params);

    delete[] bara;
    delete_TorusPolynomial_array(k2, testvect);

}

void FDFB_EBS_Modular(LweSample *result, const LweBootstrappingKeyFFT *bk, TLweKeySwitchKey* rks,  int32_t ad_bit, const LweSample *x, int32_t PM_param[2], const int32_t msize) {
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
    for (int i = 0; i < k2; i++){
        tLweClear(accum+i, accum_params);
    }
    lweCopy(ctemp, x, bk->in_out_params);
    ctemp->b += modSwitchToTorus32(1, 2*msize);
    int32_t barb = modSwitchFromTorus32(ctemp->b, Nk2);

    FDFB_signeval_FFT(Tgsp, bk, rks, ad_bit, ctemp, PM_param);
    FDFB_PubMux(accum, Tgsp, bk, ad_bit, PM_param, msize, barb);
    FDFB_BRnSE(c1, accum, bk, ad_bit, ctemp);
    lweKeySwitch(result, bk->ks, c1);

    delete_LweSample(c1);
    delete_LweSample(ctemp);
    delete_TLweSample_array(PM_param[1], Tgsp);
    delete_TLweSample_array(k2, accum);
}

void FDFB_EBS_Real(LweSample *result, const LweBootstrappingKeyFFT *bk, TLweKeySwitchKey* rks,  int32_t ad_bit, const LweSample *x, int32_t PM_param[2], int32_t interval) {
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
    for (int i = 0; i < k2; i++){
        tLweClear(accum+i, accum_params);
    }
    lweCopy(ctemp, x, bk->in_out_params);
    int32_t barb = modSwitchFromTorus32(ctemp->b, Nk2);

    FDFB_signeval_FFT(Tgsp, bk, rks, ad_bit, ctemp, PM_param);
    FDFB_PubMux_Real(accum, Tgsp, bk, ad_bit, PM_param, interval, barb);
    FDFB_BRnSE(c1, accum, bk, ad_bit, ctemp);
    lweKeySwitch(result, bk->ks, c1);

    delete_LweSample(c1);
    delete_LweSample(ctemp);
    delete_TLweSample_array(PM_param[1], Tgsp);
    delete_TLweSample_array(k2, accum);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////// COMP ///////////////////////////////////////////////////////////

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

void Comp_funceval(LweSample *result, const LweBootstrappingKeyFFT *bk, TorusPolynomial* testvect, int32_t ad_bit, const LweSample *x, const int32_t msize) {
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

    int32_t *bara = new int32_t[N];


    int32_t barb = modSwitchFromTorus32(ctemp->b, Nk2);
    for (int i = 0; i < n; i++){
        bara[i] = modSwitchFromTorus32(ctemp->a[i], Nk2);
    }

    ext_blindRotateAndExtract_FFT(cks, testvect, bk->bkFFT, barb, bara, n, ad_bit, bk_params);
    lweKeySwitch(result, bk->ks, cks);

    delete[] bara;
    delete_LweSample(ctemp);
    delete_LweSample(cks);
}

void Comp_EBS_Modular(LweSample *result, const LweBootstrappingKeyFFT *bk, int32_t ad_bit, const LweSample *x, const int32_t msize) {
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
    LweSample* cGeh2 = new_LweSample(in_params);

    for (int32_t i = 0; i < k2; i++) {
        for (int32_t j = 0; j < N; j++) {
            int32_t m = (i + k2*j)/N2kM;
            (fpm_LUT+i)->coefsT[j] = modSwitchToTorus32(1 + 2*m, 2*msize);
            (fmm_LUT+i)->coefsT[j] = modSwitchToTorus32(2*msize - 1, 2*msize);

            // (fpm_LUT+i)->coefsT[j] = 0;
            // (fmm_LUT+i)->coefsT[j] = 0;

            // (odd_LUT+i)->coefsT[j] =  (fpm_LUT+i)->coefsT[j] - (fmm_LUT+i)->coefsT[j];
            // (even_LUT+i)->coefsT[j] =  (fpm_LUT+i)->coefsT[j] + (fmm_LUT+i)->coefsT[j];
            (odd_LUT+i)->coefsT[j] =  (fpm_LUT+i)->coefsT[j];
            (even_LUT+i)->coefsT[j] = (fmm_LUT+i)->coefsT[j];
        }
        // torusPolynomialSubTo(odd_LUT+i, fmm_LUT+i);
        // torusPolynomialAddTo(even_LUT+i, fmm_LUT+i);
    }

    // #pragma omp parallel
    // #pragma omp sections
    // {
    //     #pragma omp section
    //     {
    //     Comp_odd_h1(ch1, bk, ad_bit, x, msize);
    //     Comp_funceval(result, bk, odd_LUT, ad_bit, ch1, msize);
    //     }

    //     #pragma omp section
    //     {
    //     Comp_even_h2(ch2, bk, ad_bit, x, msize);
    //     Comp_funceval(cGeh2, bk, even_LUT, ad_bit, ch2, msize);
    //     }
    // }
    Comp_odd_h1(ch1, bk, ad_bit, x, msize);
    Comp_funceval(result, bk, odd_LUT, ad_bit, ch1, msize);
    Comp_even_h2(ch2, bk, ad_bit, x, msize);
    Comp_funceval(cGeh2, bk, even_LUT, ad_bit, ch2, msize);
    lweAddTo(result, cGeh2, in_params);


    delete_LweSample(ch1);
    delete_LweSample(ch2);
    delete_LweSample(cGeh2);
    delete_TorusPolynomial_array(k2, fpm_LUT);
    delete_TorusPolynomial_array(k2, fmm_LUT);
    delete_TorusPolynomial_array(k2, odd_LUT);
    delete_TorusPolynomial_array(k2, even_LUT);
}

void Comp_EBS_Modular_parallel(LweSample *result, const LweBootstrappingKeyFFT *bk, int32_t ad_bit, const LweSample *x, const int32_t msize) {
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
    LweSample* cGeh2 = new_LweSample(in_params);

    for (int32_t i = 0; i < k2; i++) {
        for (int32_t j = 0; j < N; j++) {
            int32_t m = (i + k2*j)/N2kM;
            (fpm_LUT+i)->coefsT[j] = modSwitchToTorus32(1 + 2*m, 2*msize);
            (fmm_LUT+i)->coefsT[j] = modSwitchToTorus32(2*msize - 1, 2*msize);

            // (fpm_LUT+i)->coefsT[j] = 0;
            // (fmm_LUT+i)->coefsT[j] = 0;

            // (odd_LUT+i)->coefsT[j] =  (fpm_LUT+i)->coefsT[j] - (fmm_LUT+i)->coefsT[j];
            // (even_LUT+i)->coefsT[j] =  (fpm_LUT+i)->coefsT[j] + (fmm_LUT+i)->coefsT[j];
            (odd_LUT+i)->coefsT[j] =  (fpm_LUT+i)->coefsT[j];
            (even_LUT+i)->coefsT[j] = (fmm_LUT+i)->coefsT[j];
        }
        // torusPolynomialSubTo(odd_LUT+i, fmm_LUT+i);
        // torusPolynomialAddTo(even_LUT+i, fmm_LUT+i);
    }

    #pragma omp parallel
    #pragma omp sections
    {
        #pragma omp section
        {
        Comp_odd_h1(ch1, bk, ad_bit, x, msize);
        Comp_funceval(result, bk, odd_LUT, ad_bit, ch1, msize);
        }

        #pragma omp section
        {
        Comp_even_h2(ch2, bk, ad_bit, x, msize);
        Comp_funceval(cGeh2, bk, even_LUT, ad_bit, ch2, msize);
        }
    }
    // Comp_odd_h1(ch1, bk, ad_bit, x, msize);
    // Comp_funceval(result, bk, odd_LUT, ad_bit, ch1, msize);
    // Comp_even_h2(ch2, bk, ad_bit, x, msize);
    // Comp_funceval(cGeh2, bk, even_LUT, ad_bit, ch2, msize);
    lweAddTo(result, cGeh2, in_params);


    delete_LweSample(ch1);
    delete_LweSample(ch2);
    delete_LweSample(cGeh2);
    delete_TorusPolynomial_array(k2, fpm_LUT);
    delete_TorusPolynomial_array(k2, fmm_LUT);
    delete_TorusPolynomial_array(k2, odd_LUT);
    delete_TorusPolynomial_array(k2, even_LUT);
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

void Comp_Rfunceval(LweSample *result, const LweBootstrappingKeyFFT *bk, TorusPolynomial* testvect, int32_t ad_bit, const LweSample *x, const int32_t interval) {
    const TGswParams *bk_params = bk->bk_params;
    const TLweParams *accum_params = bk->accum_params;
    const LweParams *in_params = bk->in_out_params;
    const int32_t N = accum_params->N;
    const int32_t n = in_params->n;
    const int32_t k2 = pow(2, ad_bit);
    const int32_t Nk2 = 2*N*k2;
    const int32_t Nk = N*k2;
    LweSample* cks = new_LweSample(&accum_params->extracted_lweparams);
    LweSample* ctemp = new_LweSample(in_params);

    int32_t *bara = new int32_t[N];


    int32_t barb = modSwitchFromTorus32(x->b, Nk2);
    for (int i = 0; i < n; i++){
        bara[i] = modSwitchFromTorus32(x->a[i], Nk2);
    }

    ext_blindRotateAndExtract_FFT(cks, testvect, bk->bkFFT, barb, bara, n, ad_bit, bk_params);
    lweKeySwitch(result, bk->ks, cks);

    delete[] bara;
    delete_LweSample(ctemp);
    delete_LweSample(cks);
}

void Comp_EBS_Real(LweSample *result, const LweBootstrappingKeyFFT *bk, int32_t ad_bit, const LweSample *x, const int32_t interval) {
    const TGswParams *bk_params = bk->bk_params;
    const TLweParams *accum_params = bk->accum_params;
    const LweParams *in_params = bk->in_out_params;
    const int32_t N = accum_params->N;
    const int32_t n = in_params->n;
    const int32_t k2 = pow(2, ad_bit);
    const int32_t Nk2 = 2*N*k2;
    const int32_t Nk = N*k2;  
    const int32_t int_length = 2*interval;
    double delta = pow(2., 32)/(interval*4);

    TorusPolynomial* fpm_LUT = new_TorusPolynomial_array(k2, N); // f(m)
    TorusPolynomial* fmm_LUT = new_TorusPolynomial_array(k2, N); // f(-m)
    TorusPolynomial* odd_LUT = new_TorusPolynomial_array(k2, N);
    TorusPolynomial* even_LUT = new_TorusPolynomial_array(k2, N);

    LweSample* ch1 = new_LweSample(in_params);
    LweSample* ch2 = new_LweSample(in_params);
    LweSample* cGeh2 = new_LweSample(in_params);

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

    // #pragma omp parallel
    // #pragma omp sections
    // {
    //     #pragma omp section
    //     {
    //     Comp_odd_h1(ch1, bk, ad_bit, x, msize);
    //     Comp_funceval(result, bk, odd_LUT, ad_bit, ch1, msize);
    //     }

    //     #pragma omp section
    //     {
    //     Comp_even_h2(ch2, bk, ad_bit, x, msize);
    //     Comp_funceval(cGeh2, bk, even_LUT, ad_bit, ch2, msize);
    //     }
    // }
    Comp_odd_Rh1(ch1, bk, ad_bit, x, interval);
    Comp_Rfunceval(result, bk, odd_LUT, ad_bit, ch1, interval);
    Comp_even_Rh2(ch2, bk, ad_bit, x, interval);
    Comp_Rfunceval(cGeh2, bk, even_LUT, ad_bit, ch2, interval);
    lweAddTo(result, cGeh2, in_params);


    delete_LweSample(ch1);
    delete_LweSample(ch2);
    delete_LweSample(cGeh2);
    delete_TorusPolynomial_array(k2, fpm_LUT);
    delete_TorusPolynomial_array(k2, fmm_LUT);
    delete_TorusPolynomial_array(k2, odd_LUT);
    delete_TorusPolynomial_array(k2, even_LUT);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////