#include "tlwekeyswitch.h"
#include <tfhe/lagrangehalfc_arithmetic.h>
#include <tfhe/polynomials.h>
#include <tfhe/polynomials_arithmetic.h>
#include <tfhe/tlwe_functions.h>

using namespace std;

TLweKeySwitchKey::TLweKeySwitchKey(int32_t n, int32_t t, int32_t basebit, const TLweParams* tlwe_params, TLweSample* ks0_raw){
    this->basebit=basebit;
    this->tlwe_params=tlwe_params; 
    this->n=n;
    this->t=t;
    this->base=1<<basebit;
    this->ks0_raw = ks0_raw;
    ks1_raw = new TLweSample*[n*t];
    ks = new TLweSample**[n];

   
    for (int32_t p = 0; p < n*t; ++p)
	    ks1_raw[p] = ks0_raw + base*p;
	for (int32_t p = 0; p < n; ++p)
	    ks[p] = ks1_raw + t*p;
}

TLweKeySwitchKey::~TLweKeySwitchKey() {
    delete[] ks1_raw;
    delete[] ks;
}

TLweKeySwitchKey* alloc_TLweKeySwitchKey() {
    return (TLweKeySwitchKey*) malloc(sizeof(TLweKeySwitchKey));
}

TLweKeySwitchKey* alloc_TLweKeySwitchKey_array(int32_t nbelts) {
    return (TLweKeySwitchKey*) malloc(nbelts*sizeof(TLweKeySwitchKey));
}


void free_TLweKeySwitchKey(TLweKeySwitchKey* ptr) {
    free(ptr);
}

void free_TLweKeySwitchKey_array(int32_t nbelts, TLweKeySwitchKey* ptr) {
    free(ptr);
}


void init_TLweKeySwitchKey(TLweKeySwitchKey* obj, int32_t n, int32_t t, int32_t basebit, const TLweParams* tlwe_params) {
    const int32_t base=1<<basebit;
    TLweSample* ks0_raw = new_TLweSample_array(n*t*base, tlwe_params);

    new(obj) TLweKeySwitchKey(n,t,basebit, tlwe_params, ks0_raw);
}

void destroy_TLweKeySwitchKey(TLweKeySwitchKey* obj) {
    const int32_t n = obj->n;
    const int32_t t = obj->t;
    const int32_t base = obj->base;
    delete_TLweSample_array(n*t*base,obj->ks0_raw);

    obj->~TLweKeySwitchKey();
}

void init_TLweKeySwitchKey_array(int32_t nbelts, TLweKeySwitchKey* obj, int32_t n, int32_t t, int32_t basebit, const TLweParams* tlwe_params) {
    for (int32_t i=0; i<nbelts; i++) {
        init_TLweKeySwitchKey(obj+i, n, t, basebit, tlwe_params);
    }
}

void destroy_TLweKeySwitchKey_array(int32_t nbelts, TLweKeySwitchKey* obj) {
    for (int32_t i=0; i<nbelts; i++) {
        destroy_TLweKeySwitchKey(obj+i);
    }
}

TLweKeySwitchKey* new_TLweKeySwitchKey(int32_t n, int32_t t, int32_t basebit, const TLweParams* tlwe_params) {
    TLweKeySwitchKey* obj = alloc_TLweKeySwitchKey();
    init_TLweKeySwitchKey(obj, n,t,basebit, tlwe_params);
    return obj;
}

TLweKeySwitchKey* new_TLweKeySwitchKey_array(int32_t nbelts, int32_t n, int32_t t, int32_t basebit, const TLweParams* tlwe_params) {
    TLweKeySwitchKey* obj = alloc_TLweKeySwitchKey_array(nbelts);
    init_TLweKeySwitchKey_array(nbelts, obj, n,t,basebit, tlwe_params);
    return obj;
}

void delete_TLweKeySwitchKey(TLweKeySwitchKey* obj) {
    destroy_TLweKeySwitchKey(obj);
    free_TLweKeySwitchKey(obj);
}

void delete_TLweKeySwitchKey_array(int32_t nbelts, TLweKeySwitchKey* obj) {
    destroy_TLweKeySwitchKey_array(nbelts,obj);
    free_TLweKeySwitchKey_array(nbelts,obj);
}

void TLweCreateKeySwitchKey(TLweKeySwitchKey* result, const LweKey* in_key, const TLweKey* out_key) {
    const int32_t n = result->n;
    const int32_t t = result->t;
    const int32_t N = out_key->params->N;
    const int32_t k = out_key->params->k;
    const int32_t basebit = result->basebit;
    const int32_t base = 1<<basebit;
    const double alpha = out_key->params->alpha_min;
    const int32_t sizeks = n*t*(base-1);
    //const int32_t n_out = out_key->params->n;
    TorusPolynomial* mu = new_TorusPolynomial(N);
    torusPolynomialClear(mu);

    // generate the ks
    int32_t index = 0; 
    for (int32_t i = 0; i < n; ++i) {
        for (int32_t j = 0; j < t; ++j) {

            // term h=0 as trivial encryption of 0 (it will not be used in the KeySwitching)
            tLweNoiselessTrivial(&result->ks[i][j][0], mu, out_key->params);

            for (int32_t h = 1; h < base; ++h) { // pas le terme en 0
                Torus32 mess = (in_key->key[i]*h)*(1<<(32-(j+1)*basebit));
                tLweSymEncryptT(&result->ks[i][j][h], mess, alpha, out_key);
                index += 1;
            }
        }
    }
}

// Single TLWE -> TRLWE KeySwitch 
void TLweKeySwitchTranslate_fromArray(TLweSample* result, const LweSample* target,
    const TLweSample*** t_ks, const TLweParams* tlwe_params,
    const int32_t n, const int32_t t, const int32_t basebit) {

    const int32_t base=1<<basebit;       // base=2 in [CGGI16]
    const int32_t prec_offset=1<<(32-(1+basebit*t)); //precision
    const int32_t mask=base-1;

    TLweSample* tp = new_TLweSample(tlwe_params);

    for (int32_t i = 0; i < n; i++){
        const uint32_t aibar = target->a[i] + prec_offset;
        for (int32_t j = 0; j < t; j++) {
            const uint32_t aij = (aibar>>(32-(j+1)*basebit)) & mask;
            if(aij != 0) {tLweSubTo(result, &t_ks[i][j][aij], tlwe_params);}
        }
    }
}

void TLweKeySwitch(TLweSample* result, const TLweKeySwitchKey* t_ks, const LweSample* sample) {
    const TLweParams* tlwe_params = t_ks->tlwe_params;
    const int32_t n = t_ks->n;
    const int32_t basebit = t_ks->basebit;
    const int32_t t = t_ks->t;
    const int32_t N = t_ks->tlwe_params->N;
    tLweClear(result, tlwe_params);
    result->b->coefsT[0] = sample->b;
    TLweKeySwitchTranslate_fromArray(result, sample, (const TLweSample***) t_ks->ks, tlwe_params, n, t, basebit);
}


