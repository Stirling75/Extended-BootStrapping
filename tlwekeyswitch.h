#ifndef TLWEKEYSWITCH_H
#define TLWEKEYSWITCH_H

#include <tfhe/tfhe_core.h>
#include <tfhe/tfhe.h>
#include <tfhe/lweparams.h>
#include <tfhe/tlwe_functions.h>
#include <tfhe/tlwe.h>
#include <iostream>
#include <time.h>
#include <math.h>

using namespace std;

struct TLweKeySwitchKey {
    int32_t n; ///< length of the input key: s'
    int32_t t; ///< decomposition length
    int32_t basebit; ///< log_2(base)
    int32_t base; ///< decomposition base: a power of 2 
    const TLweParams* tlwe_params; ///< params of the output key s 
    TLweSample* ks0_raw; //tableau qui contient tout les Lwe samples de taille nlbase
    TLweSample** ks1_raw;// de taille nl  pointe vers un tableau ks0_raw dont les cases sont espaceés de base positions
    TLweSample*** ks; ///< the keyswitch elements: a n.l.base matrix
    // de taille n pointe vers ks1 un tableau dont les cases sont espaceés de ell positions

#ifdef __cplusplus
    TLweKeySwitchKey(int32_t n, int32_t t, int32_t basebit, const TLweParams* tlwe_params, TLweSample* ks0_raw);
    ~TLweKeySwitchKey();
    TLweKeySwitchKey(const TLweKeySwitchKey&) = delete;
    void operator=(const TLweKeySwitchKey&) = delete;
#endif
};


TLweKeySwitchKey* alloc_TLweKeySwitchKey();
TLweKeySwitchKey* alloc_TLweKeySwitchKey_array(int32_t nbelts);

void free_TLweKeySwitchKey(TLweKeySwitchKey* ptr);
void free_TLweKeySwitchKey_array(int32_t nbelts, TLweKeySwitchKey* ptr);

void init_TLweKeySwitchKey(TLweKeySwitchKey* obj, int32_t n, int32_t t, int32_t basebit, const TLweParams* tlwe_params);
void destroy_TLweKeySwitchKey(TLweKeySwitchKey* obj);

void init_TLweKeySwitchKey_array(int32_t nbelts, TLweKeySwitchKey* obj, int32_t n, int32_t t, int32_t basebit, const TLweParams* tlwe_params);
void destroy_TLweKeySwitchKey_array(int32_t nbelts, TLweKeySwitchKey* obj);

TLweKeySwitchKey* new_TLweKeySwitchKey(int32_t n, int32_t t, int32_t basebit, const TLweParams* tlwe_params);
TLweKeySwitchKey* new_TLweKeySwitchKey_array(int32_t nbelts, int32_t n, int32_t t, int32_t basebit, const TLweParams* tlwe_params);

void delete_TLweKeySwitchKey(TLweKeySwitchKey* obj);
void delete_TLweKeySwitchKey_array(int32_t nbelts, TLweKeySwitchKey* obj);

void TLweCreateKeySwitchKey(TLweKeySwitchKey* result, const LweKey* in_key, const TLweKey* out_key);
void TLweKeySwitchTranslate_fromArray(TLweSample* result, const LweSample* target,
    const TLweSample*** t_ks, const TLweParams* tlwe_params,
    const int32_t n, const int32_t t, const int32_t basebit);

void TLweKeySwitch(TLweSample* result, const TLweKeySwitchKey* t_ks, const LweSample* sample);


#endif