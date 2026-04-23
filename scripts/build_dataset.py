#!/usr/bin/env python3
"""
Build a labeled dataset of crypto vs non-crypto Linux ELF binaries.

Strategy:
- Compile ~50 crypto C programs (using OpenSSL, custom AES/SHA/MD5/DES/RC4/RSA implementations)
- Compile ~50 non-crypto C programs (sorting, string ops, math, file I/O, data structures)
- Each program is compiled with multiple flags (-O0, -O2, -Os, -static, -pie, stripped/unstripped)
  giving ~6 variants per source = ~600 total binaries
- Label: 1 = uses cryptographic algorithm, 0 = no crypto
"""

import os
import subprocess
import json
import shutil

CRYPTO_DIR = "/app/sources/crypto"
NONCRYPTO_DIR = "/app/sources/noncrypto"
BIN_DIR = "/app/binaries"
os.makedirs(CRYPTO_DIR, exist_ok=True)
os.makedirs(NONCRYPTO_DIR, exist_ok=True)
os.makedirs(BIN_DIR, exist_ok=True)

# ============================================================
# CRYPTO SOURCE FILES — programs that use crypto algorithms
# ============================================================

crypto_sources = {}

# --- OpenSSL-linked programs ---
crypto_sources["openssl_aes_encrypt.c"] = r"""
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <string.h>
#include <stdio.h>
int main() {
    unsigned char key[32], iv[16], plaintext[1024], ciphertext[1040];
    int len, ciphertext_len;
    RAND_bytes(key, 32);
    RAND_bytes(iv, 16);
    memset(plaintext, 'A', 1024);
    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    EVP_EncryptInit_ex(ctx, EVP_aes_256_cbc(), NULL, key, iv);
    EVP_EncryptUpdate(ctx, ciphertext, &len, plaintext, 1024);
    ciphertext_len = len;
    EVP_EncryptFinal_ex(ctx, ciphertext + len, &len);
    ciphertext_len += len;
    EVP_CIPHER_CTX_free(ctx);
    printf("AES-256-CBC encrypted %d bytes to %d bytes\n", 1024, ciphertext_len);
    return 0;
}
"""

crypto_sources["openssl_sha256.c"] = r"""
#include <openssl/evp.h>
#include <stdio.h>
#include <string.h>
int main() {
    const char *msg = "Hello, World! This is a SHA-256 test message.";
    unsigned char hash[32];
    unsigned int len;
    EVP_MD_CTX *ctx = EVP_MD_CTX_new();
    EVP_DigestInit_ex(ctx, EVP_sha256(), NULL);
    EVP_DigestUpdate(ctx, msg, strlen(msg));
    EVP_DigestFinal_ex(ctx, hash, &len);
    EVP_MD_CTX_free(ctx);
    printf("SHA-256: ");
    for(int i=0; i<32; i++) printf("%02x", hash[i]);
    printf("\n");
    return 0;
}
"""

crypto_sources["openssl_rsa.c"] = r"""
#include <openssl/rsa.h>
#include <openssl/evp.h>
#include <openssl/pem.h>
#include <stdio.h>
#include <string.h>
int main() {
    EVP_PKEY *pkey = EVP_PKEY_new();
    EVP_PKEY_CTX *ctx = EVP_PKEY_CTX_new_id(EVP_PKEY_RSA, NULL);
    EVP_PKEY_keygen_init(ctx);
    EVP_PKEY_CTX_set_rsa_keygen_bits(ctx, 2048);
    EVP_PKEY_keygen(ctx, &pkey);
    EVP_PKEY_CTX_free(ctx);
    printf("RSA-2048 key generated\n");
    EVP_PKEY_free(pkey);
    return 0;
}
"""

crypto_sources["openssl_hmac.c"] = r"""
#include <openssl/hmac.h>
#include <openssl/evp.h>
#include <stdio.h>
#include <string.h>
int main() {
    const char *key = "my_secret_key";
    const char *data = "message to authenticate";
    unsigned char result[EVP_MAX_MD_SIZE];
    unsigned int len;
    HMAC(EVP_sha256(), key, strlen(key), 
         (unsigned char*)data, strlen(data), result, &len);
    printf("HMAC-SHA256: ");
    for(unsigned int i=0; i<len; i++) printf("%02x", result[i]);
    printf("\n");
    return 0;
}
"""

crypto_sources["openssl_md5.c"] = r"""
#include <openssl/evp.h>
#include <stdio.h>
#include <string.h>
int main() {
    const char *msg = "MD5 hash test string for crypto detection";
    unsigned char hash[16];
    unsigned int len;
    EVP_MD_CTX *ctx = EVP_MD_CTX_new();
    EVP_DigestInit_ex(ctx, EVP_md5(), NULL);
    EVP_DigestUpdate(ctx, msg, strlen(msg));
    EVP_DigestFinal_ex(ctx, hash, &len);
    EVP_MD_CTX_free(ctx);
    printf("MD5: ");
    for(int i=0; i<16; i++) printf("%02x", hash[i]);
    printf("\n");
    return 0;
}
"""

crypto_sources["openssl_sha512.c"] = r"""
#include <openssl/evp.h>
#include <stdio.h>
#include <string.h>
int main() {
    const char *msg = "SHA-512 test";
    unsigned char hash[64];
    unsigned int len;
    EVP_MD_CTX *ctx = EVP_MD_CTX_new();
    EVP_DigestInit_ex(ctx, EVP_sha512(), NULL);
    EVP_DigestUpdate(ctx, msg, strlen(msg));
    EVP_DigestFinal_ex(ctx, hash, &len);
    EVP_MD_CTX_free(ctx);
    printf("SHA-512: ");
    for(int i=0; i<64; i++) printf("%02x", hash[i]);
    printf("\n");
    return 0;
}
"""

crypto_sources["openssl_chacha20.c"] = r"""
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <stdio.h>
#include <string.h>
int main() {
    unsigned char key[32], nonce[12], plaintext[256], ciphertext[272];
    int len;
    RAND_bytes(key, 32);
    RAND_bytes(nonce, 12);
    memset(plaintext, 'X', 256);
    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    EVP_EncryptInit_ex(ctx, EVP_chacha20_poly1305(), NULL, key, nonce);
    EVP_EncryptUpdate(ctx, ciphertext, &len, plaintext, 256);
    EVP_EncryptFinal_ex(ctx, ciphertext + len, &len);
    EVP_CIPHER_CTX_free(ctx);
    printf("ChaCha20-Poly1305 encrypted\n");
    return 0;
}
"""

crypto_sources["openssl_ecdsa.c"] = r"""
#include <openssl/evp.h>
#include <openssl/ec.h>
#include <openssl/ecdsa.h>
#include <stdio.h>
#include <string.h>
int main() {
    EVP_PKEY *pkey = NULL;
    EVP_PKEY_CTX *ctx = EVP_PKEY_CTX_new_id(EVP_PKEY_EC, NULL);
    EVP_PKEY_keygen_init(ctx);
    EVP_PKEY_CTX_set_ec_paramgen_curve_nid(ctx, NID_X9_62_prime256v1);
    EVP_PKEY_keygen(ctx, &pkey);
    EVP_PKEY_CTX_free(ctx);
    // Sign
    const char *msg = "ECDSA test message";
    EVP_MD_CTX *mdctx = EVP_MD_CTX_new();
    EVP_DigestSignInit(mdctx, NULL, EVP_sha256(), NULL, pkey);
    EVP_DigestSignUpdate(mdctx, msg, strlen(msg));
    size_t siglen;
    EVP_DigestSignFinal(mdctx, NULL, &siglen);
    unsigned char sig[512];
    EVP_DigestSignFinal(mdctx, sig, &siglen);
    EVP_MD_CTX_free(mdctx);
    printf("ECDSA signature: %zu bytes\n", siglen);
    EVP_PKEY_free(pkey);
    return 0;
}
"""

crypto_sources["openssl_pbkdf2.c"] = r"""
#include <openssl/evp.h>
#include <stdio.h>
#include <string.h>
int main() {
    const char *password = "my_password_123";
    unsigned char salt[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
    unsigned char key[32];
    PKCS5_PBKDF2_HMAC(password, strlen(password), salt, 16, 100000, EVP_sha256(), 32, key);
    printf("PBKDF2-derived key: ");
    for(int i=0; i<32; i++) printf("%02x", key[i]);
    printf("\n");
    return 0;
}
"""

crypto_sources["openssl_aes_gcm.c"] = r"""
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <stdio.h>
#include <string.h>
int main() {
    unsigned char key[16], iv[12], plaintext[512], ciphertext[528], tag[16];
    int len;
    RAND_bytes(key, 16);
    RAND_bytes(iv, 12);
    memset(plaintext, 'B', 512);
    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    EVP_EncryptInit_ex(ctx, EVP_aes_128_gcm(), NULL, NULL, NULL);
    EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_SET_IVLEN, 12, NULL);
    EVP_EncryptInit_ex(ctx, NULL, NULL, key, iv);
    EVP_EncryptUpdate(ctx, ciphertext, &len, plaintext, 512);
    EVP_EncryptFinal_ex(ctx, ciphertext + len, &len);
    EVP_CIPHER_CTX_ctrl(ctx, EVP_CTRL_GCM_GET_TAG, 16, tag);
    EVP_CIPHER_CTX_free(ctx);
    printf("AES-128-GCM encrypted\n");
    return 0;
}
"""

# --- Custom / Embedded crypto implementations (no OpenSSL) ---
crypto_sources["custom_aes_sbox.c"] = r"""
#include <stdio.h>
#include <string.h>
#include <stdint.h>
// AES S-box (full)
static const uint8_t sbox[256] = {
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
};
static const uint8_t rcon[10] = {0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x1b,0x36};

void sub_bytes(uint8_t state[16]) {
    for(int i=0; i<16; i++) state[i] = sbox[state[i]];
}
void shift_rows(uint8_t s[16]) {
    uint8_t t;
    t=s[1]; s[1]=s[5]; s[5]=s[9]; s[9]=s[13]; s[13]=t;
    t=s[2]; s[2]=s[10]; s[10]=t; t=s[6]; s[6]=s[14]; s[14]=t;
    t=s[15]; s[15]=s[11]; s[11]=s[7]; s[7]=s[3]; s[3]=t;
}
uint8_t xtime(uint8_t x) { return (x<<1) ^ (((x>>7)&1) * 0x1b); }
void mix_columns(uint8_t s[16]) {
    for(int i=0; i<4; i++) {
        uint8_t a=s[4*i], b=s[4*i+1], c=s[4*i+2], d=s[4*i+3];
        uint8_t e=a^b^c^d;
        s[4*i]^=e^xtime(a^b); s[4*i+1]^=e^xtime(b^c);
        s[4*i+2]^=e^xtime(c^d); s[4*i+3]^=e^xtime(d^a);
    }
}
void add_round_key(uint8_t s[16], const uint8_t rk[16]) {
    for(int i=0; i<16; i++) s[i]^=rk[i];
}
int main() {
    uint8_t key[16]={0x2b,0x7e,0x15,0x16,0x28,0xae,0xd2,0xa6,0xab,0xf7,0x15,0x88,0x09,0xcf,0x4f,0x3c};
    uint8_t pt[16]={0x32,0x43,0xf6,0xa8,0x88,0x5a,0x30,0x8d,0x31,0x31,0x98,0xa2,0xe0,0x37,0x07,0x34};
    uint8_t state[16];
    memcpy(state, pt, 16);
    add_round_key(state, key);
    for(int r=0; r<9; r++) { sub_bytes(state); shift_rows(state); mix_columns(state); add_round_key(state, key); }
    sub_bytes(state); shift_rows(state); add_round_key(state, key);
    printf("AES output: ");
    for(int i=0; i<16; i++) printf("%02x", state[i]);
    printf("\n");
    return 0;
}
"""

crypto_sources["custom_sha256.c"] = r"""
#include <stdio.h>
#include <string.h>
#include <stdint.h>
static const uint32_t K[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};
#define ROTR(x,n) (((x)>>(n))|((x)<<(32-(n))))
#define CH(x,y,z) (((x)&(y))^((~(x))&(z)))
#define MAJ(x,y,z) (((x)&(y))^((x)&(z))^((y)&(z)))
#define EP0(x) (ROTR(x,2)^ROTR(x,13)^ROTR(x,22))
#define EP1(x) (ROTR(x,6)^ROTR(x,11)^ROTR(x,25))
#define SIG0(x) (ROTR(x,7)^ROTR(x,18)^((x)>>3))
#define SIG1(x) (ROTR(x,17)^ROTR(x,19)^((x)>>10))
void sha256(const uint8_t *data, size_t len, uint8_t hash[32]) {
    uint32_t h[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    // simplified: just process first block
    uint32_t w[64]={0};
    for(int i=0;i<16&&i*4<(int)len;i++) {
        w[i]=(data[i*4]<<24)|(data[i*4+1]<<16)|(data[i*4+2]<<8)|data[i*4+3];
    }
    for(int i=16;i<64;i++) w[i]=SIG1(w[i-2])+w[i-7]+SIG0(w[i-15])+w[i-16];
    uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
    for(int i=0;i<64;i++){
        uint32_t t1=hh+EP1(e)+CH(e,f,g)+K[i]+w[i];
        uint32_t t2=EP0(a)+MAJ(a,b,c);
        hh=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
    }
    h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=hh;
    for(int i=0;i<8;i++){hash[i*4]=h[i]>>24;hash[i*4+1]=h[i]>>16;hash[i*4+2]=h[i]>>8;hash[i*4+3]=h[i];}
}
int main() {
    const char *msg = "abc";
    uint8_t hash[32];
    sha256((const uint8_t*)msg, 3, hash);
    printf("SHA-256('abc')=");
    for(int i=0;i<32;i++) printf("%02x",hash[i]);
    printf("\n");
    return 0;
}
"""

crypto_sources["custom_md5.c"] = r"""
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#define F(x,y,z) (((x)&(y))|((~(x))&(z)))
#define G(x,y,z) (((x)&(z))|((y)&(~(z))))
#define H(x,y,z) ((x)^(y)^(z))
#define I(x,y,z) ((y)^((x)|(~(z))))
#define ROTL(x,n) (((x)<<(n))|((x)>>(32-(n))))
static const uint32_t T[64] = {
    0xd76aa478,0xe8c7b756,0x242070db,0xc1bdceee,0xf57c0faf,0x4787c62a,0xa8304613,0xfd469501,
    0x698098d8,0x8b44f7af,0xffff5bb1,0x895cd7be,0x6b901122,0xfd987193,0xa679438e,0x49b40821,
    0xf61e2562,0xc040b340,0x265e5a51,0xe9b6c7aa,0xd62f105d,0x02441453,0xd8a1e681,0xe7d3fbc8,
    0x21e1cde6,0xc33707d6,0xf4d50d87,0x455a14ed,0xa9e3e905,0xfcefa3f8,0x676f02d9,0x8d2a4c8a,
    0xfffa3942,0x8771f681,0x6d9d6122,0xfde5380c,0xa4beea44,0x4bdecfa9,0xf6bb4b60,0xbebfbc70,
    0x289b7ec6,0xeaa127fa,0xd4ef3085,0x04881d05,0xd9d4d039,0xe6db99e5,0x1fa27cf8,0xc4ac5665,
    0xf4292244,0x432aff97,0xab9423a7,0xfc93a039,0x655b59c3,0x8f0ccc92,0xffeff47d,0x85845dd1,
    0x6fa87e4f,0xfe2ce6e0,0xa3014314,0x4e0811a1,0xf7537e82,0xbd3af235,0x2ad7d2bb,0xeb86d391
};
void md5_simple(const uint8_t *msg, size_t len, uint8_t digest[16]) {
    uint32_t a0=0x67452301, b0=0xefcdab89, c0=0x98badcfe, d0=0x10325476;
    uint8_t block[64] = {0};
    memcpy(block, msg, len < 55 ? len : 55);
    block[len] = 0x80;
    uint64_t bits = len * 8;
    memcpy(block+56, &bits, 8);
    uint32_t *M = (uint32_t*)block;
    uint32_t A=a0,B=b0,C=c0,D=d0;
    for(int i=0;i<64;i++){
        uint32_t Func,g;
        if(i<16){Func=F(B,C,D);g=i;}
        else if(i<32){Func=G(B,C,D);g=(5*i+1)%16;}
        else if(i<48){Func=H(B,C,D);g=(3*i+5)%16;}
        else{Func=I(B,C,D);g=(7*i)%16;}
        uint32_t temp=D; D=C; C=B;
        static const int s[64]={7,12,17,22,7,12,17,22,7,12,17,22,7,12,17,22,
                                5,9,14,20,5,9,14,20,5,9,14,20,5,9,14,20,
                                4,11,16,23,4,11,16,23,4,11,16,23,4,11,16,23,
                                6,10,15,21,6,10,15,21,6,10,15,21,6,10,15,21};
        B=B+ROTL(A+Func+T[i]+M[g],s[i]);
        A=temp;
    }
    a0+=A;b0+=B;c0+=C;d0+=D;
    memcpy(digest,&a0,4);memcpy(digest+4,&b0,4);memcpy(digest+8,&c0,4);memcpy(digest+12,&d0,4);
}
int main(){
    uint8_t d[16];
    md5_simple((uint8_t*)"hello",5,d);
    printf("MD5: ");for(int i=0;i<16;i++)printf("%02x",d[i]);printf("\n");
    return 0;
}
"""

crypto_sources["custom_rc4.c"] = r"""
#include <stdio.h>
#include <stdint.h>
#include <string.h>
typedef struct { uint8_t S[256]; int i,j; } RC4_CTX;
void rc4_init(RC4_CTX *ctx, const uint8_t *key, int keylen) {
    for(int i=0;i<256;i++) ctx->S[i]=i;
    int j=0;
    for(int i=0;i<256;i++){
        j=(j+ctx->S[i]+key[i%keylen])&0xff;
        uint8_t t=ctx->S[i];ctx->S[i]=ctx->S[j];ctx->S[j]=t;
    }
    ctx->i=ctx->j=0;
}
void rc4_crypt(RC4_CTX *ctx, uint8_t *data, int len) {
    for(int n=0;n<len;n++){
        ctx->i=(ctx->i+1)&0xff;
        ctx->j=(ctx->j+ctx->S[ctx->i])&0xff;
        uint8_t t=ctx->S[ctx->i];ctx->S[ctx->i]=ctx->S[ctx->j];ctx->S[ctx->j]=t;
        data[n]^=ctx->S[(ctx->S[ctx->i]+ctx->S[ctx->j])&0xff];
    }
}
int main(){
    RC4_CTX ctx;
    uint8_t key[]="SecretKey";
    uint8_t data[]="Hello, RC4 stream cipher!";
    int len=strlen((char*)data);
    rc4_init(&ctx,key,9);
    rc4_crypt(&ctx,data,len);
    printf("RC4 encrypted: ");for(int i=0;i<len;i++)printf("%02x",data[i]);printf("\n");
    return 0;
}
"""

crypto_sources["custom_des.c"] = r"""
#include <stdio.h>
#include <stdint.h>
#include <string.h>
// DES initial permutation table (subset for structure)
static const int IP[64]={
    58,50,42,34,26,18,10,2,60,52,44,36,28,20,12,4,
    62,54,46,38,30,22,14,6,64,56,48,40,32,24,16,8,
    57,49,41,33,25,17,9,1,59,51,43,35,27,19,11,3,
    61,53,45,37,29,21,13,5,63,55,47,39,31,23,15,7
};
static const int E[48]={
    32,1,2,3,4,5,4,5,6,7,8,9,8,9,10,11,12,13,
    12,13,14,15,16,17,16,17,18,19,20,21,20,21,22,23,24,25,
    24,25,26,27,28,29,28,29,30,31,32,1
};
// DES S-boxes (S1 only for demo)
static const uint8_t SBOX1[4][16]={
    {14,4,13,1,2,15,11,8,3,10,6,12,5,9,0,7},
    {0,15,7,4,14,2,13,1,10,6,12,11,9,5,3,8},
    {4,1,14,8,13,6,2,11,15,12,9,7,3,10,5,0},
    {15,12,8,2,4,9,1,7,5,11,3,14,10,0,6,13}
};
void des_demo(uint8_t block[8], uint8_t key[8]) {
    // Simplified DES-like round for structure
    uint8_t L[4], R[4];
    memcpy(L, block, 4);
    memcpy(R, block+4, 4);
    for(int round=0; round<16; round++) {
        uint8_t temp[4];
        memcpy(temp, R, 4);
        for(int i=0;i<4;i++) R[i] = L[i] ^ SBOX1[round%4][R[i]%16] ^ key[i];
        memcpy(L, temp, 4);
    }
    memcpy(block, L, 4);
    memcpy(block+4, R, 4);
}
int main(){
    uint8_t block[8]={0x01,0x23,0x45,0x67,0x89,0xAB,0xCD,0xEF};
    uint8_t key[8]={0x13,0x34,0x57,0x79,0x9B,0xBC,0xDF,0xF1};
    des_demo(block,key);
    printf("DES output: ");for(int i=0;i<8;i++)printf("%02x",block[i]);printf("\n");
    return 0;
}
"""

crypto_sources["custom_sha1.c"] = r"""
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#define ROTL32(x,n) (((x)<<(n))|((x)>>(32-(n))))
void sha1_block(uint32_t h[5], const uint8_t block[64]) {
    uint32_t w[80];
    for(int i=0;i<16;i++) w[i]=(block[4*i]<<24)|(block[4*i+1]<<16)|(block[4*i+2]<<8)|block[4*i+3];
    for(int i=16;i<80;i++) w[i]=ROTL32(w[i-3]^w[i-8]^w[i-14]^w[i-16],1);
    uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4];
    for(int i=0;i<80;i++){
        uint32_t f,k;
        if(i<20){f=(b&c)|((~b)&d);k=0x5A827999;}
        else if(i<40){f=b^c^d;k=0x6ED9EBA1;}
        else if(i<60){f=(b&c)|(b&d)|(c&d);k=0x8F1BBCDC;}
        else{f=b^c^d;k=0xCA62C1D6;}
        uint32_t t=ROTL32(a,5)+f+e+k+w[i];e=d;d=c;c=ROTL32(b,30);b=a;a=t;
    }
    h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;
}
int main(){
    uint32_t h[5]={0x67452301,0xEFCDAB89,0x98BADCFE,0x10325476,0xC3D2E1F0};
    uint8_t block[64]={0};
    const char *msg="abc";
    memcpy(block,msg,3);block[3]=0x80;block[62]=0;block[63]=24;
    sha1_block(h,block);
    printf("SHA-1: ");for(int i=0;i<5;i++)printf("%08x",h[i]);printf("\n");
    return 0;
}
"""

crypto_sources["custom_chacha20.c"] = r"""
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#define QR(a,b,c,d) (a+=b,d^=a,d=((d)<<16)|((d)>>16), \
    c+=d,b^=c,b=((b)<<12)|((b)>>20), \
    a+=b,d^=a,d=((d)<<8)|((d)>>24), \
    c+=d,b^=c,b=((b)<<7)|((b)>>25))
void chacha20_block(uint32_t out[16], const uint32_t in[16]) {
    uint32_t x[16]; memcpy(x, in, 64);
    for(int i=0; i<10; i++) {
        QR(x[0],x[4],x[8],x[12]); QR(x[1],x[5],x[9],x[13]);
        QR(x[2],x[6],x[10],x[14]); QR(x[3],x[7],x[11],x[15]);
        QR(x[0],x[5],x[10],x[15]); QR(x[1],x[6],x[11],x[12]);
        QR(x[2],x[7],x[8],x[13]); QR(x[3],x[4],x[9],x[14]);
    }
    for(int i=0;i<16;i++) out[i]=x[i]+in[i];
}
int main(){
    uint32_t state[16]={0x61707865,0x3320646e,0x79622d32,0x6b206574,
        1,2,3,4,5,6,7,8,0,0,0x09000000,0x4a000000};
    uint32_t out[16];
    chacha20_block(out, state);
    printf("ChaCha20 block: ");for(int i=0;i<16;i++)printf("%08x ",out[i]);printf("\n");
    return 0;
}
"""

crypto_sources["custom_blowfish.c"] = r"""
#include <stdio.h>
#include <stdint.h>
#include <string.h>
// Blowfish P-array init values (subset)
static uint32_t P[18]={
    0x243F6A88,0x85A308D3,0x13198A2E,0x03707344,0xA4093822,0x299F31D0,
    0x082EFA98,0xEC4E6C89,0x452821E6,0x38D01377,0xBE5466CF,0x34E90C6C,
    0xC0AC29B7,0xC97C50DD,0x3F84D5B5,0xB5470917,0x9216D5D9,0x8979FB1B
};
// S-box 0 (first 16 values for demo)
static uint32_t S0[256];
void bf_init(const uint8_t *key, int keylen) {
    for(int i=0;i<256;i++) S0[i]=i*0x01010101;
    for(int i=0;i<18;i++) P[i]^=((uint32_t)key[i%keylen]<<24)|((uint32_t)key[(i+1)%keylen]<<16)|
        ((uint32_t)key[(i+2)%keylen]<<8)|key[(i+3)%keylen];
}
uint32_t bf_f(uint32_t x) {
    return S0[(x>>24)&0xff]+S0[(x>>16)&0xff]^S0[(x>>8)&0xff]+S0[x&0xff];
}
void bf_encrypt(uint32_t *L, uint32_t *R) {
    for(int i=0;i<16;i+=2){ *L^=P[i]; *R^=bf_f(*L)^P[i+1]; uint32_t t=*L;*L=*R;*R=t; }
    uint32_t t=*L;*L=*R;*R=t; *R^=P[16]; *L^=P[17];
}
int main(){
    bf_init((uint8_t*)"TestKey!",8);
    uint32_t L=0xFEDCBA98,R=0x76543210;
    bf_encrypt(&L,&R);
    printf("Blowfish: %08x%08x\n",L,R);
    return 0;
}
"""

crypto_sources["custom_xor_cipher.c"] = r"""
#include <stdio.h>
#include <string.h>
#include <stdint.h>
// XOR cipher with key expansion (simple but still crypto)
void xor_encrypt(uint8_t *data, int len, const uint8_t *key, int keylen) {
    for(int i=0; i<len; i++) {
        data[i] ^= key[i % keylen];
        data[i] = (data[i] << 3) | (data[i] >> 5); // rotate
        data[i] ^= (uint8_t)(i * 0x37 + 0x42);
    }
}
int main() {
    uint8_t data[] = "Sensitive data to encrypt with XOR cipher";
    uint8_t key[] = "MySecretKey123";
    int len = strlen((char*)data);
    xor_encrypt(data, len, key, strlen((char*)key));
    printf("XOR cipher output: ");
    for(int i=0; i<len; i++) printf("%02x", data[i]);
    printf("\n");
    return 0;
}
"""

crypto_sources["crypto_keygen.c"] = r"""
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <openssl/bn.h>
#include <stdio.h>
int main() {
    // Generate random key material
    unsigned char key[32];
    RAND_bytes(key, 32);
    printf("Random key: ");
    for(int i=0; i<32; i++) printf("%02x", key[i]);
    printf("\n");
    // Generate DH params
    EVP_PKEY *params = NULL;
    EVP_PKEY_CTX *pctx = EVP_PKEY_CTX_new_id(EVP_PKEY_DH, NULL);
    EVP_PKEY_paramgen_init(pctx);
    EVP_PKEY_CTX_set_dh_paramgen_prime_len(pctx, 1024);
    printf("DH parameter generation initialized\n");
    EVP_PKEY_CTX_free(pctx);
    return 0;
}
"""

crypto_sources["tls_client.c"] = r"""
#include <openssl/ssl.h>
#include <openssl/err.h>
#include <stdio.h>
int main() {
    SSL_library_init();
    SSL_load_error_strings();
    OpenSSL_add_all_algorithms();
    const SSL_METHOD *method = TLS_client_method();
    SSL_CTX *ctx = SSL_CTX_new(method);
    if(!ctx) { printf("SSL_CTX creation failed\n"); return 1; }
    printf("TLS client context created with method: %s\n", SSL_get_version(SSL_new(ctx)));
    SSL_CTX_free(ctx);
    EVP_cleanup();
    return 0;
}
"""

# ============================================================
# NON-CRYPTO SOURCE FILES — no cryptographic operations
# ============================================================

noncrypto_sources = {}

noncrypto_sources["bubble_sort.c"] = r"""
#include <stdio.h>
void bubble_sort(int arr[], int n) {
    for(int i=0; i<n-1; i++)
        for(int j=0; j<n-i-1; j++)
            if(arr[j]>arr[j+1]) { int t=arr[j]; arr[j]=arr[j+1]; arr[j+1]=t; }
}
int main() {
    int arr[] = {64, 34, 25, 12, 22, 11, 90, 1, 55, 33};
    int n = 10;
    bubble_sort(arr, n);
    printf("Sorted: ");
    for(int i=0; i<n; i++) printf("%d ", arr[i]);
    printf("\n");
    return 0;
}
"""

noncrypto_sources["quicksort.c"] = r"""
#include <stdio.h>
void swap(int *a, int *b) { int t=*a; *a=*b; *b=t; }
int partition(int arr[], int low, int high) {
    int pivot=arr[high], i=low-1;
    for(int j=low; j<high; j++)
        if(arr[j]<pivot) { i++; swap(&arr[i],&arr[j]); }
    swap(&arr[i+1],&arr[high]);
    return i+1;
}
void quicksort(int arr[], int low, int high) {
    if(low<high) {
        int pi=partition(arr,low,high);
        quicksort(arr,low,pi-1);
        quicksort(arr,pi+1,high);
    }
}
int main() {
    int arr[]={10,7,8,9,1,5,3,6,4,2,15,20,13,11,17};
    int n=15;
    quicksort(arr,0,n-1);
    for(int i=0;i<n;i++) printf("%d ",arr[i]);
    printf("\n");
    return 0;
}
"""

noncrypto_sources["mergesort.c"] = r"""
#include <stdio.h>
#include <stdlib.h>
void merge(int arr[], int l, int m, int r) {
    int n1=m-l+1, n2=r-m;
    int *L=malloc(n1*sizeof(int)), *R=malloc(n2*sizeof(int));
    for(int i=0;i<n1;i++) L[i]=arr[l+i];
    for(int j=0;j<n2;j++) R[j]=arr[m+1+j];
    int i=0,j=0,k=l;
    while(i<n1&&j<n2) arr[k++]=(L[i]<=R[j])?L[i++]:R[j++];
    while(i<n1) arr[k++]=L[i++];
    while(j<n2) arr[k++]=R[j++];
    free(L); free(R);
}
void mergesort(int arr[], int l, int r) {
    if(l<r) { int m=l+(r-l)/2; mergesort(arr,l,m); mergesort(arr,m+1,r); merge(arr,l,m,r); }
}
int main() {
    int arr[]={38,27,43,3,9,82,10};
    int n=7;
    mergesort(arr,0,n-1);
    for(int i=0;i<n;i++) printf("%d ",arr[i]);
    printf("\n");
    return 0;
}
"""

noncrypto_sources["linked_list.c"] = r"""
#include <stdio.h>
#include <stdlib.h>
typedef struct Node { int data; struct Node *next; } Node;
Node* insert(Node *head, int val) {
    Node *n = malloc(sizeof(Node));
    n->data = val; n->next = head;
    return n;
}
void print_list(Node *head) {
    while(head) { printf("%d -> ", head->data); head = head->next; }
    printf("NULL\n");
}
void free_list(Node *head) {
    while(head) { Node *t=head; head=head->next; free(t); }
}
int main() {
    Node *head = NULL;
    for(int i=10; i>=1; i--) head = insert(head, i);
    print_list(head);
    free_list(head);
    return 0;
}
"""

noncrypto_sources["binary_tree.c"] = r"""
#include <stdio.h>
#include <stdlib.h>
typedef struct Node { int key; struct Node *left, *right; } Node;
Node* newNode(int k) { Node *n=malloc(sizeof(Node)); n->key=k; n->left=n->right=NULL; return n; }
Node* insert(Node *root, int k) {
    if(!root) return newNode(k);
    if(k<root->key) root->left=insert(root->left,k);
    else if(k>root->key) root->right=insert(root->right,k);
    return root;
}
void inorder(Node *root) { if(root) { inorder(root->left); printf("%d ",root->key); inorder(root->right); } }
int main() {
    Node *root=NULL;
    int keys[]={50,30,20,40,70,60,80};
    for(int i=0;i<7;i++) root=insert(root,keys[i]);
    printf("Inorder: "); inorder(root); printf("\n");
    return 0;
}
"""

noncrypto_sources["matrix_mult.c"] = r"""
#include <stdio.h>
#define N 4
void multiply(int A[N][N], int B[N][N], int C[N][N]) {
    for(int i=0;i<N;i++) for(int j=0;j<N;j++) {
        C[i][j]=0;
        for(int k=0;k<N;k++) C[i][j]+=A[i][k]*B[k][j];
    }
}
int main() {
    int A[N][N]={{1,2,3,4},{5,6,7,8},{9,10,11,12},{13,14,15,16}};
    int B[N][N]={{16,15,14,13},{12,11,10,9},{8,7,6,5},{4,3,2,1}};
    int C[N][N];
    multiply(A,B,C);
    for(int i=0;i<N;i++){for(int j=0;j<N;j++)printf("%4d ",C[i][j]);printf("\n");}
    return 0;
}
"""

noncrypto_sources["string_ops.c"] = r"""
#include <stdio.h>
#include <string.h>
#include <ctype.h>
int count_words(const char *s) {
    int c=0, in=0;
    while(*s) { if(isspace(*s)) in=0; else if(!in) { in=1; c++; } s++; }
    return c;
}
void reverse(char *s) {
    int n=strlen(s);
    for(int i=0;i<n/2;i++) { char t=s[i]; s[i]=s[n-1-i]; s[n-1-i]=t; }
}
int is_palindrome(const char *s) {
    int n=strlen(s);
    for(int i=0;i<n/2;i++) if(tolower(s[i])!=tolower(s[n-1-i])) return 0;
    return 1;
}
int main() {
    char s[]="Hello World from C programming";
    printf("Words: %d\n", count_words(s));
    char s2[]="racecar";
    printf("%s palindrome: %s\n", s2, is_palindrome(s2)?"yes":"no");
    reverse(s);
    printf("Reversed: %s\n", s);
    return 0;
}
"""

noncrypto_sources["fibonacci.c"] = r"""
#include <stdio.h>
long long fib_iter(int n) {
    if(n<=1) return n;
    long long a=0,b=1;
    for(int i=2;i<=n;i++) { long long t=a+b; a=b; b=t; }
    return b;
}
long long fib_rec(int n) { return n<=1?n:fib_rec(n-1)+fib_rec(n-2); }
int main() {
    printf("Iterative fib(30)=%lld\n", fib_iter(30));
    printf("Recursive fib(20)=%lld\n", fib_rec(20));
    for(int i=0;i<20;i++) printf("%lld ", fib_iter(i));
    printf("\n");
    return 0;
}
"""

noncrypto_sources["file_copy.c"] = r"""
#include <stdio.h>
#include <stdlib.h>
int main() {
    FILE *src = tmpfile(), *dst = tmpfile();
    if(!src||!dst) { printf("tmpfile failed\n"); return 1; }
    const char *data = "This is test data for file copy operations.\nLine 2\nLine 3\n";
    fputs(data, src);
    rewind(src);
    char buf[1024];
    size_t n;
    while((n=fread(buf,1,sizeof(buf),src))>0) fwrite(buf,1,n,dst);
    rewind(dst);
    while(fgets(buf,sizeof(buf),dst)) printf("%s", buf);
    fclose(src); fclose(dst);
    return 0;
}
"""

noncrypto_sources["calculator.c"] = r"""
#include <stdio.h>
#include <math.h>
double add(double a, double b) { return a+b; }
double sub(double a, double b) { return a-b; }
double mul(double a, double b) { return a*b; }
double divide(double a, double b) { return b!=0?a/b:0; }
double power(double a, double b) { return pow(a,b); }
int main() {
    printf("10+5=%.1f\n", add(10,5));
    printf("10-5=%.1f\n", sub(10,5));
    printf("10*5=%.1f\n", mul(10,5));
    printf("10/3=%.4f\n", divide(10,3));
    printf("2^10=%.0f\n", power(2,10));
    printf("sqrt(144)=%.0f\n", sqrt(144));
    printf("sin(pi/4)=%.4f\n", sin(M_PI/4));
    return 0;
}
"""

noncrypto_sources["hashtable.c"] = r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define TABLE_SIZE 64
typedef struct Entry { char *key; int value; struct Entry *next; } Entry;
Entry *table[TABLE_SIZE];
unsigned hash(const char *key) {
    unsigned h=0;
    while(*key) h = h*31 + *key++;
    return h % TABLE_SIZE;
}
void put(const char *key, int value) {
    unsigned idx = hash(key);
    Entry *e = malloc(sizeof(Entry));
    e->key = strdup(key);
    e->value = value;
    e->next = table[idx];
    table[idx] = e;
}
int get(const char *key) {
    unsigned idx = hash(key);
    for(Entry *e=table[idx]; e; e=e->next)
        if(strcmp(e->key, key)==0) return e->value;
    return -1;
}
int main() {
    put("apple", 1); put("banana", 2); put("cherry", 3);
    put("date", 4); put("elderberry", 5);
    printf("apple=%d banana=%d cherry=%d\n", get("apple"), get("banana"), get("cherry"));
    return 0;
}
"""

noncrypto_sources["prime_sieve.c"] = r"""
#include <stdio.h>
#include <string.h>
#define MAX 10000
int sieve[MAX];
void eratosthenes() {
    memset(sieve, 1, sizeof(sieve));
    sieve[0]=sieve[1]=0;
    for(int i=2;i*i<MAX;i++)
        if(sieve[i]) for(int j=i*i;j<MAX;j+=i) sieve[j]=0;
}
int main() {
    eratosthenes();
    int count=0;
    for(int i=2;i<MAX;i++) if(sieve[i]) count++;
    printf("Primes < %d: %d\n", MAX, count);
    printf("First 20: ");
    int c=0;
    for(int i=2;i<MAX&&c<20;i++) if(sieve[i]) { printf("%d ",i); c++; }
    printf("\n");
    return 0;
}
"""

noncrypto_sources["stack.c"] = r"""
#include <stdio.h>
#include <stdlib.h>
#define MAX 100
typedef struct { int data[MAX]; int top; } Stack;
void init(Stack *s) { s->top=-1; }
int push(Stack *s, int v) { if(s->top>=MAX-1) return 0; s->data[++s->top]=v; return 1; }
int pop(Stack *s, int *v) { if(s->top<0) return 0; *v=s->data[s->top--]; return 1; }
int peek(Stack *s) { return s->top>=0?s->data[s->top]:-1; }
int main() {
    Stack s; init(&s);
    for(int i=1;i<=10;i++) push(&s,i*i);
    printf("Stack (top to bottom): ");
    int v;
    while(pop(&s,&v)) printf("%d ",v);
    printf("\n");
    return 0;
}
"""

noncrypto_sources["graph_bfs.c"] = r"""
#include <stdio.h>
#define V 6
int adj[V][V]={{0,1,1,0,0,0},{1,0,0,1,1,0},{1,0,0,0,1,0},{0,1,0,0,0,1},{0,1,1,0,0,1},{0,0,0,1,1,0}};
void bfs(int start) {
    int visited[V]={0}, queue[V], front=0, rear=0;
    visited[start]=1; queue[rear++]=start;
    printf("BFS from %d: ", start);
    while(front<rear) {
        int node=queue[front++];
        printf("%d ", node);
        for(int i=0;i<V;i++) if(adj[node][i]&&!visited[i]) { visited[i]=1; queue[rear++]=i; }
    }
    printf("\n");
}
int main() { bfs(0); bfs(3); return 0; }
"""

noncrypto_sources["json_parser.c"] = r"""
#include <stdio.h>
#include <string.h>
#include <ctype.h>
// Simple JSON-like key extractor
void extract_keys(const char *json) {
    int in_key=0;
    const char *p=json;
    while(*p) {
        if(*p=='"') {
            if(!in_key) { in_key=1; printf("Key: "); }
            else { in_key=0; printf("\n"); while(*p && *p!=':' && *p!=',') p++; }
        } else if(in_key) putchar(*p);
        p++;
    }
}
int main() {
    const char *json = "{\"name\":\"Alice\",\"age\":30,\"city\":\"NYC\",\"score\":95.5}";
    printf("Parsing: %s\n", json);
    extract_keys(json);
    return 0;
}
"""

noncrypto_sources["compression_rle.c"] = r"""
#include <stdio.h>
#include <string.h>
int rle_encode(const char *in, char *out, int maxout) {
    int oi=0, n=strlen(in);
    for(int i=0; i<n && oi<maxout-3;) {
        char c=in[i]; int count=1;
        while(i+count<n && in[i+count]==c && count<255) count++;
        out[oi++]=c; out[oi++]='0'+count/10; out[oi++]='0'+count%10;
        i+=count;
    }
    out[oi]=0;
    return oi;
}
int main() {
    const char *data = "AAABBBCCCCDDDDDEEEEE";
    char encoded[256];
    rle_encode(data, encoded, 256);
    printf("Input:   %s\n", data);
    printf("Encoded: %s\n", encoded);
    return 0;
}
"""

noncrypto_sources["regex_simple.c"] = r"""
#include <stdio.h>
#include <string.h>
int match_star(char c, const char *regex, const char *text);
int match_here(const char *regex, const char *text) {
    if(regex[0]=='\0') return 1;
    if(regex[1]=='*') return match_star(regex[0], regex+2, text);
    if(regex[0]=='$' && regex[1]=='\0') return *text=='\0';
    if(*text!='\0' && (regex[0]=='.' || regex[0]==*text)) return match_here(regex+1, text+1);
    return 0;
}
int match_star(char c, const char *regex, const char *text) {
    do { if(match_here(regex, text)) return 1; } while(*text!='\0' && (*text++==c || c=='.'));
    return 0;
}
int match(const char *regex, const char *text) {
    if(regex[0]=='^') return match_here(regex+1, text);
    do { if(match_here(regex, text)) return 1; } while(*text++!='\0');
    return 0;
}
int main() {
    printf("match 'ab*c' in 'abbbbc': %d\n", match("ab*c", "abbbbc"));
    printf("match '^hello' in 'hello world': %d\n", match("^hello", "hello world"));
    printf("match 'x.z' in 'xyz': %d\n", match("x.z", "xyz"));
    return 0;
}
"""

noncrypto_sources["memory_pool.c"] = r"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define POOL_SIZE 4096
#define BLOCK_SIZE 64
typedef struct { char pool[POOL_SIZE]; int used[POOL_SIZE/BLOCK_SIZE]; int count; } MemPool;
void pool_init(MemPool *p) { memset(p->used,0,sizeof(p->used)); p->count=POOL_SIZE/BLOCK_SIZE; }
void* pool_alloc(MemPool *p) {
    for(int i=0;i<p->count;i++) if(!p->used[i]) { p->used[i]=1; return p->pool+i*BLOCK_SIZE; }
    return NULL;
}
void pool_free(MemPool *p, void *ptr) {
    int idx=((char*)ptr-p->pool)/BLOCK_SIZE;
    if(idx>=0&&idx<p->count) p->used[idx]=0;
}
int main() {
    MemPool pool; pool_init(&pool);
    void *ptrs[10];
    for(int i=0;i<10;i++) { ptrs[i]=pool_alloc(&pool); printf("Alloc %d: %p\n",i,ptrs[i]); }
    for(int i=0;i<10;i+=2) pool_free(&pool,ptrs[i]);
    printf("Freed even blocks, re-allocating...\n");
    for(int i=0;i<5;i++) printf("Re-alloc: %p\n", pool_alloc(&pool));
    return 0;
}
"""

noncrypto_sources["image_process.c"] = r"""
#include <stdio.h>
#include <stdlib.h>
#define W 100
#define H 100
unsigned char img[H][W];
void fill(unsigned char val) { for(int i=0;i<H;i++) for(int j=0;j<W;j++) img[i][j]=val; }
void draw_rect(int x,int y,int w,int h,unsigned char val) {
    for(int i=y;i<y+h&&i<H;i++) for(int j=x;j<x+w&&j<W;j++) img[i][j]=val;
}
double avg_brightness() {
    long sum=0;
    for(int i=0;i<H;i++) for(int j=0;j<W;j++) sum+=img[i][j];
    return (double)sum/(H*W);
}
void threshold(unsigned char t) {
    for(int i=0;i<H;i++) for(int j=0;j<W;j++) img[i][j]=img[i][j]>t?255:0;
}
int main() {
    fill(128);
    draw_rect(10,10,30,30,255);
    draw_rect(50,50,40,40,0);
    printf("Avg brightness: %.1f\n", avg_brightness());
    threshold(128);
    printf("After threshold: %.1f\n", avg_brightness());
    return 0;
}
"""

noncrypto_sources["signal_process.c"] = r"""
#include <stdio.h>
#include <math.h>
#define N 256
double signal_data[N];
void generate_signal() {
    for(int i=0;i<N;i++)
        signal_data[i] = sin(2*M_PI*5*i/N) + 0.5*sin(2*M_PI*12*i/N) + 0.3*cos(2*M_PI*20*i/N);
}
void low_pass_filter(double *data, int n, int window) {
    double temp[N];
    for(int i=0;i<n;i++) {
        double sum=0; int count=0;
        for(int j=-window/2;j<=window/2;j++) {
            int idx=i+j;
            if(idx>=0&&idx<n) { sum+=data[idx]; count++; }
        }
        temp[i]=sum/count;
    }
    for(int i=0;i<n;i++) data[i]=temp[i];
}
double rms(double *data, int n) {
    double sum=0;
    for(int i=0;i<n;i++) sum+=data[i]*data[i];
    return sqrt(sum/n);
}
int main() {
    generate_signal();
    printf("RMS before filter: %.4f\n", rms(signal_data, N));
    low_pass_filter(signal_data, N, 5);
    printf("RMS after filter:  %.4f\n", rms(signal_data, N));
    return 0;
}
"""

# Write all source files
for name, code in crypto_sources.items():
    with open(os.path.join(CRYPTO_DIR, name), "w") as f:
        f.write(code)

for name, code in noncrypto_sources.items():
    with open(os.path.join(NONCRYPTO_DIR, name), "w") as f:
        f.write(code)

print(f"Written {len(crypto_sources)} crypto sources")
print(f"Written {len(noncrypto_sources)} non-crypto sources")

# ============================================================
# COMPILE WITH MULTIPLE FLAGS → generate binary variants
# ============================================================

compile_configs = [
    {"suffix": "O0",       "flags": "-O0"},
    {"suffix": "O2",       "flags": "-O2"},
    {"suffix": "Os",       "flags": "-Os"},
    {"suffix": "O3",       "flags": "-O3"},
    {"suffix": "O0_pie",   "flags": "-O0 -fpie -pie"},
    {"suffix": "O2_strip", "flags": "-O2 -s"},
    {"suffix": "O0_static","flags": "-O0 -static"},
    {"suffix": "Os_strip", "flags": "-Os -s"},
]

metadata = []
compiled = 0
failed = 0

def compile_source(src_path, name_base, label, configs):
    global compiled, failed
    results = []
    for cfg in configs:
        out_name = f"{name_base}_{cfg['suffix']}"
        out_path = os.path.join(BIN_DIR, out_name)
        
        # Determine link flags
        link_flags = ""
        if label == 1:  # crypto — may need OpenSSL
            # Check if it uses OpenSSL headers
            with open(src_path) as f:
                content = f.read()
            if "openssl/" in content:
                link_flags = "-lssl -lcrypto"
            if "math.h" in content:
                link_flags += " -lm"
        else:
            with open(src_path) as f:
                content = f.read()
            if "math.h" in content:
                link_flags = "-lm"
        
        # Skip static builds for OpenSSL (requires static libs)
        if "static" in cfg["suffix"] and "openssl/" in content:
            continue
            
        cmd = f"gcc {cfg['flags']} -o {out_path} {src_path} {link_flags} 2>/dev/null"
        ret = subprocess.run(cmd, shell=True, capture_output=True)
        
        if ret.returncode == 0:
            results.append({
                "binary_path": out_path,
                "binary_name": out_name,
                "source": os.path.basename(src_path),
                "label": label,
                "label_name": "crypto" if label == 1 else "non_crypto",
                "opt_level": cfg["suffix"],
                "compile_flags": cfg["flags"],
            })
            compiled += 1
        else:
            failed += 1
    return results

# Compile crypto sources
for name in sorted(crypto_sources.keys()):
    src = os.path.join(CRYPTO_DIR, name)
    base = name.replace(".c", "")
    metadata.extend(compile_source(src, base, 1, compile_configs))

# Compile non-crypto sources
for name in sorted(noncrypto_sources.keys()):
    src = os.path.join(NONCRYPTO_DIR, name)
    base = name.replace(".c", "")
    metadata.extend(compile_source(src, base, 0, compile_configs))

print(f"\nCompiled: {compiled} binaries ({failed} failed)")
print(f"Crypto: {sum(1 for m in metadata if m['label']==1)}")
print(f"Non-crypto: {sum(1 for m in metadata if m['label']==0)}")

# Save metadata
with open("/app/binary_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\nMetadata saved to /app/binary_metadata.json")
